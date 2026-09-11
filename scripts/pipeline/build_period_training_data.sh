#!/usr/bin/env bash
#
# Reliable wrapper for building period training data on a standing
# Dataproc cluster.
#
# - Restarts the cluster once before the pipeline.
# - Retries only known Py4J / driver gateway failures.
# - Fails immediately on application/data errors.
# - Uses Dataproc status.details if gcloud hides the traceback.
# - Restarts the cluster before the final retry.
# - Uses bash script.sh so executable bits do not matter.
#

set -euo pipefail


SOURCE="${1:-raw}"

CLUSTER="${DATAPROC_CLUSTER:-instacart-dataproc-cluster-fc45ebb3}"
REGION="${DATAPROC_REGION:-europe-west1}"

MAX_ATTEMPTS="${DATAPROC_JOB_RETRIES:-3}"
RETRY_DELAY_SECONDS="${DATAPROC_RETRY_DELAY_SECONDS:-30}"

SKIP_CLUSTER_RESTART="${SKIP_CLUSTER_RESTART:-0}"

CLUSTER_WAIT_INTERVAL_SECONDS=15
CLUSTER_WAIT_TIMEOUT_SECONDS=900

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"


get_cluster_state() {
    gcloud dataproc clusters describe "${CLUSTER}" \
        --region="${REGION}" \
        --format='value(status.state)'
}


wait_for_cluster_state() {
    local target_state=$1
    local elapsed=0
    local state

    while (( elapsed < CLUSTER_WAIT_TIMEOUT_SECONDS )); do
        state="$(get_cluster_state)"

        echo "Cluster state: ${state}"

        if [[ "${state}" == "${target_state}" ]]; then
            return 0
        fi

        if [[ "${state}" == "ERROR" ]]; then
            echo "Cluster entered ERROR state." >&2
            return 1
        fi

        sleep "${CLUSTER_WAIT_INTERVAL_SECONDS}"
        elapsed=$((elapsed + CLUSTER_WAIT_INTERVAL_SECONDS))
    done

    echo \
        "Timed out waiting for cluster to reach ${target_state}." \
        >&2

    return 1
}


restart_cluster() {
    local state

    state="$(get_cluster_state)"

    echo "Restarting Dataproc cluster ${CLUSTER}"
    echo "Current state: ${state}"

    case "${state}" in
        RUNNING)
            gcloud --quiet dataproc clusters stop "${CLUSTER}" \
                --region="${REGION}"

            wait_for_cluster_state "STOPPED"
            ;;

        STOPPING)
            wait_for_cluster_state "STOPPED"
            ;;

        STOPPED)
            ;;

        STARTING | CREATING | UPDATING)
            wait_for_cluster_state "RUNNING"

            gcloud --quiet dataproc clusters stop "${CLUSTER}" \
                --region="${REGION}"

            wait_for_cluster_state "STOPPED"
            ;;

        *)
            echo \
                "Cannot restart cluster from state ${state}." \
                >&2

            return 1
            ;;
    esac

    gcloud --quiet dataproc clusters start "${CLUSTER}" \
        --region="${REGION}"

    wait_for_cluster_state "RUNNING"

    echo "Cluster ${CLUSTER} is ready."
}


is_retryable_failure() {
    local log_file=$1

    grep -Eiq \
        -e 'Py4JError' \
        -e 'applyModifiableSettings' \
        -e 'Target Object ID does not exist' \
        -e 'Py4JNetworkError' \
        -e 'Answer from Java side is empty' \
        -e 'Java gateway process exited' \
        -e 'gateway.*(died|disconnected|closed)' \
        -e 'driver.*(lost|disconnected|exited unexpectedly)' \
        "${log_file}"
}


extract_job_id() {
    local log_file=$1

    grep -Eo 'Job \[[^]]+\]' "${log_file}" \
        | tail -n 1 \
        | sed -E 's/Job \[([^]]+)\]/\1/' \
        || true
}


append_job_details() {
    local log_file=$1
    local job_id
    local details

    job_id="$(extract_job_id "${log_file}")"

    if [[ -z "${job_id}" ]]; then
        return 0
    fi

    echo "Checking Dataproc status.details for job ${job_id}"

    details="$(
        gcloud dataproc jobs describe "${job_id}" \
            --region="${REGION}" \
            --format='value(status.details)' \
            2>/dev/null \
            || true
    )"

    if [[ -n "${details}" ]]; then
        printf '\n%s\n' "${details}" >> "${log_file}"
    fi
}


run_with_retry() {
    local attempt=1
    local exit_code
    local log_file

    log_file="$(mktemp)"

    while true; do
        : > "${log_file}"

        echo
        echo "Attempt ${attempt}/${MAX_ATTEMPTS}: $*"

        set +e

        "$@" 2>&1 | tee "${log_file}"

        exit_code="${PIPESTATUS[0]}"

        set -e

        if (( exit_code == 0 )); then
            rm -f "${log_file}"
            return 0
        fi


        if ! is_retryable_failure "${log_file}"; then
            append_job_details "${log_file}"
        fi

        if ! is_retryable_failure "${log_file}"; then
            echo \
                "Non-retryable application/data failure. Stopping." \
                >&2

            rm -f "${log_file}"
            return "${exit_code}"
        fi

        echo "Transient Spark/Py4J failure detected."

        if (( attempt >= MAX_ATTEMPTS )); then
            echo \
                "Failed after ${MAX_ATTEMPTS} attempts." \
                >&2

            rm -f "${log_file}"
            return "${exit_code}"
        fi

        if (( attempt == MAX_ATTEMPTS - 1 )); then
            echo "Restarting cluster before final retry."
            restart_cluster
        else
            echo "Retrying in ${RETRY_DELAY_SECONDS}s."
            sleep "${RETRY_DELAY_SECONDS}"
        fi

        attempt=$((attempt + 1))
    done
}


build_features() {
    local source=$1
    local snapshot=$2
    local mode=$3

    echo
    echo \
        "Building features: source=${source}, snapshot=${snapshot}, mode=${mode}"

    run_with_retry bash \
        "${ROOT_DIR}/scripts/feature/create_user_data.sh" \
        "${source}" "${snapshot}" "${mode}"

    run_with_retry bash \
        "${ROOT_DIR}/scripts/feature/create_user_product_count_data.sh" \
        "${source}" "${snapshot}" "${mode}"

    run_with_retry bash \
        "${ROOT_DIR}/scripts/feature/create_product_history_data.sh" \
        "${source}" "${snapshot}" "${mode}"

    run_with_retry bash \
        "${ROOT_DIR}/scripts/feature/create_aisle_history_data.sh" \
        "${source}" "${snapshot}" "${mode}"

    run_with_retry bash \
        "${ROOT_DIR}/scripts/feature/create_product_training_data.sh" \
        "${source}" "${snapshot}" "${mode}"

    run_with_retry bash \
        "${ROOT_DIR}/scripts/feature/create_aisle_training_data.sh" \
        "${source}" "${snapshot}" "${mode}"

    run_with_retry bash \
        "${ROOT_DIR}/scripts/feature/create_reorder_size_training_data.sh" \
        "${source}" "${snapshot}" "${mode}"
}


build_base_period() {
    local period=$1
    local mode

    echo
    echo "============================================================"
    echo "Building base data: source=${SOURCE}, period=${period}"
    echo "============================================================"

    run_with_retry bash \
        "${ROOT_DIR}/scripts/simulation/create_order_role_split_data.sh" \
        "${SOURCE}" "${period}"

    run_with_retry bash \
        "${ROOT_DIR}/scripts/feature/create_order_products_data.sh" \
        "${SOURCE}" "${period}"

    run_with_retry bash \
        "${ROOT_DIR}/scripts/simulation/create_order_simulation_split_data.sh" \
        "${SOURCE}" "${period}" "base_train"

    for mode in train validation evaluation; do
        build_features \
            "${SOURCE}" \
            "${period}" \
            "${mode}"
    done
}


build_stacking_data() {
    local mode

    echo
    echo "============================================================"
    echo "Building stacking training data"
    echo "============================================================"

    run_with_retry bash \
        "${ROOT_DIR}/scripts/simulation/create_order_simulation_split_data.sh" \
        "${SOURCE}" "t2" "stacking_train"

    for mode in train validation; do
        build_features \
            "${SOURCE}" \
            "stacking_train" \
            "${mode}"
    done
}


if [[ "${SKIP_CLUSTER_RESTART}" == "1" ]]; then
    state="$(get_cluster_state)"

    if [[ "${state}" != "RUNNING" ]]; then
        echo \
            "SKIP_CLUSTER_RESTART=1, but cluster state is ${state}, not RUNNING." \
            >&2

        exit 1
    fi

    echo "Skipping cluster restart; cluster is already RUNNING."
else
    restart_cluster
fi


for period in initial t1 t2; do
    build_base_period "${period}"
done


build_stacking_data


echo
echo "All training data builds completed successfully."