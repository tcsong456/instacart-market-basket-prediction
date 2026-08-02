from instacart_etl_rnn.validation.models import ValidationReport


def find_result(report: ValidationReport, *, rule_name: str):
    matches = [result for result in report.results if rule_name in result.rule_name]

    assert matches, (
        f"No rule name of validation results matches the given rule name: {rule_name!r}"
    )

    assert len(matches) == 1, f"Expected one matching rule, found {len(matches)}"

    return matches[0]
