import torch
from torch.utils.data import DataLoader, Dataset


class InstacartDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _shift_left(x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                x[1:],
                torch.zeros(1, dtype=x.dtype, device=x.device),
            ]
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        is_ordered_history = torch.tensor(
            row["is_ordered_history"],
            dtype=torch.long,
        )

        batch = {
            "user_id": torch.tensor(
                row["user_id"],
                dtype=torch.long,
            ),
            "product_id": torch.tensor(
                row["product_id"],
                dtype=torch.long,
            ),
            "label": torch.tensor(
                row["label"],
                dtype=torch.float32,
            ),
            "product_name_encoded": torch.tensor(
                row["product_name_encoded"],
                dtype=torch.long,
            ),
            "position_in_order_history": torch.tensor(
                row["position_in_order_history"],
                dtype=torch.long,
            ),
            "history_order_size": torch.tensor(
                row["history_order_size"],
                dtype=torch.long,
            ),
            "history_reorder_size": torch.tensor(
                row["history_reorder_size"],
                dtype=torch.long,
            ),
            "order_dow_history": self._shift_left(
                torch.tensor(
                    row["order_dows"],
                    dtype=torch.long,
                )
            ),
            "order_hour_history": self._shift_left(
                torch.tensor(
                    row["order_hours"],
                    dtype=torch.long,
                )
            ),
            "days_since_prior_order_history": self._shift_left(
                torch.tensor(
                    row["days_since_prior_orders"],
                    dtype=torch.long,
                )
            ),
            "order_number_history": self._shift_left(
                torch.tensor(
                    row["order_numbers"],
                    dtype=torch.long,
                )
            ),
            "next_is_ordered": self._shift_left(is_ordered_history),
            "history_length": torch.tensor(
                row["history_length"] - 1,
                dtype=torch.long,
            ),
            "product_name_length": torch.tensor(
                row["product_name_length"],
                dtype=torch.long,
            ),
        }

        batch["is_none"] = batch["product_id"] == 0

        return batch


if __name__ == "__main__":
    from instacart_etl_rnn.common.io import read_parquet
    from instacart_etl_rnn.common.spark import create_spark_session

    spark = create_spark_session("random")

    df = (
        read_parquet(
            "data/gcs/product_training_data_train",
            spark,
        )
        .limit(1000)
        .toPandas()
    )
    ds = InstacartDataset(df)
    train_dataloader = DataLoader(
        ds,
        shuffle=True,
        batch_size=32,
    )
    for batch in train_dataloader:
        print(batch)
