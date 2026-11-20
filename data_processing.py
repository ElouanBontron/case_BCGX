import pandas as pd
import numpy as np
import polars as pl
import seaborn as sns

folder = "data"
file = "dataset_sales_2026.csv"
separator = ";"

def import_data():

    lf = pl.scan_csv(
        f'{folder}/{file}',
        separator=separator,
        has_header=True,
        infer_schema_length=10_000,  # Types estimés sur le début du fichier
    )
    
    lf = (
        lf
        .with_columns(
            pl.col("date_order").str.to_datetime(
                format="%Y-%m-%d"
            ),
            pl.col("date_invoice").str.to_datetime(
                format="%Y-%m-%d"
            ),
        )
        .with_columns(
            (
                pl.col("date_invoice") - pl.col("date_order")
            )
            .dt.total_days()
            .alias("delta_days")
        )
    )
    
    return lf