# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>Databricks tables used</h3>
# MAGIC <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; width:100%">
# MAGIC <thead>
# MAGIC <tr>
# MAGIC <th align="left">Table</th>
# MAGIC <th align="left">Role</th>
# MAGIC <th align="left">How it’s used</th>
# MAGIC </tr>
# MAGIC </thead>
# MAGIC <tbody>
# MAGIC <tr>
# MAGIC <td><code>aurora.silver.aurora_filtered_sales_actuals_2023_2026</code></td>
# MAGIC <td>INPUT</td>
# MAGIC <td>Derives min/max <code>sales_date</code> bounds</td>
# MAGIC </tr>
# MAGIC <tr>
# MAGIC <td><code>aurora.silver.retail_calendar_2023_2025</code></td>
# MAGIC <td>INPUT</td>
# MAGIC <td>Fiscal calendar attributes mapped to <code>date_key</code></td>
# MAGIC </tr>
# MAGIC <tr>
# MAGIC <td><code>aurora.gold.dim_time</code></td>
# MAGIC <td>OUTPUT</td>
# MAGIC <td>Date dimension populated for the observed sales date range</td>
# MAGIC </tr>
# MAGIC </tbody>
# MAGIC </table>

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *

# COMMAND ----------

INPUT_TABLE_SALES = "aurora.silver.aurora_filtered_sales_actuals_2023_2026"
INPUT_TABLE_RETAIL_CALENDAR = "aurora.silver.retail_calendar_2023_2025"
OUTPUT_TABLE_DIM_TIME = "aurora.gold.dim_time"

WRITE_MODE = "overwrite"  # full refresh dimension
WRITE_FORMAT = "delta"

# COMMAND ----------

def build_dim_time(sales_df, calendar_df):
    # Determine bounds from sales
    date_bounds = (
        sales_df.agg(
            F.min("sales_date").alias("min_date"),
            F.max("sales_date").alias("max_date"),
        ).collect()[0]
    )

    min_date = date_bounds["min_date"]
    max_date = date_bounds["max_date"]

    if min_date is None or max_date is None:
        raise ValueError("sales_date bounds are null; cannot build dim_time.")

    date_df = spark.sql(
        f"""
        SELECT explode(
            sequence(
                to_date('{min_date}'),
                to_date('{max_date}'),
                interval 1 day
            )
        ) AS date_key
        """
    )

    dim_time = (
        date_df.withColumn("day_of_week", F.dayofweek("date_key"))
        .withColumn("week_of_year", F.weekofyear("date_key"))
        .withColumn("month", F.month("date_key"))
        .withColumn("quarter", F.quarter("date_key"))
        .withColumn("year", F.year("date_key"))
        .withColumn(
            "is_weekend",
            F.when(F.dayofweek("date_key").isin(1, 7), True).otherwise(False),
        )
    )

    dim_time_final = (
        dim_time.join(
            calendar_df.select(
                "date_key",
                "fiscal_year",
                "fiscal_week",
                "fiscal_month",
            ),
            on="date_key",
            how="left",
        )
        .select(
            "date_key",
            "day_of_week",
            "week_of_year",
            "month",
            "quarter",
            "year",
            "fiscal_week",
            "fiscal_month",
            "fiscal_year",
            "is_weekend",
        )
    )

    return dim_time_final

# COMMAND ----------

def main():
    sales_df = spark.read.table(INPUT_TABLE_SALES)
    calendar_df = spark.table(INPUT_TABLE_RETAIL_CALENDAR)

    dim_time_final = build_dim_time(sales_df=sales_df, calendar_df=calendar_df)

    (
        dim_time_final.write.format(WRITE_FORMAT)
        .mode(WRITE_MODE)
        .saveAsTable(OUTPUT_TABLE_DIM_TIME)
    )

# COMMAND ----------

main()
