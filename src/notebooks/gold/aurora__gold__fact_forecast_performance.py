# Databricks notebook source
# MAGIC %md
# MAGIC <h3>Databricks tables used</h3>
# MAGIC <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; width:100%">
# MAGIC   <thead>
# MAGIC     <tr>
# MAGIC       <th align="left">Table</th>
# MAGIC       <th align="left">Role</th>
# MAGIC       <th align="left">How it’s used</th>
# MAGIC     </tr>
# MAGIC   </thead>
# MAGIC   <tbody>
# MAGIC     <tr>
# MAGIC       <td><code>aurora.default.rolling_forecast_march_april_df</code></td>
# MAGIC       <td>INPUT</td>
# MAGIC       <td>Primary source mode: provides sku/store weekly actuals + forecasts</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><code>aurora.default.baseline_fcst_2025_dec_2026_march</code></td>
# MAGIC       <td>INPUT</td>
# MAGIC       <td>Optional source mode: cluster-level forecasts</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><code>aurora.gold.fact_demand_history</code></td>
# MAGIC       <td>INPUT</td>
# MAGIC       <td>Optional source mode: provides SKU actuals</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><code>aurora.gold.dim_sku</code></td>
# MAGIC       <td>INPUT</td>
# MAGIC       <td>Optional source mode: maps SKU to department/cluster for weighting</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><code>aurora.silver.synthetic_sales_actuals_2023_2026</code></td>
# MAGIC       <td>INPUT</td>
# MAGIC       <td>Optional: seed rows when forecasts are unavailable</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><code>aurora.gold.fact_forecast_performance</code></td>
# MAGIC       <td>OUTPUT</td>
# MAGIC       <td>Gold forecast performance fact</td>
# MAGIC     </tr>
# MAGIC   </tbody>
# MAGIC </table>

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# COMMAND ----------

INPUT_TABLE_ROLLING_FORECAST = "aurora.default.rolling_forecast_march_april_df"
INPUT_TABLE_CLUSTER_FORECAST = "aurora.default.baseline_fcst_2025_dec_2026_march"
INPUT_TABLE_FACT_DEMAND_HISTORY = "aurora.gold.fact_demand_history"
INPUT_TABLE_DIM_SKU = "aurora.gold.dim_sku"
INPUT_TABLE_SYNTHETIC_ACTUALS = "aurora.silver.synthetic_sales_actuals_2023_2026"
OUTPUT_TABLE_FACT_FORECAST_PERFORMANCE = "aurora.gold.fact_forecast_performance"

# Choose a single, explicit source mode for Jobs.
# - "rolling_forecast": uses INPUT_TABLE_ROLLING_FORECAST (recommended for this repo as it already has actual_qty/forecast_qty)
# - "cluster_forecast": derives sku forecasts from cluster forecasts + recent-sales weights
# - "synthetic_seed": seeds actuals with null forecasts
SOURCE_MODE = "rolling_forecast"

# Important variables / knobs
WEIGHT_ANCHOR_DATE = "2026-03-15"
WEIGHT_LOOKBACK_DAYS = 56
MAX_FORECAST_TARGET_DATE_EXCLUSIVE = None  # e.g. "2026-03-22" or None

WRITE_MODE = "append"  # append new runs by default
WRITE_FORMAT = "delta"

# COMMAND ----------

def _apply_target_date_filter(df):
    if MAX_FORECAST_TARGET_DATE_EXCLUSIVE is None:
        return df
    return df.filter(F.col("forecast_target_date") < F.lit(MAX_FORECAST_TARGET_DATE_EXCLUSIVE))


def build_from_rolling_forecast(df):
    df = df.drop("department_code", "cluster_code")
    df = df.withColumnsRenamed(
        {
            "sku_code": "sku_id",
            "store": "location_id",
            "next_sunday": "forecast_target_date",
            "actual_qty": "actual_units",
            "forecast_qty": "forecast_units",
        }
    )

    df = (
        df.withColumn("forecast_creation_date", F.current_date())
        .withColumn("sku_id", F.col("sku_id").cast("string"))
        .withColumn("location_id", F.col("location_id").cast("string"))
        .withColumn("forecast_target_date", F.to_date("forecast_target_date"))
        .withColumn("actual_units", F.col("actual_units").cast("int"))
        .withColumn("forecast_units", F.col("forecast_units").cast(DecimalType(14, 2)))
    )

    df = _apply_target_date_filter(df)
    return df.select(
        "sku_id",
        "location_id",
        "forecast_creation_date",
        "forecast_target_date",
        "actual_units",
        "forecast_units",
    )


def build_from_synthetic_actuals(df):
    df = df.select(
        F.col("sku_code").alias("sku_id"),
        F.col("store").alias("location_id"),
        F.current_date().alias("forecast_creation_date"),
        F.col("next_sunday").alias("forecast_target_date"),
        F.col("weekly_sales_qty").alias("actual_units").cast("int"),
        F.lit(None).alias("forecast_units").cast(DecimalType(14, 2)),
    )
    df = _apply_target_date_filter(df)
    return df


def build_from_cluster_forecast(cluster_forecast_df, fact_demand_history_df, dim_sku_df):
    cluster_forecast_df = cluster_forecast_df.select(
        F.col("store").alias("location_id").cast("string"),
        F.col("department_code").cast("string"),
        F.col("cluster_code").cast("string"),
        F.to_date(F.col("weekstartdate")).alias("forecast_target_date"),
        F.col("forecast_qty").alias("cluster_forecast_units"),
    )

    weekly_actuals = (
        fact_demand_history_df.groupBy("sku_id", "location_id", F.col("demand_date").alias("forecast_target_date"))
        .agg(F.sum("units_sold").alias("actual_units"))
        .withColumn("sku_id", F.col("sku_id").cast("string"))
        .withColumn("location_id", F.col("location_id").cast("string"))
        .withColumn("forecast_target_date", F.to_date("forecast_target_date"))
    )

    dim_sku_df = dim_sku_df.withColumn("department_code", F.split(F.col("category"), ":")[0]).withColumn(
        "cluster_code", F.split(F.col("subcategory"), "_")[0]
    )

    sales_enriched = fact_demand_history_df.join(
        dim_sku_df.select("sku_id", "department_code", "cluster_code"),
        on="sku_id",
        how="left",
    ).select("sku_id", "location_id", "demand_date", "units_sold", "department_code", "cluster_code")

    recent_sales = sales_enriched.filter(
        (F.col("demand_date") >= F.date_sub(F.lit(WEIGHT_ANCHOR_DATE), WEIGHT_LOOKBACK_DAYS))
        & (F.col("demand_date") <= F.lit(WEIGHT_ANCHOR_DATE))
    )

    sku_cluster_sales = recent_sales.groupBy("location_id", "department_code", "cluster_code", "sku_id").agg(
        F.sum("units_sold").alias("sku_sales")
    )
    cluster_sales = sku_cluster_sales.groupBy("location_id", "department_code", "cluster_code").agg(
        F.sum("sku_sales").alias("cluster_sales")
    )
    weights = (
        sku_cluster_sales.join(cluster_sales, ["location_id", "department_code", "cluster_code"])
        .withColumn("weight", F.when(F.col("cluster_sales") != 0, F.col("sku_sales") / F.col("cluster_sales")).otherwise(0))
        .select("location_id", "department_code", "cluster_code", "sku_id", "weight")
    )

    sku_forecast = (
        cluster_forecast_df.join(weights, ["location_id", "department_code", "cluster_code"], "left")
        .withColumn("weight", F.coalesce("weight", F.lit(0.0)))
        .withColumn("forecast_units", (F.col("cluster_forecast_units") * F.col("weight")).cast(DecimalType(14, 2)))
        .select("sku_id", "location_id", "forecast_target_date", "forecast_units")
    )

    perf = (
        sku_forecast.join(weekly_actuals, on=["sku_id", "location_id", "forecast_target_date"], how="left")
        .withColumn("actual_units", F.coalesce(F.col("actual_units"), F.lit(0)).cast("int"))
        .withColumn("forecast_creation_date", F.current_date())
    )

    perf = _apply_target_date_filter(perf)
    return perf.select(
        "sku_id",
        "location_id",
        "forecast_creation_date",
        "forecast_target_date",
        "actual_units",
        "forecast_units",
    )


def main():
    if SOURCE_MODE == "rolling_forecast":
        df = spark.table(INPUT_TABLE_ROLLING_FORECAST)
        out_df = build_from_rolling_forecast(df)
    elif SOURCE_MODE == "cluster_forecast":
        cluster_df = spark.table(INPUT_TABLE_CLUSTER_FORECAST)
        fact_df = spark.table(INPUT_TABLE_FACT_DEMAND_HISTORY)
        dim_sku_df = spark.table(INPUT_TABLE_DIM_SKU)
        out_df = build_from_cluster_forecast(cluster_df, fact_df, dim_sku_df)
    elif SOURCE_MODE == "synthetic_seed":
        df = spark.table(INPUT_TABLE_SYNTHETIC_ACTUALS)
        out_df = build_from_synthetic_actuals(df)
    else:
        raise ValueError(f"Unsupported SOURCE_MODE: {SOURCE_MODE}")

    out_df.write.format(WRITE_FORMAT).mode(WRITE_MODE).saveAsTable(OUTPUT_TABLE_FACT_FORECAST_PERFORMANCE)


if __name__ == "__main__":
    main()