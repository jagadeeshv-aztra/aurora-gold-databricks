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
# MAGIC       <td><code>aurora.default.inventory_aware_input_df_2023_2026</code></td>
# MAGIC       <td>INPUT</td>
# MAGIC       <td>Weekly inventory/sales series used to compute rolling demand statistics</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><code>aurora.gold.fact_forecast_performance</code></td>
# MAGIC       <td>INPUT</td>
# MAGIC       <td>(Read-only) used for date range checks / validation</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><code>aurora.gold.fact_supply_coverage</code></td>
# MAGIC       <td>OUTPUT</td>
# MAGIC       <td>Coverage metrics by SKU x Location x Week</td>
# MAGIC     </tr>
# MAGIC   </tbody>
# MAGIC </table>

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
import math

# COMMAND ----------

INPUT_TABLE_INVENTORY = "aurora.default.inventory_aware_input_df_2023_2026"
INPUT_TABLE_FACT_FORECAST_PERFORMANCE = "aurora.gold.fact_forecast_performance"
OUTPUT_TABLE_FACT_SUPPLY_COVERAGE = "aurora.gold.fact_supply_coverage"

# Important variables / knobs
LEAD_TIME_DAYS = 20
REVIEW_PERIOD_DAYS = 7
SERVICE_LEVEL = 0.95
SERVICE_LEVEL_Z = 1.65  # z-score for ~95%

WRITE_MODE = "append"  # append as new computation dates are produced
WRITE_FORMAT = "delta"

# COMMAND ----------

def build_fact_supply_coverage(
    inventory_df,
    lead_time_days: int,
    review_period_days: int,
    service_level: float,
    z_score: float,
):
    lead_time_weeks = lead_time_days / 7
    review_weeks = review_period_days / 7
    window_weeks = int(lead_time_weeks + review_weeks)

    inventory_df = inventory_df.withColumnRenamed("date", "week_start_date")

    window_spec = (
        Window.partitionBy("sku_code", "store")
        .orderBy("week_start_date")
        .rowsBetween(-window_weeks, -1)
    )

    inventory_df = inventory_df.withColumn("mean_weekly_demand", F.avg("sales_quantity").over(window_spec))
    inventory_df = inventory_df.withColumn(
        "weekly_std_dev",
        F.when(
            F.count("sales_quantity").over(window_spec) > 1,
            F.stddev("sales_quantity").over(window_spec),
        ).otherwise(0),
    )

    inventory_df = inventory_df.withColumn(
        "window_demand",
        F.col("mean_weekly_demand") * (lead_time_weeks + review_weeks),
    ).withColumn(
        "window_std_dev",
        F.col("weekly_std_dev") * F.lit(math.sqrt(lead_time_weeks)),
    )

    inventory_df = inventory_df.withColumn("safety_stock", F.lit(z_score) * F.col("window_std_dev"))
    inventory_df = inventory_df.withColumn("required_coverage", F.col("window_demand") + F.col("safety_stock"))

    return inventory_df.select(
        F.col("sku_code").alias("sku_id").cast(StringType()),
        F.col("store").alias("location_id").cast(StringType()),
        F.col("week_start_date").alias("computation_date"),
        F.lit(lead_time_days).alias("lead_time_days"),
        F.lit(review_period_days).alias("review_period_days"),
        F.lit(service_level).cast("decimal(5,2)").alias("service_level"),
        F.col("window_demand").cast("decimal(18,4)").alias("window_demand"),
        F.col("window_std_dev").cast("decimal(18,6)").alias("window_std_dev"),
        F.col("safety_stock").cast("decimal(18,4)").alias("safety_stock"),
        F.col("required_coverage").cast("decimal(18,4)").alias("required_coverage"),
    )

# COMMAND ----------

def main():
    # Optional validation read (kept job-safe if table absent)
    try:
        _ = spark.table(INPUT_TABLE_FACT_FORECAST_PERFORMANCE).select(
            F.min("forecast_target_date").alias("min_forecast_target_date"),
            F.max("forecast_target_date").alias("max_forecast_target_date"),
        )
    except Exception:
        pass

    inventory_df = spark.table(INPUT_TABLE_INVENTORY)
    fact_supply_coverage = build_fact_supply_coverage(
        inventory_df=inventory_df,
        lead_time_days=LEAD_TIME_DAYS,
        review_period_days=REVIEW_PERIOD_DAYS,
        service_level=SERVICE_LEVEL,
        z_score=SERVICE_LEVEL_Z,
    )

    fact_supply_coverage.write.format(WRITE_FORMAT).mode(WRITE_MODE).saveAsTable(OUTPUT_TABLE_FACT_SUPPLY_COVERAGE)


main()