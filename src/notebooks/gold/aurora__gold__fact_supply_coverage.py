# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC - **aurora.gold.fact_supply_coverage** is formed using below tables: 
# MAGIC
# MAGIC - aurora.default.inventory_aware_input_df_2023_2026
# MAGIC - aurora.gold.fact_forecast_performance

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
import math

# COMMAND ----------

inventory_df = spark.table("aurora.default.inventory_aware_weekly_df")

display(inventory_df)

# COMMAND ----------

inventory_df = spark.table("aurora.default.inventory_aware_input_df_2023_2026")

display(inventory_df)

# COMMAND ----------

df = spark.table("aurora.gold.fact_forecast_performance")

df.select(
    F.min("forecast_target_date"),
    F.max("forecast_target_date")
).distinct().show()

# display(df.select("forecast_target_date").distinct())

# COMMAND ----------

display(df.filter(F.col("forecast_target_date") >= '2025-12-07').orderBy("forecast_target_date"))

# COMMAND ----------

inventory_df.select(
    F.min("date"),
    F.max("date")
).show()

# COMMAND ----------

# -----------------------------
# 1️⃣ Parameters
# -----------------------------
lead_time_days = 20
review_period_days = 7
service_level = 0.95

# Z-score for service level (95%) -- service level factor 
Z = 1.65

lead_time_weeks = lead_time_days / 7
review_weeks = review_period_days / 7
window_weeks = int(lead_time_weeks + review_weeks)

# -----------------------------
# 2️⃣ Prepare Base Dataset
# -----------------------------

inventory_df = inventory_df.withColumnRenamed("date", "week_start_date")

# -----------------------------
# 3️⃣ Rolling Window Calculation
# -----------------------------

window_spec = (
    Window.partitionBy("sku_code", "store")
    .orderBy("week_start_date")
    .rowsBetween(-window_weeks, -1)  # previous N weeks
)

inventory_df = inventory_df \
    .withColumn(
        "mean_weekly_demand",
        F.avg("sales_quantity").over(window_spec)
    ) \
    .withColumn(
        "weekly_std_dev",
        F.stddev("sales_quantity").over(window_spec)
    )

inventory_df = inventory_df.withColumn(
    "weekly_std_dev",
    F.when(
        F.count("sales_quantity").over(window_spec) > 1,
        F.stddev("sales_quantity").over(window_spec)
    ).otherwise(0)
)

# -----------------------------
# 4️⃣ Compute Lead Time Demand
# -----------------------------

inventory_df = inventory_df \
    .withColumn(
        "window_demand",
        F.col("mean_weekly_demand") * (lead_time_weeks + review_weeks)
    ) \
    .withColumn(
        "window_std_dev",
        F.col("weekly_std_dev") * F.lit(math.sqrt(lead_time_weeks))
    )

# -----------------------------
# 5️⃣ Safety Stock
# -----------------------------

inventory_df = inventory_df.withColumn(
    "safety_stock",
    F.lit(Z) * F.col("window_std_dev")
)

# -----------------------------
# 6️⃣ Required Coverage
# -----------------------------

inventory_df = inventory_df.withColumn(
    "required_coverage",
    F.col("window_demand") + F.col("safety_stock")
)

# -----------------------------
# 7️⃣ Build fact_supply_coverage
# -----------------------------

fact_supply_coverage = inventory_df.select(
    F.col("sku_code").alias("sku_id").cast(StringType()),
    F.col("store").alias("location_id").cast(StringType()),
    F.col("week_start_date").alias("computation_date"),
    F.lit(lead_time_days).alias("lead_time_days"),
    F.lit(review_period_days).alias("review_period_days"),
    F.lit(service_level).cast("decimal(5,2)").alias("service_level"),
    F.col("window_demand").cast("decimal(18,4)"),
    F.col("window_std_dev").cast("decimal(18,6)"),
    F.col("safety_stock").cast("decimal(18,4)"),
    F.col("required_coverage").cast("decimal(18,4)")
)

# COMMAND ----------

display(fact_supply_coverage)

# COMMAND ----------

fact_supply_coverage.select(
    F.min("computation_date"),
    F.max("computation_date")
).show()

# COMMAND ----------

(fact_supply_coverage.write
.mode("append")
.saveAsTable("aurora.gold.fact_supply_coverage")
)