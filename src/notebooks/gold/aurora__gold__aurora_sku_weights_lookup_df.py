# Databricks notebook source
# MAGIC %md
# MAGIC - **aurora.gold.aurora_sku_weights_lookup_df** is formed using below tables:
# MAGIC - aurora.silver.aurora_filtered_sales_actuals_2023_2026

# COMMAND ----------

import pyspark.sql.functions as F

actuals_df = spark.table("aurora.silver.aurora_filtered_sales_actuals_2023_2026")

print(actuals_df.count())
display(actuals_df)

# COMMAND ----------

cluster_actuals = (
    actuals_df                                       
    .groupBy("store", "department_code", "cluster_code", "next_sunday")
    .agg(F.sum("sales_quantity").alias("cluster_actual_qty"))
)

display(cluster_actuals)

# COMMAND ----------

sku_weights = (
    actuals_df
    .groupBy("store", "department_code", "cluster_code", "sku_code")
    .agg(F.sum("sales_quantity").alias("sku_total_sales"))
)

display(sku_weights)

# COMMAND ----------

cluster_totals = (
    sku_weights
    .groupBy("store", "department_code", "cluster_code")
    .agg(F.sum("sku_total_sales").alias("cluster_total_sales"))
)

display(cluster_totals)

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType

sku_weights = (
    sku_weights
    .join(cluster_totals, on=["store", "department_code", "cluster_code"], how="left")
    .withColumn(
        "sku_weight",
        F.when(
            F.col("cluster_total_sales") > 0,
            F.col("sku_total_sales") / F.col("cluster_total_sales")
        ).otherwise(F.lit(0.0))
    )
    # Normalize weights to ensure they sum to 1.0 per cluster
    # (guards against rounding errors)
    .withColumn(
        "weight_sum",
        F.sum("sku_weight").over(
            Window.partitionBy("store", "department_code", "cluster_code")
        )
    )
    .withColumn(
        "sku_weight",
        F.when(F.col("weight_sum") > 0,
            F.col("sku_weight") / F.col("weight_sum")
        ).otherwise(F.lit(0.0))
    )
    .select("store", "department_code", "cluster_code", "sku_code", "sku_weight")
)

display(sku_weights)

# COMMAND ----------

(sku_weights.write
 .mode("overwrite")
 .saveAsTable("aurora.gold.aurora_sku_weights_lookup_table")
)

# COMMAND ----------

sku_forecast = (
    base_fcst_df
    .join(sku_weights, on=["store", "department_code", "cluster_code"], how="left")
    .withColumn(
        "sku_forecast_qty",
        F.round(F.col("forecast_qty") * F.col("sku_weight"), 3)
    )
)

display(sku_forecast)

# COMMAND ----------

print("Forecast date range:")
sku_forecast.agg(
    F.min("next_sunday").alias("min_date"),
    F.max("next_sunday").alias("max_date"),
    F.countDistinct("next_sunday").alias("distinct_weeks")
).show()

print("Null sku_forecast_qty rows:", sku_forecast.filter(F.col("sku_forecast_qty").isNull()).count())

# COMMAND ----------

display(
    sku_forecast
    .groupBy("store", "department_code", "cluster_code", "sku_code", "next_sunday")
    .count()
    .filter(F.col("count") > 1)
)

# COMMAND ----------

final_df = sku_forecast.select(
    "store",
    "department_code",
    "cluster_code",
    "sku_code",
    "next_sunday",
    F.col("sku_forecast_qty").alias("forecast_qty")
)

final_df.write.mode("overwrite").saveAsTable("aurora.default.aurora_sku_forecast_march_2026_oct_2026")
print("✅ Done")

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *

df_existing = spark.table("aurora.gold.baseline_fcst_2026_store_cluster_week_ml")

display(df_existing)

# COMMAND ----------

df_existing.select(
    F.min("weekstartdate"),
    F.max("weekstartdate")
).show()