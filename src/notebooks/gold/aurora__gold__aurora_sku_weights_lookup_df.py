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
# MAGIC       <td><code>aurora.silver.aurora_filtered_sales_actuals_2023_2026</code></td>
# MAGIC       <td>INPUT</td>
# MAGIC       <td>Computes SKU sales share within each store x department x cluster</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><code>aurora.gold.aurora_sku_weights_lookup_table</code></td>
# MAGIC       <td>OUTPUT</td>
# MAGIC       <td>Normalized SKU weights lookup</td>
# MAGIC     </tr>
# MAGIC   </tbody>
# MAGIC </table>

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window

INPUT_TABLE_SALES_ACTUALS = "aurora.silver.aurora_filtered_sales_actuals_2023_2026"
OUTPUT_TABLE_SKU_WEIGHTS = "aurora.gold.aurora_sku_weights_lookup_table"

# Important variables / knobs
FILTER_START_DATE_INCLUSIVE = None  # e.g. "2025-01-01" or None
FILTER_END_DATE_INCLUSIVE = None    # e.g. "2026-03-15" or None

WRITE_MODE = "overwrite"
WRITE_FORMAT = "delta"

# COMMAND ----------

def build_sku_weights(actuals_df):
    df = actuals_df
    if FILTER_START_DATE_INCLUSIVE is not None:
        df = df.filter(F.col("next_sunday") >= F.lit(FILTER_START_DATE_INCLUSIVE))
    if FILTER_END_DATE_INCLUSIVE is not None:
        df = df.filter(F.col("next_sunday") <= F.lit(FILTER_END_DATE_INCLUSIVE))

    sku_sales = df.groupBy("store", "department_code", "cluster_code", "sku_code").agg(
        F.sum("sales_quantity").alias("sku_total_sales")
    )

    cluster_totals = sku_sales.groupBy("store", "department_code", "cluster_code").agg(
        F.sum("sku_total_sales").alias("cluster_total_sales")
    )

    weights = (
        sku_sales.join(cluster_totals, on=["store", "department_code", "cluster_code"], how="left")
        .withColumn(
            "sku_weight",
            F.when(F.col("cluster_total_sales") > 0, F.col("sku_total_sales") / F.col("cluster_total_sales")).otherwise(
                F.lit(0.0)
            ),
        )
        .withColumn("weight_sum", F.sum("sku_weight").over(Window.partitionBy("store", "department_code", "cluster_code")))
        .withColumn(
            "sku_weight",
            F.when(F.col("weight_sum") > 0, F.col("sku_weight") / F.col("weight_sum")).otherwise(F.lit(0.0)),
        )
        .select(
            F.col("store").cast("string").alias("store"),
            F.col("department_code").cast("string").alias("department_code"),
            F.col("cluster_code").cast("string").alias("cluster_code"),
            F.col("sku_code").cast("string").alias("sku_code"),
            F.col("sku_weight").cast("double").alias("sku_weight"),
        )
    )

    return weights


def main():
    actuals_df = spark.table(INPUT_TABLE_SALES_ACTUALS)
    sku_weights = build_sku_weights(actuals_df=actuals_df)
    sku_weights.write.format(WRITE_FORMAT).mode(WRITE_MODE).saveAsTable(OUTPUT_TABLE_SKU_WEIGHTS)


if __name__ == "__main__":
    main()