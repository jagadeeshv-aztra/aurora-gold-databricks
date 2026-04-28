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
# MAGIC       <td>Base weekly sales actuals (filtered by cutoff)</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><code>aurora.default.promodf_table</code></td>
# MAGIC       <td>INPUT</td>
# MAGIC       <td>Promotion attributes joined by <code>department_code</code></td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><code>aurora.gold.fact_demand_history</code></td>
# MAGIC       <td>OUTPUT</td>
# MAGIC       <td>Gold demand history at SKU x Location x Week</td>
# MAGIC     </tr>
# MAGIC   </tbody>
# MAGIC </table>

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# COMMAND ----------

INPUT_TABLE_SALES = "aurora.silver.aurora_filtered_sales_actuals_2023_2026"
INPUT_TABLE_PROMO = "aurora.default.promodf_table"
OUTPUT_TABLE_FACT_DEMAND_HISTORY = "aurora.gold.fact_demand_history"

# Important variables / knobs
SALES_CUTOFF_DATE_EXCLUSIVE = "2026-03-29"  # keep only next_sunday < cutoff

WRITE_MODE = "overwrite"  # full refresh
WRITE_FORMAT = "delta"
OVERWRITE_SCHEMA = True

# COMMAND ----------

def build_fact_demand_history(sales_df, promo_df):
    promo_df = promo_df.select("department_code", "promotion_type", "promotion_value").dropDuplicates(
        ["department_code"]
    )

    sales_promo_df = sales_df.join(promo_df, on=["department_code"], how="left")

    # Normalize promotion value and derive discount/promo flags
    sales_promo_df = sales_promo_df.withColumn(
        "promotion_value",
        F.regexp_replace(F.col("promotion_value"), "[%$]", "").cast("double"),
    )

    sales_promo_df = sales_promo_df.withColumn(
        "promotion_value",
        F.coalesce(F.col("promotion_value"), F.round(F.col("selling_price") * 0.5, 2)),
    )

    sales_promo_df = sales_promo_df.withColumn(
        "discount_value",
        F.when(
            F.col("promotion_type") == "PERCENT",
            F.round(F.col("selling_price") * (F.col("promotion_value") / 100), 2),
        )
        .when(F.col("promotion_type") == "FIXED", F.col("promotion_value"))
        .otherwise(F.col("promotion_value")),
    )

    sales_promo_df = sales_promo_df.withColumn(
        "discount_value",
        F.when(F.col("discount_value") < F.lit(0), F.lit(0)).otherwise(F.col("discount_value")),
    )

    sales_promo_df = sales_promo_df.withColumn(
        "discount_pct",
        F.round((F.col("discount_value") / F.col("selling_price")), 2),
    ).withColumn("discount_pct", F.coalesce(F.col("discount_pct"), F.lit(0)))

    pricing_df = sales_promo_df.withColumn(
        "base_promo_flag",
        F.when(
            (F.col("promotion_type") == "BOGO") | (F.col("discount_pct") >= 0.10),
            1,
        ).otherwise(0),
    )

    win = Window.partitionBy("store", "sku_code").orderBy("next_sunday")
    pricing_df = pricing_df.withColumn("promo_flag_lag", F.lag("base_promo_flag").over(win))
    pricing_df = pricing_df.withColumn("promo_flag_lag", F.coalesce(F.col("promo_flag_lag"), F.lit(0)))

    pricing_df = pricing_df.withColumn(
        "promo_flag",
        F.when((F.col("base_promo_flag") == 1) & (F.col("promo_flag_lag") == 1), 1).otherwise(0),
    )

    fact_demand_history = (
        pricing_df.select(
            F.col("sku_code").alias("sku_id"),
            F.col("store").alias("location_id"),
            F.col("next_sunday").alias("demand_date"),
            F.col("sales_quantity").cast(IntegerType()).alias("units_sold"),
            F.col("promo_flag").cast(BooleanType()).alias("promo_flag"),
            F.col("selling_price").cast(DecimalType(12, 2)).alias("selling_price"),
        )
        .groupBy("sku_id", "location_id", "demand_date", "promo_flag")
        .agg(
            (
                F.sum(F.col("units_sold") * F.col("selling_price")) / F.sum("units_sold")
            ).cast(DecimalType(12, 2)).alias("selling_price"),
            F.sum("units_sold").cast("int").alias("units_sold"),
        )
        .withColumn(
            "net_revenue",
            (F.col("units_sold") * F.col("selling_price")).cast(DecimalType(14, 2)),
        )
        .select(
            "sku_id",
            "location_id",
            "demand_date",
            "units_sold",
            "selling_price",
            "net_revenue",
            "promo_flag",
        )
    )

    return fact_demand_history

# COMMAND ----------

def main():
    sales_df = (
        spark.table(INPUT_TABLE_SALES)
        .filter(F.col("next_sunday") < F.lit(SALES_CUTOFF_DATE_EXCLUSIVE))
    )
    promo_df = spark.table(INPUT_TABLE_PROMO)

    final_fact_demand_history = build_fact_demand_history(sales_df=sales_df, promo_df=promo_df)

    writer = final_fact_demand_history.write.format(WRITE_FORMAT).mode(WRITE_MODE)
    if OVERWRITE_SCHEMA:
        writer = writer.option("overwriteSchema", "true")
    writer.saveAsTable(OUTPUT_TABLE_FACT_DEMAND_HISTORY)


if __name__ == "__main__":
    main()