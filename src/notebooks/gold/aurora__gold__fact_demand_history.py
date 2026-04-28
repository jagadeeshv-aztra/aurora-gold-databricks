# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC - **aurora.gold.fact_demand_history** is formed using below tables: 
# MAGIC
# MAGIC - aurora.silver.aurora_filtered_sales_actuals_2023_2026
# MAGIC - aurora.default.promodf_table

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# COMMAND ----------

sales_df = (
    spark
    .table("aurora.silver.aurora_filtered_sales_actuals_2023_2026")
    .filter(F.col("next_sunday") < '2026-03-29')
    .orderBy(F.col("next_sunday").desc())
)

display(sales_df)
print(sales_df.count())

# COMMAND ----------

# sales_df = sales_df.withColumnsRenamed({
#     "weekly_sales_qty": "sales_quantity",
#     "weekly_selling_price": "selling_price",
#     "weekly_revenue": "revenue",
# })

# COMMAND ----------

promoDf = spark.sql("select * from aurora.default.promodf_table")
print(promoDf.columns)
display(promoDf)

# COMMAND ----------

promo_df = promoDf.select("department_code", "promotion_type", "promotion_value")
promo_df = promo_df.dropDuplicates(['department_code'])
print(promo_df.count())
display(promo_df)

# COMMAND ----------

sales_promo_df = sales_df.join(
    promo_df,
    on=['department_code'], how='left'
)
print(sales_promo_df.count())
display(sales_promo_df)

# COMMAND ----------

sales_promo_df = sales_promo_df.withColumn("promotion_value", F.regexp_replace(F.col("promotion_value"), "[%$]", "").cast("double"))

sales_promo_df = (sales_promo_df.withColumn("promotion_value", F.coalesce(F.col("promotion_value"), F.round(F.col("selling_price") * 0.5, 2))))

sales_promo_df = (sales_promo_df.withColumn(
    "discount_value",
    F.when(F.col("promotion_type") == "PERCENT",
           F.round(F.col("selling_price") * (F.col("promotion_value") / 100), 2))
     .when(F.col("promotion_type") == "FIXED", F.col("promotion_value"))
     .otherwise(F.col("promotion_value")))
)
               
sales_promo_df = sales_promo_df.withColumn("discount_value", F.when(F.col("discount_value") < F.lit(0), F.lit(0)).otherwise(F.col("discount_value")))

# Compute 'discount_pct'
sales_promo_df = sales_promo_df.withColumn("discount_pct", F.round((F.col("discount_value") / F.col("selling_price")), 2))
sales_promo_df = sales_promo_df.withColumn("discount_pct", F.coalesce(F.col("discount_pct"), F.lit(0)))

# Derive base_promo_flag
pricing_df = sales_promo_df.withColumn(
    "base_promo_flag",
    F.when(
        (F.col("promotion_type") == "BOGO") |
        (F.col("discount_pct") >= 0.10),
        1
    ).otherwise(0)
)

# Add lag to promo_flag
win = Window.partitionBy("store", "sku_code").orderBy("next_sunday")

pricing_df = pricing_df.withColumn("promo_flag_lag", F.lag("base_promo_flag").over(win))
pricing_df = pricing_df.withColumn("promo_flag_lag", F.coalesce(F.col("promo_flag_lag"), F.lit(0)))

# Derive promo_flag
pricing_df = pricing_df.withColumn(
    "promo_flag",
    F.when(
        (F.col("base_promo_flag") == 1) &
        (F.col("promo_flag_lag") == 1),
        1
    ).otherwise(0)
)

display(pricing_df)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

fact_demand_history = (
    pricing_df
    .select(
        F.col("sku_code").alias("sku_id"),
        F.col("store").alias("location_id"),
        F.col("next_sunday").alias("demand_date"),
        F.col("sales_quantity").cast(IntegerType()).alias("units_sold"),
        F.col("promo_flag").cast(BooleanType()).alias("promo_flag"),
        F.col("selling_price").cast(DecimalType(12, 2)).alias("selling_price")
    )
    .groupBy(
        "sku_id",
        "location_id",
        "demand_date",
        "promo_flag"
    )
    .agg(
        (F.sum(F.col("units_sold") * F.col("selling_price")) /
         F.sum("units_sold")).cast(DecimalType(12, 2)).alias("selling_price"),
        F.sum("units_sold").cast("int").alias("units_sold")
    )
    .withColumn(
        "net_revenue",
        (F.col("units_sold") * F.col("selling_price")).cast(DecimalType(14, 2))
    )
)

display(fact_demand_history)
print(fact_demand_history.count())

# COMMAND ----------

dup_df = (
    fact_demand_history
    .groupBy("sku_id", "location_id", "demand_date", "promo_flag")
    .count()
    .filter(F.col("count") > 1)
)

display(dup_df)
print(dup_df.count())

# COMMAND ----------

final_fact_demand_history = fact_demand_history.select(
    "sku_id",
    "location_id",
    "demand_date",
    "units_sold",
    "selling_price",
    "net_revenue",
    "promo_flag"
)

display(final_fact_demand_history)

# COMMAND ----------

display(fact_demand_history.filter(F.col("next_sunday") < '2024-01-01'))
print(fact_demand_history.count())

# COMMAND ----------

(final_fact_demand_history.write
 .format("delta")
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable("aurora.gold.fact_demand_history")
)