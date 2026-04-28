# Databricks notebook source
# MAGIC %md
# MAGIC - **aurora.gold.fact_forecast_performance** is formed using below tables: [_**finalized logic is unclear**_] 
# MAGIC - aurora.default.baseline_fcst_2025_dec_2026_march
# MAGIC - aurora.gold.dim_sku
# MAGIC - aurora.silver.synthetic_sales_actuals_2023_2026
# MAGIC - aurora.default.rolling_forecast_march_april_df

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# COMMAND ----------

cluster_forecast_df = spark.table("aurora.default.baseline_fcst_2025_dec_2026_march").select(
    F.col("store"),
    F.col("department_code"),
    F.col("cluster_code"),
    F.col("weekstartdate").alias("next_sunday"),
    F.col("forecast_qty")
)

cluster_forecast_df = cluster_forecast_df.withColumn("next_sunday", F.to_date(F.col("next_sunday")))

display(cluster_forecast_df)

# COMMAND ----------

fact_df = spark.table("aurora.gold.fact_demand_history")

display(fact_df)

# COMMAND ----------

weekly_actuals = (
    fact_df
    .withColumn(
        "forecast_target_date",
        F.date_sub(F.col("next_sunday"), F.dayofweek("next_sunday") - F.lit(1))
    )
    .groupBy("sku_id", "location_id", "forecast_target_date")
    .agg(
        F.sum("units_sold").alias("actual_units")
    )
)

display(weekly_actuals)

# COMMAND ----------

dim_sku = spark.table("aurora.gold.dim_sku")

dim_sku = dim_sku.withColumn(
    "category",
    F.split(F.col("category"), ":")[0]
).withColumn(
    "subcategory",
    F.split(F.col("subcategory"), "_")[0]
)

display(dim_sku)

# COMMAND ----------

sales_enriched = (
    fact_df
    .join(
        dim_sku.select(
            "sku_id",
            F.col("category").alias("department_code"),
            F.col("subcategory").alias("cluster_code")),
        on="sku_id",
        how="left"
    )
)

display(sales_enriched)

# COMMAND ----------

weight_anchor = "2026-03-15"

recent_sales = sales_enriched.filter(
    (F.col("next_sunday") >= F.date_sub(F.lit(weight_anchor), 56)) &
    (F.col("next_sunday") <= F.lit(weight_anchor))
)

sku_cluster_sales = (
    recent_sales
    .groupBy("location_id", "department_code", "cluster_code", "sku_id")
    .agg(F.sum("units_sold").alias("sku_sales"))
)

cluster_sales = (
    sku_cluster_sales
    .groupBy("location_id", "department_code", "cluster_code")
    .agg(F.sum("sku_sales").alias("cluster_sales"))
)

weights = (
    sku_cluster_sales
    .join(
        cluster_sales,
        ["location_id", "department_code", "cluster_code"]
    )
    .withColumn(
        "weight",
        F.when(F.col("cluster_sales") != 0,
               F.col("sku_sales") / F.col("cluster_sales"))
         .otherwise(0)
    )
)

# COMMAND ----------

display(weights)

# COMMAND ----------

cluster_forecast_df = cluster_forecast_df.withColumnRenamed("store", "location_id")

sku_forecast = (
    cluster_forecast_df
    .join(
        weights,
        ["location_id", "department_code", "cluster_code"],
        "left"
    )
)

sku_forecast = (
    sku_forecast
    .withColumn("weight", F.coalesce("weight", F.lit(0)))
    .withColumn("sku_forecast_qty", F.col("forecast_qty") * F.col("weight"))
)

# COMMAND ----------

display(sku_forecast.groupBy(
    "location_id","department_code","cluster_code","next_sunday"
).agg(
    F.round(F.sum("sku_forecast_qty"), 2).alias("sum_sku_forecast")
))

# COMMAND ----------

display(cluster_forecast_df)

# display(cluster_forecast_df.groupBy(
#     "location_id","department_code","cluster_code","next_sunday"
# ).agg(
#     F.sum("forecast_qty").alias("sum_cluster_forecast")
# ))

# COMMAND ----------

sku_actuals = weekly_actuals.select(
    "sku_id",
    "location_id",
    F.col("forecast_target_date").alias("next_sunday"),
    "actual_units"
)

sku_perf = (
    sku_forecast
    .join(
        sku_actuals,
        on=["sku_id", "location_id", "next_sunday"],
        how="left"
    )
    .fillna(0)
)

# COMMAND ----------

weekly_actuals.filter(F.col("forecast_target_date") == "2025-07-13").show()

# COMMAND ----------

sku_forecast.select("next_sunday").distinct().orderBy("next_sunday").show()
weekly_actuals.select("forecast_target_date").distinct().orderBy(F.col("forecast_target_date").desc()).show()

# COMMAND ----------

display(sku_perf)

# COMMAND ----------

fact_forecast_performance_df = sku_perf.select(
    "sku_id",
    "location_id",
    F.current_date().alias("forecast_creation_date"),
    F.col("next_sunday").alias("forecast_target_date"),
    "actual_units",
    F.col("sku_forecast_qty").alias("forecast_units").cast(DecimalType(14, 2))
)

fact_forecast_performance = fact_forecast_performance_df.filter(F.col("forecast_target_date") < '2026-03-22') 

display(fact_forecast_performance)

# COMMAND ----------

fact_df = (
    spark
    .table("aurora.silver.synthetic_sales_actuals_2023_2026")
    .filter(F.col("next_sunday") > "2025-09-28")
    .select(
        F.col("sku_code").alias("sku_id"),
        F.col("store").alias("location_id"),
        F.current_date().alias("forecast_creation_date"),
        F.col("next_sunday").alias("forecast_target_date"),
        F.col("weekly_sales_qty").alias("actual_units").cast("int"),
        F.lit(None).alias("forecast_units").cast(DecimalType(14, 2))
    )
)

display(fact_df)
print(fact_df.count())

# COMMAND ----------

display(fact_df.groupBy("sku_id", "location_id", "forecast_target_date").count())

# COMMAND ----------

fact_df.select(
    F.min("forecast_target_date"),
    F.max("forecast_target_date")
).show()

# COMMAND ----------

display(fact_forecast_performance.filter(F.col("location_id") == "10424"))

# COMMAND ----------

fact_forecast_performance = (
    fact_forecast_performance
    .withColumn("sku_id", F.coalesce(F.col("sku_id"), F.lit("NA")))
    .withColumn("forecast_units", F.coalesce(F.col("forecast_units"), F.lit(0)))
)

# COMMAND ----------

fact_forecast_performance_df = spark.table("aurora.gold.fact_forecast_performance")

display(fact_forecast_performance_df)

# COMMAND ----------


    

display(fact_df)

# COMMAND ----------

(fact_df.write
 .mode("overwrite")
 .saveAsTable("aurora.gold.fact_forecast_performance")
)

# COMMAND ----------

fact_forecast_performance = fact_forecast_performance \
    .withColumn(
        "actual_units",
        (F.col("actual_units")).cast(IntegerType())
    ) \
    .withColumn(
        "forecast_error",
        (F.col("actual_units") - F.col("forecast_units")).cast(DecimalType(14, 4))
    ) \
    .withColumn(
        "absolute_error",
        (F.abs("forecast_error")).cast(DecimalType(14, 4))
    ) \
    .withColumn(
        "squared_error",
        (F.pow("forecast_error", 2)).cast(DecimalType(18, 6))
    ) \
    .withColumn(
        "pct_error",
        F.when(
            F.col("actual_units") != 0,
            (F.col("forecast_error") / F.col("actual_units")) * 100
        ).otherwise(F.lit(None)).cast(DecimalType(12, 6))
    ) \
    .withColumn(
        "bias_pct",
        F.when(
            F.col("actual_units") != 0,
            ((F.col("forecast_units") - F.col("actual_units")) / F.col("actual_units")) * 100            
        ).otherwise(F.lit(None)).cast(DecimalType(5, 2))
    ) \
    .withColumn(
        "confidence_pct", 
        F.when(
            F.col("actual_units") != 0,
            F.greatest(
                F.lit(0),
                1 - F.abs(F.col("forecast_error") / F.col("actual_units"))) * 100
        ).otherwise(F.lit(None)).cast(DecimalType(5, 2))
    )

display(fact_forecast_performance)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *

df = spark.table("aurora.default.rolling_forecast_march_april_df").drop("department_code", "cluster_code")

fcst_performance_df = df.withColumnsRenamed({
    "sku_code": "sku_id",
    "store": "location_id",
    "next_sunday": "forecast_target_date",
    "actual_qty": "actual_units",
    "forecast_qty": "forecast_units",
})

fcst_performance_df = (
    fcst_performance_df
    .withColumn("forecast_creation_date", F.current_date())
    .withColumn("forecast_units", F.col("forecast_units").cast(DecimalType(14, 2)))
)

display(fcst_performance_df)

# COMMAND ----------

fact_df = spark.table("aurora.gold.fact_forecast_performance")

display(fact_df)

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *

fact_df.select(
    F.min("forecast_target_date"),
    F.max("forecast_target_date")
).show()

# COMMAND ----------

(fcst_performance_df.write
 .mode("append")
 .saveAsTable("aurora.gold.fact_forecast_performance")
)