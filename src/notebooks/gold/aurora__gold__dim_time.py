# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC - **aurora.gold.dim_time** is formed using below tables: 
# MAGIC
# MAGIC - aurora.silver.aurora_filtered_sales_actuals_2023_2026
# MAGIC - aurora.silver.retail_calendar_2023_2025
# MAGIC

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# COMMAND ----------

#sales_df = spark.table("aurora.default.salesdf2_table")
#need to replace sales actuals 2026 
sales_df = spark.read.table("aurora.silver.aurora_filtered_sales_actuals_2023_2026")
display(sales_df)

# COMMAND ----------

# Generate continuous dates
date_bounds = (
    sales_df
    .agg(
        F.min("sales_date").alias("min_date"),
        F.max("sales_date").alias("max_date")
    )
    .collect()[0]
)

min_date = date_bounds["min_date"]
max_date = date_bounds["max_date"]

display(min_date)
display(max_date)

# COMMAND ----------

calendar_df = spark.table("aurora.silver.retail_calendar_2023_2025")

display(calendar_df)

# COMMAND ----------

# Generate continuous dates
date_bounds = (
    sales_df
    .agg(
        F.min("sales_date").alias("min_date"),
        F.max("sales_date").alias("max_date")
    )
    .collect()[0]
)

min_date = date_bounds["min_date"]
max_date = date_bounds["max_date"]

date_df = spark.sql(f"""
    SELECT explode(
        sequence(
            to_date('{min_date}'),
            to_date('{max_date}'),
            interval 1 day
        )
    ) AS date_key
""")

# COMMAND ----------

# Standard calendar attributes
dim_time = (
    date_df
    .withColumn("day_of_week", F.dayofweek("date_key"))
    .withColumn("week_of_year", F.weekofyear("date_key"))
    .withColumn("month", F.month("date_key"))
    .withColumn("quarter", F.quarter("date_key"))
    .withColumn("year", F.year("date_key"))
    .withColumn(
        "is_weekend",
        F.when(F.dayofweek("date_key").isin(1,7), True).otherwise(False)
    )
)

# COMMAND ----------

# Join with Retail calendar Mapping
dim_time_final = (
    dim_time
    .join(
        calendar_df.select(
            "date_key",
            "fiscal_year",
            "fiscal_week",
            "fiscal_month",
        ),
        on="date_key",
        how="left"
    )
)

# COMMAND ----------

dim_time_final = dim_time_final.select(
    "date_key",
    "day_of_week",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "fiscal_week",
    "fiscal_month",
    "fiscal_year",
    "is_weekend"
)

display(dim_time_final)

# COMMAND ----------

(dim_time_final.write
 .format("delta")
 .mode("overwrite")
 .saveAsTable("aurora.gold.dim_time")
)