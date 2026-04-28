# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC - **aurora.gold.dim_location** is formed using below tables: 
# MAGIC
# MAGIC - aurora.silver.store_dim
# MAGIC - aurora.silver.retail_calendar_2023_2025
# MAGIC

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
import random

# COMMAND ----------

# location_df = spark.sql("SELECT * FROM aurora.silver.store_dim WHERE storecode IN (11059, 11082, 10425, 10424, 11384, 10051, 11451)").filter(F.col("countrycode") == "US")
location_df = spark.sql("SELECT * FROM aurora.silver.store_dim WHERE storecode IN (10425, 10424, 11384)").filter(F.col("countrycode") == "US")
display(location_df)

# COMMAND ----------

df = spark.sql("SELECT * FROM aurora.silver.store_dim")

display(df)

# COMMAND ----------

dim_location = location_df.select(
    "storecode",
    "storename",
    "regioncode",
    "city",
    "statedescription"
)

display(dim_location)

# COMMAND ----------

new_location = pd.DataFrame({
    "storecode": [11059, 11082, 11451],
    "storename": [" Twelve Oaks, MI", " International Plaza, FL", " Parkdale, TX"],
    "regioncode": ["1009 - Midwest", "1017 - Florida", "1099 - Undecided - US-US"],
    "city": ["Novi", "Tampa", "Beaumont"],
    "statedescription": ["Michigan", "Florida", "Texas"]
})

new_location_df = spark.createDataFrame(new_location)

location_df_fixed = dim_location.union(new_location_df)
display(location_df_fixed)

# COMMAND ----------

#sales_df = spark.table("aurora.default.salesdf2_table").select("store").distinct()
sales_df = spark.table("aurora.silver.aurora_filtered_sales_actuals_2023_2026").select("store").distinct()
display(sales_df)

# COMMAND ----------

df = location_df.toPandas()
display(df.groupby('storecode')['storename'].nunique())

# COMMAND ----------

dim_location = location_df_fixed.withColumnsRenamed({
    "storecode": "location_id",
    "storename": "location_name",
    "regioncode": "region",
    "statedescription": "state"
})

# channels = ["BANDM", "Direct", "Default"]

dim_location = (
    dim_location
    .withColumn("location_type", F.lit("BANDM"))
    .withColumn("created_at", F.current_timestamp())
)

dim_location_final = dim_location.select(
    "location_id",
    "location_name",
    "location_type",
    "region",
    "city",
    "state",
    "created_at"
)

display(dim_location_final)

# COMMAND ----------

(dim_location_final.write
 .format("delta")
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable("aurora.gold.dim_location")
)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

dim_forecast_model = spark.createDataFrame(
    [
        (
            "AUTO_ARIMA_BASE_V01",
            "BASELINE",
            "2023-07-29",
            "2025-07-28"
        ),
        (
            "XGBOOST_PROMO_V01",
            "PROMO AWARE",
            "2023-07-29",
            "2025-07-28"
        ),
        (
            "XGBOOST_EVENT_V01",
            "EVENT AWARE",
            "2023-07-29",
            "2025-07-28"
        ),
        (
            "XGBOOST_WEATHER_V01",
            "WEATHER AWARE",
            "2023-07-29",
            "2025-07-28"
        ),
        (
            "OLS_ELASTICITY_V01",
            "ELASTICITY",
            "2023-07-29",
            "2025-07-28"
        ),
        (
            "XGBOOST_INVENTORY_AWARE_ADJUSTMENTS_V01",
            "INVENTORY AWARE",
            "2024-01-01",
            "2025-03-31"
        ),
        (
            "AUTO_ARIMA_x_XGBOOST_BASE_V01",
            "ENSEMBLE",
            "2023-07-29",
            "2025-07-28"
        ),
    ],
    ["model_version", "model_type", "training_start_date", "training_end_date"]
).withColumn(
    "training_start_date", F.to_date("training_start_date")
).withColumn(
    "training_end_date", F.to_date("training_end_date")
).withColumn(
    "created_at", F.current_timestamp()
)

display(dim_forecast_model)

# COMMAND ----------

(dim_forecast_model.write
 .format("delta")
 .mode("overwrite")
 .saveAsTable("aurora.gold.dim_forecast_model")
)

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO aurora.gold.dim_forecast_model (model_version, model_type, training_start_date, training_end_date, created_at)
# MAGIC VALUES (
# MAGIC   'AUTO_ARIMA_BASE_V01',
# MAGIC   'Baseline',
# MAGIC   current_timestamp(),
# MAGIC   timestampadd(MINUTE, 20, current_timestamp()),
# MAGIC   current_timestamp()
# MAGIC )