# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC - **aurora.gold.dim_forecast_model** is formed using hardcoded manual inserts
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