# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>Databricks tables used</h3>
# MAGIC <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; width:100%">
# MAGIC <thead>
# MAGIC <tr>
# MAGIC <th align="left">Table</th>
# MAGIC <th align="left">Role</th>
# MAGIC <th align="left">How it’s used</th>
# MAGIC </tr>
# MAGIC </thead>
# MAGIC <tbody>
# MAGIC <tr>
# MAGIC <td><code>aurora.gold.dim_forecast_model</code></td>
# MAGIC <td>OUTPUT</td>
# MAGIC <td>Forecast model dimension populated with a fixed seed set</td>
# MAGIC </tr>
# MAGIC </tbody>
# MAGIC </table>

# COMMAND ----------

OUTPUT_TABLE_DIM_FORECAST_MODEL = "aurora.gold.dim_forecast_model"

# If this notebook is run as a Databricks Job, keep writes deterministic/idempotent.
WRITE_MODE = "overwrite"  # "overwrite" for full refresh dimension
WRITE_FORMAT = "delta"

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
 .format(WRITE_FORMAT)
 .mode(WRITE_MODE)
 .saveAsTable(OUTPUT_TABLE_DIM_FORECAST_MODEL)
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Note:** the old `%sql INSERT` cell was removed to keep this job idempotent (no duplicate inserts). Use the seed DataFrame above as the single source of truth.
