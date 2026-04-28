# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC <h3>Databricks tables used</h3>
# MAGIC <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; width:100%">
# MAGIC   <thead>
# MAGIC     <tr>
# MAGIC       <th align="left">Table / Model</th>
# MAGIC       <th align="left">Role</th>
# MAGIC       <th align="left">How it’s used</th>
# MAGIC     </tr>
# MAGIC   </thead>
# MAGIC   <tbody>
# MAGIC     <tr>
# MAGIC       <td><code>aurora.silver.aurora_filtered_sales_actuals_2023_2026</code></td>
# MAGIC       <td>INPUT</td>
# MAGIC       <td>Cluster-level weekly sales history used for training</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><code>aurora.gold.aurora_baseline_forecast_cluster_march_2026_oct_2026</code></td>
# MAGIC       <td>OUTPUT</td>
# MAGIC       <td>Cluster-level baseline forecast (weekly)</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><code>aurora.gold.aurora_sku_weights_lookup_table</code></td>
# MAGIC       <td>OUTPUT (optional)</td>
# MAGIC       <td>SKU weights lookup (can be produced here, but recommended via its dedicated notebook)</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><code>aurora.default.aurora_sku_forecast_march_2026_oct_2026</code></td>
# MAGIC       <td>OUTPUT (optional)</td>
# MAGIC       <td>SKU-level forecast derived from cluster forecast x sku weights</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><code>aurora.gold.baseline_forecast_model</code></td>
# MAGIC       <td>MODEL</td>
# MAGIC       <td>MLflow registered model name used for training/serving</td>
# MAGIC     </tr>
# MAGIC   </tbody>
# MAGIC </table>

# COMMAND ----------

INPUT_TABLE_SALES_ACTUALS = "aurora.silver.aurora_filtered_sales_actuals_2023_2026"

OUTPUT_TABLE_CLUSTER_FORECAST = "aurora.gold.aurora_baseline_forecast_cluster_march_2026_oct_2026"
OUTPUT_TABLE_SKU_WEIGHTS = "aurora.gold.aurora_sku_weights_lookup_table"
OUTPUT_TABLE_SKU_FORECAST = "aurora.default.aurora_sku_forecast_march_2026_oct_2026"

MLFLOW_MODEL_NAME = "aurora.gold.baseline_forecast_model"

# Important variables / knobs
TRAIN_END_DATE_INCLUSIVE = "2026-03-22"
FORECAST_HORIZON_WEEKS = 30

# Whether to register/update the MLflow model (requires MLflow registry permissions)
LOG_AND_REGISTER_MODEL = True

# Optional downstream outputs (usually run via dedicated notebooks/jobs)
WRITE_SKU_WEIGHTS = False
WRITE_SKU_FORECAST = False

WRITE_MODE = "overwrite"

# COMMAND ----------

import pandas as pd
import numpy as np
from hierarchicalforecast.utils import aggregate
from hierarchicalforecast.methods import MinTrace, BottomUp, TopDown
from hierarchicalforecast.core import HierarchicalReconciliation
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import warnings
from typing import List, Optional
warnings.filterwarnings("ignore")
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from functools import reduce
import operator
import datetime as dt

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import pickle
import os
import tempfile
import joblib
import json

# COMMAND ----------

class BaselineForecastWrapper(mlflow.pyfunc.PythonModel):
    
    def load_context(self, context):
        import joblib
        import json

        self.model = joblib.load(context.artifacts["model"])
        self.Y_df_train = joblib.load(context.artifacts["Y_df_train"])
        self.S_df = joblib.load(context.artifacts["S_df_train"])

        with open(context.artifacts["tags"], "r") as f:
            self.tags = json.load(f)

        with open(context.artifacts["group_cols"], "r") as f:
            self.group_cols = json.load(f)

    def predict(self, context, model_input: pd.DataFrame):
        
        if "h" not in model_input.columns:
            raise ValueError("Input must conatin column 'h'")

        h = int(model_input["h"].iloc[0])

        base_fcst = self.model.predict(h=h)

        # Setup Reconcilers
        reconcilers = [BottomUp(), TopDown(method='average_proportions'), MinTrace(method='mint_shrink')]

        # Create Reconciler object
        reconciler = HierarchicalReconciliation(reconcilers=reconcilers)

        # Generate Reconciled forecast for defined horizon
        y_hat = base_fcst[['unique_id', 'ds', "AutoARIMA"]].rename(columns={"AutoARIMA": 'y'})

        reconciled_fcst = reconciler.reconcile(
            Y_hat_df=y_hat,
            Y_df=self.Y_df_train,
            S_df=self.S_df,
            tags=self.tags
        )

        # Extract forecast by Hierarchy levels
        def extract_forecasts_by_level(forecast_df, tags):
            results = {}
            for level, level_tags in tags.items():
                level_forecasts = forecast_df[forecast_df['unique_id'].isin(level_tags)]
                results[level] = level_forecasts

            return results
        
        fcst_by_level = extract_forecasts_by_level(reconciled_fcst, self.tags)

        # Join the Group columns to show single string unique ID
        def join_group_cols(group_cols):
            return "/".join(group_cols)

        result = fcst_by_level[join_group_cols(self.group_cols)]

        result[self.group_cols] = result['unique_id'].str.split('/', expand=True)

        # Define base week and cutoff week
        base_week = result['ds'].max() + dt.timedelta(days=1)

        # Result DataFrame
        final_df = result[self.group_cols + ['ds', 'y']]
        final_df.rename(columns={'y': 'forecast_qty', 'ds': 'weekstartdate'}, inplace=True)

        # Metadata
        final_df['base_week'] = base_week
        final_df['base_year'] = base_week.year
        final_df['time_grain'] = 'WEEKLY'
        final_df['forecast_type'] = 'BASE'
        final_df['forecast_qty'] = final_df['forecast_qty'].round(3) 
        final_df['modelrunid'] = 'ARIMA_BASE_V01'
        final_df['createtimestamp'] = dt.datetime.now()
        final_df['updatetimestamp'] = dt.datetime.now()
        final_df['activeflag'] = 'Y'
        final_df['period'] = final_df['weekstartdate'].rank(method='dense').astype(int)

        final_df = final_df.reset_index(drop=True)

        final_df = final_df[self.group_cols + ['base_week', 'weekstartdate', 'period', 'forecast_qty', 'forecast_type', 'time_grain', 'modelrunid', 'createtimestamp', 'updatetimestamp', 'activeflag', 'base_year']]

        for col in self.group_cols:
            final_df[col] = final_df[col].astype(str)

        return final_df

class BaselineForecastClient:

    def __init__(self, model_uri: str):
        self.model = mlflow.pyfunc.load_model(model_uri)

    def predict(self, h: int = 52):
        import pandas as pd
        return self.model.predict(pd.DataFrame({"h": [h]}))

class BaselineForecast:
    def __init__(
        self, 
        df: pd.DataFrame,
        model_name: str, 
        date_col: str = "next_sunday", 
        sales_col: str = "sales_quantity", 
        group_cols: list = ["store", "department_code", "cluster_code"], 
        hierarchy_cols: Optional[List[List[str]]] = None,
        freq: str = "W-SUN",
        season_length: int = 52 
        ):

        """
        Initializes BaselineForecast.

        Args:
            df (pd.DataFrame): Input DataFrame containing sales history.
            date_col (str): Name of the date column.
            sales_col (str): Name of the sales quantity column.
            group_cols (list): Optional list of columns to group by (e.g., ['store', 'department_code']).
            hierarchy_cols (list): Optional list of columns to define the hierarchy specification (e.g., ['store'], ['store', 'department_code']).
        """
        
        self.df = df
        self.date_col = date_col
        self.sales_col = sales_col
        self.group_cols = group_cols
        self.model_name = model_name
        self.freq = freq
        self.season_length = season_length

        # Validation Check
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Dataframe {self.df} must be Pandas DataFrame.")

        # Datetime check
        if self.df[self.date_col].dtype != "datetime64[ns]":
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], format="%Y-%m-%d", errors="coerce")
        
        # Null Datetime value check
        if self.df[self.date_col].isna().any():
            raise ValueError(f"Datetime column {self.date_col} contains null values.")

        # Null Group columns check
        if self.group_cols is None:
            raise ValueError("Group columns must be provided.")
        
        # Generate Hierarchy Columns
        if hierarchy_cols is None:
            self.hierarchy_cols = [self.group_cols[:i] for i in range(1, len(self.group_cols) + 1)]
        else:
            self.hierarchy_cols = hierarchy_cols

        # Handle null Sales Quantity
        self.df[self.sales_col].fillna(0, inplace=True)
            
        # Sort the DataFrame by Sales date 
        self.df = self.df.sort_values(self.date_col)

        self.model = None
        self.Y_df_train = None
        self.S_df = None
        self.tags = None
        self.last_train_date = None
    
    def fit(self):
        
        train_df = self.df.rename(columns={self.sales_col: 'y', self.date_col: 'ds'})

        train_df['y'] = np.log1p(train_df['y'])

        # Drop duplicate values
        unique_keys = train_df[self.group_cols].drop_duplicates()

        # Create data range for each hierarchy level
        all_dates = pd.date_range(start=train_df['ds'].min(), end=train_df['ds'].max(), freq=self.freq)

        # Create MultiIndex with combinations of hierarchy levels and dates
        all_dates_df = pd.DataFrame({'ds': all_dates})

        # Reindex DataFrame
        train_df_full = unique_keys.join(all_dates_df, how='cross')

        train_df = train_df_full.merge(train_df, on=self.group_cols + ['ds'], how='left')

        # Handle null values in Train dataset
        train_df.fillna(0, inplace=True)

        # Aggregate data based on defined Hierarchy
        Y_df_train, S_df, tags = aggregate(train_df, self.hierarchy_cols)

        self.Y_df_train = Y_df_train
        self.S_df = S_df
        self.tags = tags
        self.last_train_date = Y_df_train['ds'].max()

        print("Last train date:", self.last_train_date)

        # Train AutoARIMA model 
        model = [AutoARIMA(season_length=self.season_length)]
        self.model = StatsForecast(models=model, freq=self.freq, n_jobs=-1)
        self.model.fit(df=Y_df_train)
    
    def predict(self, h: int = 30):

        if self.model is None:
            raise ValueError("ERROR: Model not fitted.")

        # HIERARCHICAL RECONCILIATION

        # Fitted values for Hierarchical Reconciliation
        base_fcst = self.model.predict(h=h)

        # Setup Reconcilers
        reconcilers = [BottomUp(), TopDown(method='average_proportions'), MinTrace(method='mint_shrink')]

        # Create Reconciler object
        reconciler = HierarchicalReconciliation(reconcilers=reconcilers)

        # Generate Reconciled forecast for defined horizon
        y_hat = base_fcst[['unique_id', 'ds', "AutoARIMA"]].rename(columns={"AutoARIMA": 'y'})

        reconciled_fcst = reconciler.reconcile(
            Y_hat_df=y_hat,
            Y_df=self.Y_df_train,
            S_df=self.S_df,
            tags=self.tags
        )

        # Extract forecast by Hierarchy levels
        def extract_forecasts_by_level(forecast_df, tags):
            results = {}
            for level, level_tags in tags.items():
                level_forecasts = forecast_df[forecast_df['unique_id'].isin(level_tags)]
                results[level] = level_forecasts

            return results
        
        fcst_by_level = extract_forecasts_by_level(reconciled_fcst, self.tags)

        # Join the Group columns to show single string unique ID
        def join_group_cols(group_cols):
            return "/".join(group_cols)

        result = fcst_by_level[join_group_cols(self.group_cols)]

        result[self.group_cols] = result['unique_id'].str.split('/', expand=True)

        # Define base week
        base_week = self.last_train_date

        # Result DataFrame
        final_df = result[self.group_cols + ['ds', 'y']]
        final_df.rename(columns={'y': 'forecast_qty', 'ds': 'weekstartdate'}, inplace=True)

        # Metadata
        final_df['base_week'] = base_week
        final_df['base_year'] = base_week.year
        final_df['time_grain'] = 'WEEKLY'
        final_df['forecast_type'] = 'BASE'
        final_df['forecast_qty'] = np.expm1(final_df['forecast_qty']).round(3) 
        final_df['modelrunid'] = 'ARIMA_BASE_V01'
        final_df['createtimestamp'] = dt.datetime.now()
        final_df['updatetimestamp'] = dt.datetime.now()
        final_df['activeflag'] = 'Y'
        final_df['period'] = final_df['weekstartdate'].rank(method='dense').astype(int)

        final_df['forecast_qty'] = final_df['forecast_qty'].clip(lower=0)

        final_df = final_df.reset_index(drop=True)

        final_df = final_df[self.group_cols + ['base_week', 'weekstartdate','period', 'forecast_qty', 'forecast_type', 'time_grain', 'modelrunid', 'createtimestamp', 'updatetimestamp', 'activeflag', 'base_year']]

        for col in self.group_cols:
            final_df[col] = final_df[col].astype(str)

        return final_df
    
    def rolling_forecast(
        self,
        h: int = 13,
        min_hist: int = 10,
        start_cutoff: str = None
    ):
        
        results = []

        all_dates = sorted(self.df[self.date_col].unique())

        cutoff_dates = all_dates[min_hist:]

        if start_cutoff is not None:
            start_cutoff = pd.to_datetime(start_cutoff)
            cutoff_dates = [d for d in cutoff_dates if pd.to_datetime(d) >= start_cutoff]

        if not cutoff_dates:
            raise ValueError(f"No cutoff dates found on or after {start_cutoff}. Check your start_cutoff or min_hist.")

        print(f"Cutoff range : {pd.to_datetime(cutoff_dates[0]).date()} → {pd.to_datetime(cutoff_dates[-1]).date()}")
        print(f"Total cutoffs: {len(cutoff_dates)}")

        for cutoff in cutoff_dates:

            cutoff = pd.to_datetime(cutoff)
            print(f"Running cutoff: {cutoff.date()}")

            train_subset = self.df[self.df[self.date_col] <= cutoff].copy()

            try:
                temp_model = BaselineForecast(
                    df=train_subset,
                    model_name=self.model_name,
                    date_col=self.date_col,
                    sales_col=self.sales_col,
                    group_cols=self.group_cols,
                    hierarchy_cols=self.hierarchy_cols,
                    freq=self.freq,
                    season_length=self.season_length
                )

                temp_model.fit()
                fcst = temp_model.predict(h=h)

                fcst["base_week"] = cutoff

                results.append(fcst)
            
            except Exception as e:
                print(f"Eror at {cutoff.date()}: {e}")
                continue
        
        if not results:
            raise ValueError("No forecast generated")

        final_results = pd.concat(results, ignore_index=True)
        final_results = final_results.sort_values(self.group_cols + ['base_week', 'weekstartdate'])

        final_results['offset'] = ((pd.to_datetime(final_results['weekstartdate']) - pd.to_datetime(final_results['base_week'])).dt.days // 7)

        final_results['offset'] = final_results['offset'].astype(int)

        final_results = final_results[
            (final_results['offset'] >= 1) &
            (final_results['offset'] <= h)
        ]

        return final_results

    def log_model(self):
        
        joblib.dump(self.model, "baseline_model.pkl")
        joblib.dump(self.Y_df_train, "baseline_Y_train.pkl")
        joblib.dump(self.S_df, "baseline_S_df.pkl")

        tags_serializable = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in self.tags.items()
        }

        with open("baseline_tags.json", "w") as f:
            json.dump(tags_serializable, f)

        with open("baseline_group_cols.json", "w") as f:
            json.dump(self.group_cols, f)

        input_example = pd.DataFrame({"h": [1]})
        sample_output = self.predict(h=1)

        signature = infer_signature(input_example, sample_output)

        with mlflow.start_run() as run:

            mlflow.pyfunc.log_model(
                artifact_path="baseline_forecast_model",
                python_model=BaselineForecastWrapper(),
                artifacts={
                    "model": "baseline_model.pkl",
                    "Y_df_train": "baseline_Y_train.pkl",
                    "S_df_train": "baseline_S_df.pkl",
                    "tags": "baseline_tags.json",
                    "group_cols": "baseline_group_cols.json"
                },
                signature=signature,
                input_example=input_example,
                pip_requirements=[
                    "pandas",
                    "numpy",
                    "joblib",
                    "statsforecast",
                    "hierarchicalforecast"
                ]
            )

            model_info = mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/baseline_forecast_model", name=self.model_name)

            client = MlflowClient()
            
            client.set_registered_model_alias(
                name=self.model_name,
                alias="challenger",
                version=model_info.version
            )
    
    @staticmethod
    def load_model(model_name: str):
        
        return BaselineForecastClient(f"models:/{model_name}@challenger")
    
    def update(self, new_df: pd.DataFrame):

        if isinstance(new_df, spark.DataFrame):
            new_df = new_df.toPandas()

        self.df = pd.concat([self.df, new_df])
        self.df = self.df.sort_values(self.date_col)

        # Retrain model on new data
        self.fit()

        # Log updated model
        self.log_model() 

# COMMAND ----------

from pyspark.sql.window import Window


def prepare_training_data(actuals_df):
    input_df = actuals_df.filter(F.col("next_sunday") <= F.lit(TRAIN_END_DATE_INCLUSIVE))
    train_df = (
        input_df.groupBy("store", "department_code", "cluster_code", "next_sunday")
        .agg(F.sum(F.col("sales_quantity")).alias("sales_quantity"))
    )
    return train_df


def train_model(train_spark_df):
    df_train = train_spark_df.toPandas()
    for col in ["store", "department_code", "cluster_code"]:
        df_train[col] = df_train[col].astype(str)

    bf = BaselineForecast(df_train, model_name=MLFLOW_MODEL_NAME)
    bf.fit()
    if LOG_AND_REGISTER_MODEL:
        bf.log_model()
    return bf


def write_cluster_forecast(bf: BaselineForecast):
    df_forecast = bf.predict(h=FORECAST_HORIZON_WEEKS)
    forecast_df = spark.createDataFrame(df_forecast)
    forecast_df.write.mode(WRITE_MODE).saveAsTable(OUTPUT_TABLE_CLUSTER_FORECAST)
    return forecast_df


def build_sku_weights(actuals_df):
    sku_sales = actuals_df.groupBy("store", "department_code", "cluster_code", "sku_code").agg(
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
        .select("store", "department_code", "cluster_code", "sku_code", "sku_weight")
    )
    return weights


def write_sku_forecast(cluster_forecast_df, sku_weights_df):
    base_fcst_df = cluster_forecast_df.select(
        "store",
        "department_code",
        "cluster_code",
        F.to_date(F.col("weekstartdate")).alias("next_sunday"),
        "forecast_qty",
    )

    sku_forecast_df = (
        base_fcst_df.join(sku_weights_df, on=["store", "department_code", "cluster_code"], how="left")
        .withColumn("sku_weight", F.coalesce(F.col("sku_weight"), F.lit(0.0)))
        .withColumn("forecast_qty", F.round(F.col("forecast_qty") * F.col("sku_weight"), 3))
        .select("store", "department_code", "cluster_code", "sku_code", "next_sunday", "forecast_qty")
    )

    sku_forecast_df.write.mode(WRITE_MODE).saveAsTable(OUTPUT_TABLE_SKU_FORECAST)
    return sku_forecast_df


def main():
    actuals_df = spark.table(INPUT_TABLE_SALES_ACTUALS)

    train_df = prepare_training_data(actuals_df=actuals_df)
    bf = train_model(train_spark_df=train_df)
    cluster_forecast_df = write_cluster_forecast(bf=bf)

    if WRITE_SKU_WEIGHTS or WRITE_SKU_FORECAST:
        sku_weights_df = build_sku_weights(actuals_df=actuals_df)

        if WRITE_SKU_WEIGHTS:
            sku_weights_df.write.mode(WRITE_MODE).saveAsTable(OUTPUT_TABLE_SKU_WEIGHTS)

        if WRITE_SKU_FORECAST:
            write_sku_forecast(cluster_forecast_df=cluster_forecast_df, sku_weights_df=sku_weights_df)


main()