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
# MAGIC       <td><code>aurora.silver.store_dim</code></td>
# MAGIC       <td>INPUT</td>
# MAGIC       <td>Source of store attributes (name, region, city, state, country)</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><code>aurora.gold.dim_location</code></td>
# MAGIC       <td>OUTPUT</td>
# MAGIC       <td>Gold location dimension</td>
# MAGIC     </tr>
# MAGIC   </tbody>
# MAGIC </table>

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *

# COMMAND ----------

INPUT_TABLE_STORE_DIM = "aurora.silver.store_dim"
OUTPUT_TABLE_DIM_LOCATION = "aurora.gold.dim_location"

# Important variables / knobs
COUNTRY_CODE_FILTER = "US"

# Optional store filter (set to None to load all US stores)
STORECODE_ALLOWLIST = [10425, 10424, 11384]

# Manual rows to union (used when store_dim is missing specific locations)
MANUAL_LOCATIONS = [
    {"storecode": 11059, "storename": "Twelve Oaks, MI", "regioncode": "1009 - Midwest", "city": "Novi", "statedescription": "Michigan"},
    {"storecode": 11082, "storename": "International Plaza, FL", "regioncode": "1017 - Florida", "city": "Tampa", "statedescription": "Florida"},
    {"storecode": 11451, "storename": "Parkdale, TX", "regioncode": "1099 - Undecided - US-US", "city": "Beaumont", "statedescription": "Texas"},
]

WRITE_MODE = "overwrite"  # dimension full refresh
WRITE_FORMAT = "delta"
OVERWRITE_SCHEMA = True

# COMMAND ----------

def build_dim_location(store_dim_df):
    df = store_dim_df.filter(F.col("countrycode") == F.lit(COUNTRY_CODE_FILTER))
    if STORECODE_ALLOWLIST is not None:
        df = df.filter(F.col("storecode").isin([int(x) for x in STORECODE_ALLOWLIST]))

    base = df.select(
        F.col("storecode").cast("string").alias("location_id"),
        F.col("storename").alias("location_name"),
        F.lit("BANDM").alias("location_type"),
        F.col("regioncode").alias("region"),
        F.col("city").alias("city"),
        F.col("statedescription").alias("state"),
        F.current_timestamp().alias("created_at"),
    )

    if MANUAL_LOCATIONS:
        manual_df = spark.createDataFrame(MANUAL_LOCATIONS).select(
            F.col("storecode").cast("string").alias("location_id"),
            F.col("storename").alias("location_name"),
            F.lit("BANDM").alias("location_type"),
            F.col("regioncode").alias("region"),
            F.col("city").alias("city"),
            F.col("statedescription").alias("state"),
            F.current_timestamp().alias("created_at"),
        )
        base = base.unionByName(manual_df)

    # Deduplicate by location_id, preferring store_dim rows over manual if both exist
    base = base.dropDuplicates(["location_id"])
    return base

# COMMAND ----------

def main():
    store_dim_df = spark.table(INPUT_TABLE_STORE_DIM)
    dim_location_final = build_dim_location(store_dim_df=store_dim_df)

    writer = dim_location_final.write.format(WRITE_FORMAT).mode(WRITE_MODE)
    if OVERWRITE_SCHEMA:
        writer = writer.option("overwriteSchema", "true")
    writer.saveAsTable(OUTPUT_TABLE_DIM_LOCATION)


if __name__ == "__main__":
    main()