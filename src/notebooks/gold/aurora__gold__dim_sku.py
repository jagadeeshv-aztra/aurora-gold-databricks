# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC - **aurora.gold.dim_sku** is formed using below tables: 
# MAGIC
# MAGIC - aurora.silver.article
# MAGIC - aurora.silver.aurora_filtered_sales_actuals_2023_2026

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# COMMAND ----------

# DBTITLE 1,sales_actuals_df
sales_actuals_df = spark.table("aurora.silver.aurora_filtered_sales_actuals_2023_2026")
HIER_COLS    = ["store", "department_code", "cluster_code", "sku_code"]  # Hierarchy columns

sales_actuals_df = (
    sales_actuals_df
    .withColumn("sku_code", F.col("sku_code").cast(StringType()))
    .withColumn("store", F.col("store").cast(StringType()))
    .withColumn("department_code", F.col("department_code").cast(StringType()))
)
display(sales_actuals_df)

# COMMAND ----------

actuals_df = (
    sales_actuals_df
    .groupBy("store", "department_code", "cluster_code", "sku_code")
    .agg(
        F.round(F.sum("selling_price") / F.sum("sales_quantity"), 2).alias("unit_cost")
    )
)
display(actuals_df)

# COMMAND ----------

# DBTITLE 1,sku_item_xref_df
sku_item_xref_df = spark.table("aurora.silver.sku_item_xref_lkp")

sku_item_xref_df = sku_item_xref_df.withColumnsRenamed({
    "skucode": "sku_code",
    "itemcode": "item_code",
    "stylecode": "style_code",
    "colorcode": "color_code",
    "sizecode": "size_code"
})
display(sku_item_xref_df)

# COMMAND ----------

# Check if same item+style+color+size has multiple sku_codes
sku_item_xref_df.groupBy("item_code", "style_code", "color_code", "size_code") \
  .agg(
      F.countDistinct("sku_code").alias("unique_skus"),
      F.collect_set("sku_code").alias("sku_codes")
  ) \
  .filter(F.col("unique_skus") > 1) \
  .show(10, truncate=False)

# COMMAND ----------

# DBTITLE 1,sku_lifecycle
# First and last sale date per SKU
sku_dates = (
    sales_actuals_df
    .groupBy("sku_code")
    .agg(
        F.min("next_sunday").alias("first_sale_date"),
        F.max("next_sunday").alias("last_sale_date")
    )
)

# Get max available sales date
as_of_date = (
    sales_actuals_df
    .agg(F.max("next_sunday").alias("max_date"))
    .collect()[0]["max_date"]
)

# Synthetic launch_date
sku_dates = sku_dates.withColumn(
    "launch_date",
    F.col("first_sale_date")
)

current_date = F.lit(as_of_date)

sku_dates = sku_dates.withColumn(
    "days_since_last_sale",
    F.datediff(current_date, F.col("last_sale_date"))
)

sku_dates = sku_dates.withColumn(
    "discontinue_date",
    F.when(F.col("days_since_last_sale") > 90, F.col("last_sale_date"))
     .otherwise(F.lit(None))
)

# Adjust windows to week multiples
sales_26w = sales_actuals_df.filter(
    F.col("next_sunday") >= F.date_sub(F.lit(as_of_date), 26 * 7)
)
sales_13w = sales_actuals_df.filter(
    F.col("next_sunday") >= F.date_sub(F.lit(as_of_date), 13 * 7)
)
sales_4w = sales_actuals_df.filter(
    F.col("next_sunday") >= F.date_sub(F.lit(as_of_date), 4 * 7)
)

sales_26w_agg = sales_26w.groupBy("sku_code").agg(F.sum("sales_quantity").alias("last_26w_sales"))
sales_13w_agg = sales_13w.groupBy("sku_code").agg(F.sum("sales_quantity").alias("last_13w_sales"))
sales_4w_agg  = sales_4w.groupBy("sku_code").agg(F.sum("sales_quantity").alias("last_4w_sales"))

sku_lifecycle = (
    sku_dates
    .join(sales_26w_agg, "sku_code", "left")
    .join(sales_13w_agg, "sku_code", "left")
    .join(sales_4w_agg,  "sku_code", "left")
    .fillna(0, ["last_26w_sales", "last_13w_sales", "last_4w_sales"])
)

sku_lifecycle = sku_lifecycle.withColumn(
    "lifecycle_status",
    # DISCONTINUED — no sales for 26 weeks
    F.when(F.col("days_since_last_sale") > 182, "DISCONTINUED")

    # NEW — first sale within last 13 weeks
    .when(
        F.datediff(F.lit(as_of_date), F.col("launch_date")) <= 91,
        "NEW"
    )

    # PHASE_OUT — no sales in last 4 weeks but had sales in last 13 weeks
    .when(
        (F.col("last_4w_sales") == 0) & (F.col("last_13w_sales") > 0),
        "PHASE_OUT"
    )

    # OBSOLETE — no sales in last 26 weeks but not yet discontinued
    .when(F.col("last_26w_sales") == 0, "OBSOLETE")

    # ACTIVE
    .otherwise("ACTIVE")
)

display(sku_lifecycle)

# COMMAND ----------

# Check if both old and new sku_codes appear in actuals
old_new_skus = sku_item_xref_df.groupBy("item_code", "style_code", "color_code", "size_code") \
  .agg(
      F.min("sku_code").alias("old_sku"),
      F.max("sku_code").alias("new_sku")
  )

actuals_skus = sales_actuals_df.select("sku_code").distinct()

# How many old skus still exist in actuals?
old_new_skus.join(actuals_skus, old_new_skus["old_sku"] == actuals_skus["sku_code"], "inner") \
  .count()

# COMMAND ----------

# DBTITLE 1,join with sku_ref
sales_join_df = actuals_df.join(
    sku_item_xref_df,
    on=["sku_code"],
    how="left"
)

display(sales_join_df)
print(sales_join_df.count())

# COMMAND ----------

# Check if same item+style+color+size has multiple sku_codes
sales_join_df.groupBy("item_code", "style_code", "color_code", "size_code") \
  .agg(
      F.countDistinct("sku_code").alias("unique_skus"),
      F.collect_set("sku_code").alias("sku_codes")
  ) \
  .filter(F.col("unique_skus") > 1) \
  .show(10, truncate=False)

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import StringType

article_df = spark.table("aurora.silver.article")

for col in article_df.columns:
    article_df = article_df.withColumnsRenamed({col: col.lower()})

article_df = (
    article_df
    .withColumn("item_code", F.col("item_code").cast(StringType()))
    .withColumn("color_code", F.col("color_code").cast(StringType()))
    .withColumn("style_code", F.col("style_code").cast(StringType()))
).dropDuplicates()

display(article_df)

print(article_df.count())

# COMMAND ----------

article_df = article_df.withColumn(
    "class_desc",
    F.when(
        F.col("class_desc").contains("DO NOT USE"),
        F.trim(F.regexp_replace("class_desc", r"\s*DO NOT USE\s*", ""))
    ).otherwise(F.col("class_desc")).alias("class_desc")
)

# COMMAND ----------

display(article_df.filter(F.col("department_desc").contains("LV")))

# COMMAND ----------

article_x_ref = article_df.join(
    sku_item_xref_df,
    on=["item_code", "style_code", "color_code"],
    how="left"
)

article_x_ref = article_x_ref.withColumn(
    "sku_name",
    F.initcap(
        F.concat_ws(
            " ",
            F.col("class_desc"),
            F.col("color_desc"),
            F.col("size_code"),
            F.lit(" - "),
            F.col("sku_code")
        )
    )
)
# class_desc + color_desc + size_code
# FASHION FIT DARK 282

# article_x_ref = article_x_ref.withColumn("sku_name", F.concat(F.col("sku_name"), F.lit(" - "), F.col("sku_code")))

display(article_x_ref)
print(article_x_ref.count())

# COMMAND ----------

# color_lookup_pd = pd.read_csv("/Workspace/Users/viveik.cataram@aztra.ai/color_lookup_resolved.csv")
# color_lookup_spark_df = (
#     spark.createDataFrame(color_lookup_pd)
#     .withColumn("color_code", F.col("color_code").cast("bigint"))
# )
# for col in color_lookup_spark_df.columns:
#     color_lookup_spark_df = color_lookup_spark_df.withColumnRenamed(col, col.upper())

# color_lookup_spark_df = color_lookup_spark_df.withColumnRenamed("COLOR_DESC", "COLOR_DESC_NEW")
# display(color_lookup_spark_df)

# COMMAND ----------

# # Update color_desc in aurora.silver.article using color_desc from color_lookup_spark_df
# from pyspark.sql.functions import expr

# article_silver_df = spark.table("aurora.silver.article").alias("sc")
# article_silver_df.printSchema()
# updated_article_df = (
#     article_df
#     .join(
#         color_lookup_spark_df.alias("cref").select("COLOR_CODE", "COLOR_DESC_NEW"),
#         on="COLOR_CODE",
#         how="left"
#     )
#     .withColumn(
#         "COLOR_DESC", expr("cref.COLOR_DESC_NEW")
#     ).drop("COLOR_DESC_NEW")
# )

# display(updated_article_df)

# updated_article_df.dropDuplicates().write.mode("overwrite").saveAsTable("aurora.silver.article")



# COMMAND ----------

# color_lookup_df = article_df.select("color_code", "color_desc").where("color_desc IS NOT NULL").distinct()
# display(color_lookup_df)

# dup_color_df = color_lookup_df.groupBy("color_code").count().where("count > 1")
# dup_joined_df = color_lookup_df.join(dup_color_df, on=["color_code"], how="inner")
# display(dup_joined_df)

# COMMAND ----------

from pyspark.sql.functions import expr
sales_article_joined_df = sales_join_df.join(
    article_x_ref.select(
        "sku_code",
        "sku_name",
        "item_code",
        "item_desc",
        "style_code",
        "color_code",
        "size_code",
        "department_desc",
        "cluster_desc",
        "brand_code",
        "brand_desc",
        "color_desc"
    ),
    on=["sku_code", "item_code", "style_code", "color_code", "size_code"],
    how="left"
)

display(sales_article_joined_df)

# COMMAND ----------

display(
    sales_article_joined_df.groupBy("sku_name")
    .agg(F.countDistinct("sku_code").alias("distinct_sku_count"))
    .filter(F.col("distinct_sku_count") > 1)
    .orderBy(F.col("distinct_sku_count").desc())
)

# COMMAND ----------

df_fixed = sales_article_joined_df.dropDuplicates(["sku_code"])

display(df_fixed)

# COMMAND ----------

df_sku_lifecycle = df_fixed.join(
    sku_lifecycle.select(
        "sku_code",
        "launch_date",
        "discontinue_date",
        "lifecycle_status",
    ),
    on="sku_code",
    how="left"    
)

display(df_sku_lifecycle)

# COMMAND ----------

final_df = df_sku_lifecycle.withColumnsRenamed({"sku_code": "sku_id", "color_code": "cc_code", "color_desc": "color_name"})

final_df = (
    final_df
    .withColumn("cluster_desc_clean", F.split(F.col("cluster_desc"), ":").getItem(1))
    .withColumn("subcategory", F.concat_ws("_", F.col("cluster_code"), F.col("cluster_desc_clean")))
    .withColumn("category", F.concat(F.col("department_code"), F.lit(":"), F.col("department_desc")))
    .withColumn("brand", F.concat(F.col("brand_code"), F.lit(":"), F.col("brand_desc")))
    .withColumn("base_selling_price", F.round(F.col("unit_cost") * (1.8 + (F.rand() * 1.2)), 2))
).drop("cluster_desc_clean")

display(final_df)


# COMMAND ----------

dim_sku_df = final_df.select(
    "sku_id",
    "sku_name",
    "category",
    "subcategory",
    "brand",
    "cc_code",
    "color_name",
    "lifecycle_status",
    "launch_date",
    "discontinue_date",
    F.current_timestamp().alias("created_at"),
    "unit_cost",
    "base_selling_price",
    F.col("item_desc").alias("item_name")
)

display(dim_sku_df)

# COMMAND ----------

display(
    dim_sku_df.groupBy("sku_id")
    .count()
    .filter(F.col("count") > 1)
    .orderBy(F.col("count").desc())
)

# COMMAND ----------

display(
    dim_sku_df.groupBy("item_name")
    .agg(F.countDistinct("sku_id").alias("distinct_sku_count"))
    .filter("distinct_sku_count > 1")
)

# COMMAND ----------

dim_sku_df_fixed = dim_sku_df.dropDuplicates(["sku_id"])
# dim_sku_df.write.mode("overwrite").saveAsTable("aurora.silver.dim_sku")

display(dim_sku_df_fixed)
print(dim_sku_df_fixed.count())

# COMMAND ----------

dim_sku_df_fixed = dim_sku_df_fixed.withColumn("sku_name", F.concat(F.col("sku_name"), F.lit(" - "), F.col("sku_id")))

display(dim_sku_df_fixed)

# COMMAND ----------

(dim_sku_df.write
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable("aurora.gold.dim_sku")
)

# COMMAND ----------

display(
    dim_sku_df_fixed.groupBy("sku_name")
    .agg(F.countDistinct("sku_id").alias("distinct_sku_count"))
    .filter("distinct_sku_count > 1")
)

# COMMAND ----------

# dim_df = spark.table("aurora.gold.dim_sku_fixed")

# display(dim_df)
display(dim_df.filter(F.col("color_name") == "ONYX SD/TEXTURE"))

# COMMAND ----------



# COMMAND ----------

# sales_article_df = actuals.join(
#     article_size.select(
#         "sku_code",
#         "item_code",
#         "style_code",
#         "color_code",
#         "size_code",
#         F.concat(F.col("ITEM_DESC"), F.lit(" "), F.col("size_code")).alias("sku_name"),
#         "DEPARTMENT_DESC",
#         "CLUSTER_DESC",
#         "BRAND_CODE",
#         "BRAND_DESC",
#         F.col("COLOR_DESC").alias("color_name")
#     ),
#     on=["sku_code"],
#     how="left"    
# )

# display(sales_article_df)