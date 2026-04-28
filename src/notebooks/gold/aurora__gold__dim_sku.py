# Databricks notebook source
# COMMAND ----------
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
# MAGIC       <td>Unit cost + lifecycle metrics derived from weekly sales</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><code>aurora.silver.sku_item_xref_lkp</code></td>
# MAGIC       <td>INPUT</td>
# MAGIC       <td>Maps SKU ↔ item/style/color/size attributes for article enrichment</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><code>aurora.silver.article</code></td>
# MAGIC       <td>INPUT</td>
# MAGIC       <td>Product master attributes (class/department/brand/color)</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><code>aurora.gold.dim_sku</code></td>
# MAGIC       <td>OUTPUT</td>
# MAGIC       <td>Gold SKU dimension</td>
# MAGIC     </tr>
# MAGIC   </tbody>
# MAGIC </table>

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# COMMAND ----------

INPUT_TABLE_SALES_ACTUALS = "aurora.silver.aurora_filtered_sales_actuals_2023_2026"
INPUT_TABLE_SKU_ITEM_XREF = "aurora.silver.sku_item_xref_lkp"
INPUT_TABLE_ARTICLE = "aurora.silver.article"
OUTPUT_TABLE_DIM_SKU = "aurora.gold.dim_sku"

# Important variables / knobs
LIFECYCLE_NEW_DAYS = 91
LIFECYCLE_PHASE_OUT_LOOKBACK_WEEKS = 4
LIFECYCLE_NEW_LOOKBACK_WEEKS = 13
LIFECYCLE_OBSOLETE_LOOKBACK_WEEKS = 26
LIFECYCLE_DISCONTINUED_DAYS = 182
DISCONTINUE_AFTER_DAYS_NO_SALE = 90

# Deterministic base price factor: base_selling_price = unit_cost * markup_factor
MARKUP_MIN = 1.8
MARKUP_MAX = 3.0

WRITE_MODE = "overwrite"
WRITE_FORMAT = "delta"
OVERWRITE_SCHEMA = True

# COMMAND ----------

def _normalize_article_columns(article_df):
    # Lowercase columns for consistent downstream references
    for c in article_df.columns:
        article_df = article_df.withColumnRenamed(c, c.lower())
    return article_df


def build_sku_lifecycle(sales_actuals_df):
    # As-of date driven by latest observed next_sunday in the input
    as_of_date = sales_actuals_df.agg(F.max("next_sunday").alias("max_date")).collect()[0]["max_date"]
    if as_of_date is None:
        raise ValueError("next_sunday max_date is null; cannot compute lifecycle.")

    sku_dates = sales_actuals_df.groupBy("sku_code").agg(
        F.min("next_sunday").alias("first_sale_date"),
        F.max("next_sunday").alias("last_sale_date"),
    )

    sku_dates = sku_dates.withColumn("launch_date", F.col("first_sale_date"))
    sku_dates = sku_dates.withColumn("days_since_last_sale", F.datediff(F.lit(as_of_date), F.col("last_sale_date")))
    sku_dates = sku_dates.withColumn(
        "discontinue_date",
        F.when(F.col("days_since_last_sale") > F.lit(DISCONTINUE_AFTER_DAYS_NO_SALE), F.col("last_sale_date")).otherwise(F.lit(None)),
    )

    sales_26w = sales_actuals_df.filter(F.col("next_sunday") >= F.date_sub(F.lit(as_of_date), LIFECYCLE_OBSOLETE_LOOKBACK_WEEKS * 7))
    sales_13w = sales_actuals_df.filter(F.col("next_sunday") >= F.date_sub(F.lit(as_of_date), LIFECYCLE_NEW_LOOKBACK_WEEKS * 7))
    sales_4w = sales_actuals_df.filter(F.col("next_sunday") >= F.date_sub(F.lit(as_of_date), LIFECYCLE_PHASE_OUT_LOOKBACK_WEEKS * 7))

    sales_26w_agg = sales_26w.groupBy("sku_code").agg(F.sum("sales_quantity").alias("last_26w_sales"))
    sales_13w_agg = sales_13w.groupBy("sku_code").agg(F.sum("sales_quantity").alias("last_13w_sales"))
    sales_4w_agg = sales_4w.groupBy("sku_code").agg(F.sum("sales_quantity").alias("last_4w_sales"))

    sku_lifecycle = (
        sku_dates.join(sales_26w_agg, "sku_code", "left")
        .join(sales_13w_agg, "sku_code", "left")
        .join(sales_4w_agg, "sku_code", "left")
        .fillna(0, ["last_26w_sales", "last_13w_sales", "last_4w_sales"])
    )

    sku_lifecycle = sku_lifecycle.withColumn(
        "lifecycle_status",
        F.when(F.col("days_since_last_sale") > F.lit(LIFECYCLE_DISCONTINUED_DAYS), F.lit("DISCONTINUED"))
        .when(F.datediff(F.lit(as_of_date), F.col("launch_date")) <= F.lit(LIFECYCLE_NEW_DAYS), F.lit("NEW"))
        .when((F.col("last_4w_sales") == 0) & (F.col("last_13w_sales") > 0), F.lit("PHASE_OUT"))
        .when(F.col("last_26w_sales") == 0, F.lit("OBSOLETE"))
        .otherwise(F.lit("ACTIVE")),
    )

    return sku_lifecycle.select("sku_code", "launch_date", "discontinue_date", "lifecycle_status")


def build_dim_sku(sales_actuals_df, sku_item_xref_df, article_df):
    # Normalize dtypes for stable joins
    sales_actuals_df = (
        sales_actuals_df.withColumn("sku_code", F.col("sku_code").cast(StringType()))
        .withColumn("store", F.col("store").cast(StringType()))
        .withColumn("department_code", F.col("department_code").cast(StringType()))
        .withColumn("color_code", F.col("color_code").cast(StringType()))
        .withColumn("style_code", F.col("style_code").cast(StringType()))
        .withColumn("item_code", F.col("item_code").cast(StringType()))
        .withColumn("size_code", F.col("size_code").cast(StringType()))
    )

    # Unit cost (weighted avg price proxy)
    unit_cost_df = sales_actuals_df.groupBy("store", "department_code", "cluster_code", "sku_code").agg(
        F.round(F.sum("selling_price") / F.sum("sales_quantity"), 2).alias("unit_cost")
    )

    sku_item_xref_df = sku_item_xref_df.withColumnsRenamed(
        {
            "skucode": "sku_code",
            "itemcode": "item_code",
            "stylecode": "style_code",
            "colorcode": "color_code",
            "sizecode": "size_code",
        }
    )

    article_df = _normalize_article_columns(article_df)
    article_df = (
        article_df.withColumn("item_code", F.col("item_code").cast(StringType()))
        .withColumn("color_code", F.col("color_code").cast(StringType()))
        .withColumn("style_code", F.col("style_code").cast(StringType()))
        .dropDuplicates()
    )

    article_df = article_df.withColumn(
        "class_desc",
        F.when(
            F.col("class_desc").contains("DO NOT USE"),
            F.trim(F.regexp_replace("class_desc", r"\s*DO NOT USE\s*", "")),
        ).otherwise(F.col("class_desc")),
    )

    article_x_ref = article_df.join(sku_item_xref_df, on=["item_code", "style_code", "color_code"], how="left")
    article_x_ref = article_x_ref.withColumn(
        "sku_name",
        F.initcap(
            F.concat_ws(
                " ",
                F.col("class_desc"),
                F.col("color_desc"),
                F.col("size_code"),
                F.lit(" - "),
                F.col("sku_code"),
            )
        ),
    )

    sales_join_df = unit_cost_df.join(sku_item_xref_df, on=["sku_code"], how="left")
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
            "color_desc",
        ),
        on=["sku_code", "item_code", "style_code", "color_code", "size_code"],
        how="left",
    )

    df_fixed = sales_article_joined_df.dropDuplicates(["sku_code"])
    sku_lifecycle = build_sku_lifecycle(sales_actuals_df=sales_actuals_df)
    df_sku_lifecycle = df_fixed.join(sku_lifecycle, on="sku_code", how="left")

    # Deterministic markup factor based on sku_id hash (avoids non-repeatable F.rand())
    sku_id_col = F.col("sku_code").cast("string")
    hash01 = (F.pmod(F.abs(F.xxhash64(sku_id_col)), F.lit(1000)) / F.lit(1000.0))
    markup_factor = F.lit(MARKUP_MIN) + (F.lit(MARKUP_MAX - MARKUP_MIN) * hash01)

    final_df = df_sku_lifecycle.withColumnsRenamed({"sku_code": "sku_id", "color_code": "cc_code", "color_desc": "color_name"})
    final_df = (
        final_df.withColumn("cluster_desc_clean", F.split(F.col("cluster_desc"), ":").getItem(1))
        .withColumn("subcategory", F.concat_ws("_", F.col("cluster_code"), F.col("cluster_desc_clean")))
        .withColumn("category", F.concat(F.col("department_code"), F.lit(":"), F.col("department_desc")))
        .withColumn("brand", F.concat(F.col("brand_code"), F.lit(":"), F.col("brand_desc")))
        .withColumn("base_selling_price", F.round(F.col("unit_cost") * markup_factor, 2))
        .drop("cluster_desc_clean")
    )

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
        F.col("item_desc").alias("item_name"),
    ).dropDuplicates(["sku_id"])

    # Make sku_name explicitly include sku_id to reduce collisions
    dim_sku_df = dim_sku_df.withColumn("sku_name", F.concat(F.col("sku_name"), F.lit(" - "), F.col("sku_id")))
    return dim_sku_df


def main():
    sales_actuals_df = spark.table(INPUT_TABLE_SALES_ACTUALS)
    sku_item_xref_df = spark.table(INPUT_TABLE_SKU_ITEM_XREF)
    article_df = spark.table(INPUT_TABLE_ARTICLE)

    dim_sku_df = build_dim_sku(
        sales_actuals_df=sales_actuals_df,
        sku_item_xref_df=sku_item_xref_df,
        article_df=article_df,
    )

    writer = dim_sku_df.write.format(WRITE_FORMAT).mode(WRITE_MODE)
    if OVERWRITE_SCHEMA:
        writer = writer.option("overwriteSchema", "true")
    writer.saveAsTable(OUTPUT_TABLE_DIM_SKU)


main()

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