

import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from elasticsearch import Elasticsearch, helpers
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col
from functools import reduce
import uuid

spark = SparkSession.builder.config("spark.sql.caseSensitive", "true").getOrCreate()

raw_books = spark.read.json('/user/amm9801_nyu_edu/project/Books.json')
raw_clothing = spark.read.json('/user/sg6482_nyu_edu/project/Clothing_Shoes_and_Jewelry.json')
raw_electronics = spark.read.json('/user/sa6142_nyu_edu/project/electronics/Electronics.json')

# raw_metadata = spark.read.json('project/meta_Books.json')
# metadata = raw_metadata.select('asin', 'title', 'description', 'brand')

raw_books.createOrReplaceTempView("view_raw_books")
raw_clothing.createOrReplaceTempView("view_raw_clothing")
raw_electronics.createOrReplaceTempView("view_raw_electronics")

reviews_filtered = spark.sql("select asin, reviewerId, overall from\
                            (select *, count(*) over (partition by reviewerId) as c\
                            from view_raw_books) where c >= 5")
clothing_filtered = spark.sql("select asin, reviewerId, overall from\
                            (select *, count(*) over (partition by reviewerId) as c\
                            from view_raw_clothing) where c >= 5")
electronics_filtered = spark.sql("select asin, reviewerId, overall from\
                            (select *, count(*) over (partition by reviewerId) as c\
                            from view_raw_electronics) where c >= 5")

# reviews_filtered.write.option("header",True).csv("/user/amm9801_nyu_edu/project/reviews_filtered")
# clothing_filtered.write.option("header",True).csv("/user/sg6482_nyu_edu/project/clothing_filtered")
# electronics_filtered.write.option("header",True).csv("/user/sa6142_nyu_edu/project/electronics_filtered")
# reviews_filtered = spark.read.option("header", True).csv("/user/amm9801_nyu_edu/project/reviews_filtered")
# clothing_filtered = spark.read.option("header", True).csv("/user/sg6482_nyu_edu/project/clothing_filtered")
# electronics_filtered = spark.read.option("header", True).csv("/user/sa6142_nyu_edu/project/electronics_filtered")


def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)
merged_reviews = unionAll(reviews_filtered, clothing_filtered, electronics_filtered)


stringindexer = StringIndexer().setInputCol("reviewerId").setOutputCol("reviewerIdIdx")
model_reviewerId = stringindexer.fit(merged_reviews)
merged_reviews_transformed = model_reviewerId.transform(merged_reviews)
stringindexer = StringIndexer().setInputCol("asin").setOutputCol("asinIdx")
model_asin = stringindexer.fit(merged_reviews_transformed)
merged_reviews_transformed = model_asin.transform(merged_reviews_transformed)


merged_reviews_filtered_transformed = merged_reviews_transformed\
                                .withColumn("reviewerIdIdx", col("reviewerIdIdx").cast('int'))\
                                .withColumn("asinIdx", col("asinIdx").cast('int'))\
                                .withColumn("overall", col("overall").cast('float'))


(training, validation) = merged_reviews_filtered_transformed.randomSplit([0.8, 0.2])

als = ALS(maxIter=10, regParam=0.05, rank=48, userCol="reviewerIdIdx", itemCol="asinIdx", ratingCol="overall",
          coldStartStrategy="drop")
model = als.fit(training)
model.save("/user/sg6482_nyu_edu/project/als_model_merged")
