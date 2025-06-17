
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, unix_timestamp, avg

spark = SparkSession.builder.appName("USAccidentsSparkAnalysis").getOrCreate()

# Load data from GCS
df = spark.read.csv("gs://us-accidents-bucket-g6/US_Accidents_March23.csv", header=True, inferSchema=True)

# Drop rows with nulls
df_clean = df.na.drop()

# 1. Total record count
print("1. Total record count:")
print(df_clean.count())

# 2. Accidents in March 2023
print("2. Accidents in March 2023:")
march_accidents = df_clean.filter(col("Start_Time").contains("2023-03")).select("ID").count()
print(march_accidents)

# 3. Accidents per state
print("3. Accident count per state:")
df_clean.groupBy("State").count().orderBy("count", ascending=False).show(10)

# 4. Average accident duration in minutes
print("4. Average accident duration (minutes):")
df_duration = df_clean.withColumn("Duration", 
    (unix_timestamp("End_Time") - unix_timestamp("Start_Time")) / 60)
df_duration.select(avg("Duration")).show()

# 5. Top 5 cities with most accidents
print("5. Top 5 cities with most accidents:")
df_clean.groupBy("City").count().orderBy("count", ascending=False).show(5)

# 6. Accidents in rainy weather
print("6. Accidents in rainy weather:")
rain_accidents = df_clean.filter(col("Weather_Condition").like("%Rain%")).count()
print(rain_accidents)

spark.stop()
