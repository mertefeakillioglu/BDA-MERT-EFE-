
from pyspark.sql import SparkSession
from pyspark.sql.functions import unix_timestamp

spark = SparkSession.builder \
    .appName("USAccidentsHiveAnalysis") \
    .enableHiveSupport() \
    .getOrCreate()

# Load data from GCS
df = spark.read.csv("gs://us-accidents-bucket-g6/US_Accidents_March23.csv", header=True, inferSchema=True)

# Drop nulls
df_clean = df.na.drop()

# Write to Hive table
df_clean.write.mode("overwrite").format("parquet").saveAsTable("us_accidents_data")

# 1. Total record count
spark.sql("SELECT COUNT(*) AS total_records FROM us_accidents_data").show()

# 2. Accidents in March 2023
spark.sql("""
SELECT COUNT(*) AS march_2023_accidents 
FROM us_accidents_data 
WHERE Start_Time LIKE '2023-03%'
""").show()

# 3. Accidents per state
spark.sql("""
SELECT State, COUNT(*) AS total 
FROM us_accidents_data 
GROUP BY State 
ORDER BY total DESC 
LIMIT 10
""").show()

# 4. Average accident duration
spark.sql("""
SELECT AVG(UNIX_TIMESTAMP(End_Time) - UNIX_TIMESTAMP(Start_Time)) / 60 AS avg_duration_minutes 
FROM us_accidents_data
""").show()

# 5. Top 5 cities with most accidents
spark.sql("""
SELECT City, COUNT(*) AS total 
FROM us_accidents_data 
GROUP BY City 
ORDER BY total DESC 
LIMIT 5
""").show()

# 6. Accidents in rainy weather
spark.sql("""
SELECT COUNT(*) AS rain_accidents 
FROM us_accidents_data 
WHERE Weather_Condition LIKE '%Rain%'
""").show()

spark.stop()
