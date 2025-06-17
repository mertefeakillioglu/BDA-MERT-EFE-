
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, unix_timestamp, when
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import MulticlassMetrics

spark = SparkSession.builder.appName("AdvancedRFModelFixed").getOrCreate()

# 1. Veri Yükleme
df = spark.read.csv("gs://us-accidents-bucket-g6/US_Accidents_March23.csv", header=True, inferSchema=True)

# 2. Null Temizliği
df = df.dropna(subset=["Start_Time", "End_Time", "Severity", "Temperature(F)", "Humidity(%)", 
                       "Visibility(mi)", "Wind_Speed(mph)", "Weather_Condition"])

# 3. Özellik Oluşturma
df = df.withColumn("Duration", (unix_timestamp("End_Time") - unix_timestamp("Start_Time")) / 60)
df = df.withColumn("duration_category", when(col("Duration") > 60, 1).otherwise(0))
df = df.withColumn("Start_Hour", hour("Start_Time"))
df = df.withColumn("Day_Of_Week", dayofweek("Start_Time"))
df = df.withColumn("Is_Weekend", when(col("Day_Of_Week").isin([1,7]), 1).otherwise(0))

# 4. Kategorik Sütun Dönüştürme
indexer = StringIndexer(inputCol="Weather_Condition", outputCol="Weather_Condition_Indexed")
df = indexer.fit(df).transform(df)

# 5. Feature Vector Oluşturma
feature_cols = ["Severity", "Start_Hour", "Day_Of_Week", "Is_Weekend", 
                "Temperature(F)", "Humidity(%)", "Visibility(mi)", 
                "Wind_Speed(mph)", "Weather_Condition_Indexed"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# 6. Eğitim / Test Ayrımı
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# 7. Random Forest Modeli (maxBins fixli)
rf = RandomForestClassifier(
    featuresCol="features", 
    labelCol="duration_category", 
    seed=42,
    maxBins=200
)

# 8. Parametre Tuning (Grid Search)
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [20, 50]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()

evaluator = BinaryClassificationEvaluator(labelCol="duration_category")

crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

cv_model = crossval.fit(train_data)

# 9. Tahmin ve Değerlendirme
predictions = cv_model.transform(test_data)

accuracy = predictions.filter(col("duration_category") == col("prediction")).count() / float(predictions.count())
auc = evaluator.evaluate(predictions)

predictionAndLabels = predictions.select("prediction", "duration_category").rdd.map(lambda x: (float(x[0]), float(x[1])))
metrics = MulticlassMetrics(predictionAndLabels)

precision = metrics.precision(1.0)
recall = metrics.recall(1.0)
f1_score = metrics.fMeasure(1.0)

print("Model Evaluation Results (Random Forest with maxBins=200):")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# 10. Feature Importance
importances = cv_model.bestModel.featureImportances.toArray()
input_features = assembler.getInputCols()

print("\nFeature Importances:")
for col_name, importance in zip(input_features, importances):
    print(f"{col_name}: {importance:.4f}")

spark.stop()
