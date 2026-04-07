import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans

def train_and_save_model():
    print(">>> Initializing Spark for AI Training (Unit V)...")
    spark = SparkSession.builder \
        .appName("DevOps_AI_Train") \
        .getOrCreate()
    
    # Path to the shared cleaned crime dataset
    data_path = "../crime---analysis/data/cleaned_crime_data.csv"
    
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        return

    df = spark.read.csv(data_path, header=True, inferSchema=True)
    df = df.withColumn("crime_count", col("crime_count").cast("integer")).dropna()

    # Get Top 5 Crime Types for Feature Engineering
    top_crimes = df.groupBy("crime_type").sum("crime_count").orderBy(col("sum(crime_count)").desc()).limit(5)
    top_5 = [row['crime_type'] for row in top_crimes.collect()]
    
    # Pivot for State-wise feature vectors
    features_df = df.filter(col("crime_type").isin(top_5)) \
                    .groupBy("STATE/UT") \
                    .pivot("crime_type") \
                    .sum("crime_count") \
                    .na.fill(0)
    
    # Feature Vectorization
    assembler = VectorAssembler(inputCols=top_5, outputCol="features")
    assembled_df = assembler.transform(features_df)

    # Standard Scaling
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
    scalerModel = scaler.fit(assembled_df)
    scaled_df = scalerModel.transform(assembled_df)

    # K-Means Training (k=3)
    kmeans = KMeans(featuresCol="scaledFeatures", predictionCol="cluster", k=3, seed=1)
    model = kmeans.fit(scaled_df)

    # Save the Model for Deployment (The DevOps Way)
    model_path = "crime_kmeans_model"
    scaler_path = "crime_scaler_model"
    
    # Using Spark's built-in overwrite capability
    model.write().overwrite().save(model_path)
    scalerModel.write().overwrite().save(scaler_path)
    
    # Save the list of top 5 crimes for the API to use
    with open("top_crimes.txt", "w") as f:
        f.write(",".join(top_5))

    print(f"\n>>> SUCCESS: AI Model and Scaler saved for deployment in {model_path}!")
    
    spark.stop()

if __name__ == "__main__":
    train_and_save_model()
