from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, lower, trim
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create a Spark Session
spark = SparkSession.builder \
            .appName("Mini Spark Project") \
            .getOrCreate()

# Read the dataset - both test.csv and train.csv
test_df = spark.read.csv("data/test.csv", header=True, inferSchema=True)
train_df = spark.read.csv("data/train.csv", header=True, inferSchema=True)

# Preprocessing - Remove specific patterns and convert all text to lowercase
def preprocess(df, is_train=True):
    if is_train:
        return df.select("tweet", "label") \
                .withColumn("tweet", regexp_replace(lower(trim(df.tweet)), "[^a-zA-Z\\s]", " ")) \
                .filter("tweet != ''")
    else:
        return df.select("tweet") \
                .withColumn("tweet", regexp_replace(lower(trim(df.tweet)), "[^a-zA-Z\\s]", " ")) \
                .filter("tweet != ''")
    
train_df = preprocess(train_df, is_train=True)
test_df = preprocess(test_df, is_train=False)

train_df.cache()
test_df.cache()

def extract_features(df):
    tokenizer = Tokenizer(inputCol="tweet", outputCol="words")
    words_df = tokenizer.transform(df)

    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    filtered_df = remover.transform(words_df)

    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures")
    featurized_df = hashingTF.transform(filtered_df)

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurized_df)
    rescaledData = idfModel.transform(featurized_df)

    if "label" in df.columns:
        # Select the required columns explicitly
        output_df = rescaledData.select("features", df["label"].alias("label"))
    else:
        output_df = rescaledData.select("features")

    return output_df

train_data = extract_features(train_df)
test_data = extract_features(test_df)

# Model Training 
lr = LogisticRegression(labelCol="label", featuresCol="features")
lrmodel = lr.fit(train_data)

predictions = lrmodel.transform(test_data)
predicted_label = predictions.select("prediction").collect()

predictions = lrmodel.transform(test_data)

