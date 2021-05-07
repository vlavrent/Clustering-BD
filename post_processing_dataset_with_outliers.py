import argparse

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F, SparkSession
from pyspark.sql import types as spark_types
from pyspark.ml.functions import vector_to_array


def post_processing(dataset_path, saving_path):
	spark = SparkSession.builder.master("local[*]"). \
		appName("post_processing"). \
		getOrCreate()

	dataset = spark.read.parquet(dataset_path)\
		.withColumn("new", F.arrays_zip("features_list", "outlier_list", "prediction_list")) \
		.withColumn("new", F.explode("new"))\
		.select(F.col("new.features_list").alias("features"),
			F.col("new.outlier_list").alias("outlier").cast(spark_types.DoubleType()),
			F.col("new.prediction_list").alias("prediction").cast(spark_types.DoubleType()))

	evaluator = BinaryClassificationEvaluator()
	evaluator.setLabelCol("outlier").setRawPredictionCol("prediction")
	print("AUC ROC:", evaluator.evaluate(dataset))

	dataset.select(vector_to_array(F.col("features")).getItem(0).alias("0"),
				   vector_to_array(F.col("features")).getItem(1).alias("1"))\
		.write.csv(saving_path, header=True, mode="overwrite")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--dataset_path",
		"-d",
		help="path of the dataset",
		default="Datasets/Data1_without_outliers"
	)
	parser.add_argument(
		"--saving_path",
		"-s",
		help="path to save final dataset",
		default="Datasets/final_Data1_without_outliers"
	)
	args = parser.parse_args()
	post_processing(args.dataset_path, args.saving_path)
