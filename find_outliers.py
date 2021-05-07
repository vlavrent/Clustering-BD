import argparse

import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F, SparkSession
from pyspark.sql import types as spark_types
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean


def cellUDF(minValue, res):
	def f(x):
		return int((x - minValue) / res)
	return F.udf(f, spark_types.IntegerType())


@F.udf(returnType=spark_types.ArrayType(spark_types.IntegerType()))
def compute_outliers(features_list):

	if len(features_list) < 50:
		return [1] * len(features_list)

	neigh = NearestNeighbors(n_neighbors=50)
	neigh.fit(features_list)

	neighbourhoods = []
	neighbours_list = neigh.kneighbors(features_list, n_neighbors=50, return_distance=False)
	for index, neighbours in enumerate(neighbours_list):
		neighbours_features = []
		for neighbour in neighbours:
			neighbours_features.append(features_list[neighbour])
		neighbourhoods.append(neighbours_features)

	predictions = []
	for index, neighbours_features in enumerate(neighbourhoods):
		mean = np.mean(neighbours_features, axis=0)
		std = np.std(neighbours_features, axis=0)
		if euclidean(features_list[index], mean) > euclidean(2 * std, mean):
			predictions.append(1)
		else:
			predictions.append(0)

	return predictions


def find_outliers(dataset_path, saving_path):
	spark = SparkSession.builder.master("local[*]"). \
		appName("find_outliers"). \
		getOrCreate()

	dataset_with_outliers = spark.read.csv(dataset_path, header=True)\
		.select(F.monotonically_increasing_id().alias("id"),
		F.col("0").cast(spark_types.FloatType()),
			F.col("1").cast(spark_types.FloatType()),
				F.col("outlier"))

	assembler = VectorAssembler(
		inputCols=["0", "1"],
		outputCol="features")

	dataset_with_outliers = assembler.transform(dataset_with_outliers)

	resolution = 0.2
	mins = dataset_with_outliers.select(F.min("0"), F.min("1")).head()
	minX = mins["min(0)"]
	minY = mins["min(1)"]

	XCoordsUdf = cellUDF(minX, resolution)
	YCoordsUdf = cellUDF(minY, resolution)
	relData = dataset_with_outliers.withColumn("cellx", XCoordsUdf("0")).withColumn("celly", YCoordsUdf("1"))

	grid_dataset = relData.groupby("cellx", "celly").agg(F.collect_list("features").alias("features_list"),
														 F.collect_list("outlier").alias("outlier_list"))
	grid_dataset_with_predictions = grid_dataset.withColumn("prediction_list", compute_outliers(F.col("features_list")))
	grid_dataset_with_predictions.write.parquet(saving_path, mode="overwrite")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--dataset_path",
		"-d",
		help="path of the dataset",
		default="Datasets/Data1_with_outliers"
	)
	parser.add_argument(
		"--saving_path",
		"-s",
		help="path to save dataset without outliers",
		default="Datasets/Data1_without_outliers"
	)
	args = parser.parse_args()
	find_outliers(args.dataset_path, args.saving_path)