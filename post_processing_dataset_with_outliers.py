import argparse

import numpy as np
from pyspark.sql import functions as F, SparkSession
from pyspark.sql import types as spark_types
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from scipy.spatial.distance import euclidean


def post_processing(dataset_path):
	spark = SparkSession.builder.master("local[*]"). \
		appName("post_processing"). \
		getOrCreate()

	dataset = spark.read.json(dataset_path)\
		.withColumn("new", F.arrays_zip("features_list", "outlier_list", "prediction_list")) \
		.withColumn("new", F.explode("new"))\
		.select(F.col("new.features_list").alias("features"),
			F.col("new.outlier_list").alias("outlier"),
			F.col("new.prediction_list").alias("prediction"))

	print(dataset.count())

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--dataset_path",
		"-d",
		help="path of the dataset",
		default="Datasets/Data1_with_outliers"
	)
	args = parser.parse_args()
	post_processing(args.dataset_path)
