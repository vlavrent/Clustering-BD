import argparse

from pyspark.ml.feature import VectorAssembler, BucketedRandomProjectionLSH
from pyspark.sql import functions as F, SparkSession
from pyspark.sql import types as spark_types
from knn import compute_neighbors


def find_outliers(dataset_path):
	spark = SparkSession.builder.master("local[*]"). \
		appName("find_outliers"). \
		getOrCreate()

	dataset_with_outliers = spark.read.csv(dataset_path, header=True)\
		.select(F.monotonically_increasing_id().alias("id"),
		F.col("0").cast(spark_types.FloatType()),
			F.col("1").cast(spark_types.FloatType()),
				F.col("outlier"))
	# print(dataset_with_outliers.show())

	assembler = VectorAssembler(
		inputCols=["0", "1"],
		outputCol="features")

	dataset_with_outliers = assembler.transform(dataset_with_outliers)

	dataset_with_outliers_rdd = dataset_with_outliers.select("id", "features").rdd
	results = compute_neighbors(dataset_with_outliers_rdd, dataset_with_outliers_rdd)
	spark.createDataFrame(results).show()
	# brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=10.0,
	# 								  numHashTables=5, seed=13)
	# model = brp.fit(dataset_with_outliers)
	#
	# transformed_dataset_with_outliers = model.transform(dataset_with_outliers)
	#
	# joined = model.approxSimilarityJoin(transformed_dataset_with_outliers, transformed_dataset_with_outliers, 0.5,
	# 									distCol="EuclideanDistance") \
	# 	.select(F.col("datasetA.features").alias("idA"),
	# 			F.col("datasetB.features").alias("idB"),
	# 			F.col("EuclideanDistance"))
	#
	#
	# joined.show(100, truncate=False)
	# print(joined.count())
	# print(joined.groupBy("idA").count().count())


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
		help="path to save dataset with outliers",
		default="Datasets/Data1_with_outliers"
	)
	args = parser.parse_args()
	find_outliers(args.dataset_path)