from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F, SparkSession
from pyspark.sql import types as spark_types
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from matplotlib import pyplot as plt
import argparse


def simulate_kmeans(dataset_path, startk=2, endk=6):
	spark = SparkSession.builder.master("local[*]"). \
		appName("kmeans"). \
		getOrCreate()

	dataset = spark.read.csv(dataset_path, header=True) \
		.select(F.col("0").cast(spark_types.FloatType()),
				F.col("1").cast(spark_types.FloatType()))

	assembler = VectorAssembler(
		inputCols=["0", "1"],
		outputCol="features")

	dataset = assembler.transform(dataset)

	euclidean_silhouette_scores = {}
	for k in range(startk, endk + 1):
		kmeans = KMeans().setK(k).setSeed(13) \
			.setFeaturesCol("features") \
			.setPredictionCol("prediction") \
			.setDistanceMeasure('euclidean')
		model = kmeans.fit(dataset)

		# Make predictions
		predictions = model.transform(dataset)

		# Evaluate clustering by computing Silhouette score
		evaluator = ClusteringEvaluator().setDistanceMeasure('squaredEuclidean')

		silhouette = evaluator.evaluate(predictions)
		euclidean_silhouette_scores[k] = silhouette

	print('euclidean_silhouette_scores: ', euclidean_silhouette_scores)

	cosine_silhouette_scores = {}
	for k in range(startk, endk + 1):
		kmeans = KMeans().setK(k).setSeed(13) \
			.setFeaturesCol("features") \
			.setPredictionCol("prediction") \
			.setDistanceMeasure('cosine')
		model = kmeans.fit(dataset)

		# Make predictions
		predictions = model.transform(dataset)

		# Evaluate clustering by computing Silhouette score
		evaluator = ClusteringEvaluator().setDistanceMeasure('cosine')

		silhouette = evaluator.evaluate(predictions)
		cosine_silhouette_scores[k] = silhouette

	print('cosine_silhouette_scores: ', cosine_silhouette_scores)

	plt.clf()
	plt.plot(euclidean_silhouette_scores.keys(), euclidean_silhouette_scores.values())
	plt.title("euclidean_silhouette_scores")
	plt.legend(['silhouette'], loc='best')
	plt.xlabel("Number of clusters")
	plt.ylabel("Silhouette score")
	plt.savefig("euclidean_silhouette_scores.png")

	plt.clf()
	plt.plot(cosine_silhouette_scores.keys(), cosine_silhouette_scores.values())
	plt.title("cosine_silhouette_scores")
	plt.legend(['silhouette'], loc='best')
	plt.xlabel("Number of clusters")
	plt.ylabel("Silhouette score")
	plt.savefig("cosine_silhouette_scores.png")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--dataset_path",
		"-d",
		help="path of the dataset",
		default="Datasets/Data1.csv"
	)
	parser.add_argument(
		"--startk",
		"-s",
		help="start value of k for fine tuning",
		default=2
	)
	parser.add_argument(
		"--endk",
		"-e",
		help="end value of k for fine tuning",
		default=6
	)
	args = parser.parse_args()
	simulate_kmeans(dataset_path=args.dataset_path, startk=int(args.startk), endk=int(args.endk))
