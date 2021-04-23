from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F, SparkSession
from pyspark.sql import types as spark_types
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from matplotlib import pyplot as plt
import argparse
from utils import calculate_sse


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
	euclidean_sse_scores = {}
	for k in range(startk, endk + 1):
		kmeans = KMeans().setK(k).setSeed(13) \
			.setFeaturesCol("features") \
			.setPredictionCol("prediction") \
			.setDistanceMeasure('euclidean')
		model = kmeans.fit(dataset)

		# Make predictions
		predictions = model.transform(dataset)
		predictions.persist()

		# Evaluate clustering by computing Silhouette score
		evaluator = ClusteringEvaluator().setDistanceMeasure('squaredEuclidean')
		silhouette = evaluator.evaluate(predictions)
		euclidean_silhouette_scores[k] = silhouette

		# Evaluate clustering by computing SSE
		cluster_centers = model.clusterCenters()
		spark.sparkContext.broadcast(cluster_centers)
		sse = calculate_sse(predictions, cluster_centers)
		predictions.unpersist()
		euclidean_sse_scores[k] = sse

	print('euclidean_silhouette_scores: ', euclidean_silhouette_scores)
	print('euclidean_sse_scores: ', euclidean_sse_scores)

	cosine_silhouette_scores = {}
	cosine_sse_scores = {}
	for k in range(startk, endk + 1):
		kmeans = KMeans().setK(k).setSeed(13) \
			.setFeaturesCol("features") \
			.setPredictionCol("prediction") \
			.setDistanceMeasure('cosine')
		model = kmeans.fit(dataset)

		# Make predictions
		predictions = model.transform(dataset)
		predictions.persist()

		# Evaluate clustering by computing Silhouette score
		evaluator = ClusteringEvaluator().setDistanceMeasure('squaredEuclidean')
		silhouette = evaluator.evaluate(predictions)
		cosine_silhouette_scores[k] = silhouette

		# Evaluate clustering by computing SSE
		cluster_centers = model.clusterCenters()
		spark.sparkContext.broadcast(cluster_centers)
		sse = calculate_sse(predictions, cluster_centers)
		predictions.unpersist()
		cosine_sse_scores[k] = sse

	print('cosine_silhouette_scores: ', cosine_silhouette_scores)
	print('cosine_sse_scores: ', cosine_sse_scores)

	plt.clf()
	plt.plot(euclidean_silhouette_scores.keys(), euclidean_silhouette_scores.values())
	plt.plot(cosine_silhouette_scores.keys(), cosine_silhouette_scores.values())
	plt.title("Optimal number of clusters based on Silhouette")
	plt.legend(['silhouette_euclidean', 'silhouette_cosine'], loc='best')
	plt.xlabel("Number of clusters")
	plt.ylabel("Silhouette score")
	plt.savefig("kmeans_silhouette_scores.png")


	plt.clf()
	plt.plot(euclidean_sse_scores.keys(), euclidean_sse_scores.values())
	plt.plot(cosine_sse_scores.keys(), cosine_sse_scores.values())
	plt.title("Optimal number of clusters based on SSE")
	plt.legend(['sse_euclidean', 'sse_cosine'], loc='best')
	plt.xlabel("Number of clusters")
	plt.ylabel("SSE")
	plt.savefig("kmeans_SSE.png")


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
