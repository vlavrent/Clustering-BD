from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F, SparkSession
from pyspark.sql import types as spark_types
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from matplotlib import pyplot as plt
import argparse
from utils import calculate_sse, visualize_predictions
import time
import os
import pandas as pd
from tqdm import tqdm


def simulate_kmeans(dataset_path, startk=2, endk=6, algorithm='kmeans'):
	spark = SparkSession.builder.master("local[*]"). \
		appName("kmeans"). \
		getOrCreate()

	saving_path = dataset_path.split('/')[1].split('.')[0] + '_clustering_experiments_results/'

	if not os.path.exists(saving_path):
		os.mkdir(saving_path)

	dataset = spark.read.csv(dataset_path, header=True) \
		.select(F.col("0").cast(spark_types.FloatType()),
				F.col("1").cast(spark_types.FloatType()))

	assembler = VectorAssembler(
		inputCols=["0", "1"],
		outputCol="features")

	dataset = assembler.transform(dataset)

	euclidean_silhouette_scores = {}
	euclidean_sse_scores = {}
	euclidean_times = {}
	for k in tqdm(range(startk, endk + 1)):
		start = time.time()

		if algorithm == 'kmeans':
			algo = KMeans(maxIter=40)
		elif algorithm == 'bkmeans':
			algo = BisectingKMeans(maxIter=40)

		algo.setK(k).setSeed(4) \
			.setFeaturesCol("features") \
			.setPredictionCol("prediction") \
			.setDistanceMeasure('euclidean')
		model = algo.fit(dataset)

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
		# sse = model.summary.trainingCost
		sse = calculate_sse(predictions, cluster_centers)
		euclidean_sse_scores[k] = sse
		end = time.time()
		euclidean_times[k] = end - start

		visualize_predictions(predictions, saving_path, model_name=str(k) + "_e_" + algorithm)
		predictions.unpersist()
	print('euclidean_silhouette_scores: ', euclidean_silhouette_scores)
	print('euclidean_sse_scores: ', euclidean_sse_scores)
	print('euclidean_times: ', euclidean_times)

	cosine_silhouette_scores = {}
	cosine_sse_scores = {}
	cosine_times = {}
	for k in tqdm(range(startk, endk + 1)):
		start = time.time()

		if algorithm == 'kmeans':
			algo = KMeans(maxIter=40)
		elif algorithm == 'bkmeans':
			algo = BisectingKMeans(maxIter=40)

		algo.setK(k).setSeed(4) \
			.setFeaturesCol("features") \
			.setPredictionCol("prediction") \
			.setDistanceMeasure('cosine')
		model = algo.fit(dataset)

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
		# sse = model.summary.trainingCost
		sse = calculate_sse(predictions, cluster_centers)
		cosine_sse_scores[k] = sse
		end = time.time()
		cosine_times[k] = end - start

		visualize_predictions(predictions, saving_path, model_name=str(k) + "_c_" + algorithm)
		predictions.unpersist()

	print('cosine_silhouette_scores: ', cosine_silhouette_scores)
	print('cosine_sse_scores: ', cosine_sse_scores)
	print('cosine_times: ', cosine_times)

	results_dir = {'euclidean_silhouette_scores': euclidean_silhouette_scores,
				   'euclidean_sse_scores': euclidean_sse_scores,
				   'euclidean_times': euclidean_times,
				   'cosine_silhouette_scores': cosine_silhouette_scores,
				   'cosine_sse_scores': cosine_sse_scores,
				   'cosine_times': cosine_times}

	pd.DataFrame.from_dict(results_dir).to_csv(saving_path + algorithm + '_results.csv')

	plt.clf()
	plt.plot(list(euclidean_silhouette_scores.keys()), list(euclidean_silhouette_scores.values()))
	plt.plot(list(cosine_silhouette_scores.keys()), list(cosine_silhouette_scores.values()))
	plt.title("Optimal number of clusters based on Silhouette")
	plt.legend(['silhouette_euclidean', 'silhouette_cosine'], loc='best')
	plt.xlabel("Number of clusters")
	plt.ylabel("Silhouette score")
	plt.savefig(saving_path + algorithm + "_silhouette_scores.png")

	plt.clf()
	plt.plot(list(euclidean_sse_scores.keys()), list(euclidean_sse_scores.values()))
	plt.plot(list(cosine_sse_scores.keys()), list(cosine_sse_scores.values()))
	plt.title("Optimal number of clusters based on SSE")
	plt.legend(['sse_euclidean', 'sse_cosine'], loc='best')
	plt.xlabel("Number of clusters")
	plt.ylabel("SSE")
	plt.savefig(saving_path + algorithm + "_SSE.png")

	plt.clf()
	plt.plot(list(euclidean_times.keys()), list(euclidean_times.values()))
	plt.plot(list(cosine_times.keys()), list(cosine_times.values()))
	plt.title("Time fluctuation")
	plt.legend(['euclidean_times', 'cosine_times'], loc='best')
	plt.xlabel("Number of clusters")
	plt.ylabel("Time(s)")
	plt.savefig(saving_path + algorithm + "_times.png")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--dataset_path",
		"-d",
		help="path of the dataset",
		default="Datasets/Data1.csv"
	)
	parser.add_argument(
		"--algorithm",
		"-a",
		help="name of clustering algorithm {kmeans|bkmeans",
		default="kmeans"
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
	simulate_kmeans(
		dataset_path=args.dataset_path,
		startk=int(args.startk),
		endk=int(args.endk),
		algorithm=args.algorithm)
