import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F, SparkSession
from pyspark.sql import types as spark_types
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from matplotlib import pyplot as plt
import argparse
from utils import calculate_sse, visualize_predictions
import time
import os
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

def assign_points(transform_dict):
	def f(prediction):
		return int(transform_dict[prediction])
	return F.udf(f, spark_types.IntegerType())


# Example: spark-submit --master local[*] --driver-memory 10g  kmeans_hierarchical.py -d Datasets/Data1.csv -s 2 -e 14
def simulate_kmeans_hierarchical(dataset_path, startk=2, endk=6):
	spark = SparkSession.builder. \
		appName("clustering"). \
		getOrCreate()

	spark.sparkContext.setLogLevel('WARN')
	outer_saving_path = dataset_path.split('/')[1].split('.')[0] + '_clustering_experiments_results/'
	if not os.path.exists(outer_saving_path):
		os.mkdir(outer_saving_path)

	saving_path = outer_saving_path + "kmeans_hierarchical" + "/"

	if not os.path.exists(saving_path):
		os.mkdir(saving_path)

	start = time.time()

	dataset = spark.read.csv(dataset_path, header=True) \
		.select(F.col("0").cast(spark_types.FloatType()),
				F.col("1").cast(spark_types.FloatType()))

	assembler = VectorAssembler(
		inputCols=["0", "1"],
		outputCol="features")

	dataset = assembler.transform(dataset)

	algo = KMeans(maxIter=20)

	algo.setK(20).setSeed(4) \
		.setFeaturesCol("features") \
		.setPredictionCol("prediction") \
		.setDistanceMeasure('euclidean')
	model = algo.fit(dataset)

	# Make predictions
	predictions = model.transform(dataset)
	predictions.persist()

	# Evaluate clustering by computing SSE
	cluster_centers = model.clusterCenters()
	spark.sparkContext.broadcast(cluster_centers)

	kmeans_time = time.time() - start
	euclidean_silhouette_scores = {}
	euclidean_sse_scores = {}
	euclidean_times = {}


	for k in tqdm(range(startk, endk + 1)):
		start = time.time()
		clustering = AgglomerativeClustering(n_clusters=k).fit(cluster_centers)
		labels = clustering.labels_
		transform_dict = {index: label for index, label in enumerate(labels)}
		centroids_per_cluster = defaultdict(list)

		for index, label in enumerate(labels):
			centroids_per_cluster[label].append(cluster_centers[index])

		centroids_per_cluster = {key: np.mean(centroids, axis=0) for key, centroids in centroids_per_cluster.items()}
		centroids_per_cluster = dict(sorted(centroids_per_cluster.items()))

		centroids_per_cluster = [centroid for centroid in centroids_per_cluster.values()]

		agglomerative_predictions = predictions.withColumn(
			"final_prediction", assign_points(transform_dict)(F.col("prediction")))

		agglomerative_predictions.persist()
		# Evaluate clustering by computing Silhouette score
		evaluator = ClusteringEvaluator(predictionCol="final_prediction").setDistanceMeasure('squaredEuclidean')
		silhouette = evaluator.evaluate(agglomerative_predictions)
		euclidean_silhouette_scores[k] = silhouette

		# sse = model.summary.trainingCost
		sse = calculate_sse(agglomerative_predictions, centroids_per_cluster, prediction_column="final_prediction")
		euclidean_sse_scores[k] = sse
		end = time.time()
		euclidean_times[k] = end - start + kmeans_time

		visualizations_saving_path = saving_path + "euclidean/"
		if not os.path.exists(visualizations_saving_path):
			os.mkdir(visualizations_saving_path)

		visualize_predictions(
			agglomerative_predictions,
			visualizations_saving_path,
			model_name=str(k) + "_e_" + "kmeans_hierarchical",
			prediction_column="final_prediction")
		agglomerative_predictions.unpersist()
	print('euclidean_silhouette_scores: ', euclidean_silhouette_scores)
	print('euclidean_sse_scores: ', euclidean_sse_scores)
	print('euclidean_times: ', euclidean_times)

	predictions.unpersist()
	results_dir = {
		'euclidean_silhouette_scores': euclidean_silhouette_scores,
		'euclidean_sse_scores': euclidean_sse_scores,
		'euclidean_times': euclidean_times}

	pd.DataFrame.from_dict(results_dir).to_csv(saving_path + "kmeans_hierarchical" + '_results.csv')

	plt.clf()
	plt.plot(list(euclidean_silhouette_scores.keys()), list(euclidean_silhouette_scores.values()))
	plt.title("Optimal number of clusters based on Silhouette")
	plt.legend(['silhouette_euclidean'], loc='best')
	plt.xlabel("Number of clusters")
	plt.ylabel("Silhouette score")
	plt.savefig(saving_path + "kmeans_hierarchical" + "_silhouette_scores.png")

	plt.clf()
	plt.plot(list(euclidean_sse_scores.keys()), list(euclidean_sse_scores.values()))
	plt.title("Optimal number of clusters based on SSE")
	plt.legend(['sse_euclidean'], loc='best')
	plt.xlabel("Number of clusters")
	plt.ylabel("SSE")
	plt.savefig(saving_path + "kmeans_hierarchical" + "_SSE.png")

	plt.clf()
	plt.plot(list(euclidean_times.keys()), list(euclidean_times.values()))
	plt.title("Time fluctuation")
	plt.legend(['euclidean_times'], loc='best')
	plt.xlabel("Number of clusters")
	plt.ylabel("Time(s)")
	plt.savefig(saving_path + "kmeans_hierarchical" + "_times.png")


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
		default=3
	)
	parser.add_argument(
		"--endk",
		"-e",
		help="end value of k for fine tuning",
		default=6
	)
	args = parser.parse_args()
	simulate_kmeans_hierarchical(
		dataset_path=args.dataset_path,
		startk=int(args.startk),
		endk=int(args.endk))
