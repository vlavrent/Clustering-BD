from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F, SparkSession
from pyspark.sql import types as spark_types
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from matplotlib import pyplot as plt


def simulate_kmeans(startk=2, endk=6):
	spark = SparkSession.builder.master("local[*]"). \
		appName("kmeans"). \
		getOrCreate()

	dataset = spark.read.csv("Datasets/Data1.csv", header=True) \
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
	simulate_kmeans()
