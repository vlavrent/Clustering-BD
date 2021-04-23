import numpy as np
from pyspark.sql import functions as F


def cosine_distance(x, y):
	x_norm = np.linalg.norm(x, axis=1, keepdims=True)
	y_norm = np.linalg.norm(y, axis=1, keepdims=True)

	similarity = np.dot(x, y.T) / (x_norm * y_norm)
	dist = 1. - similarity
	return dist


def euclidean_distance(x, y):
	return np.linalg.norm(x - y)


def calculate_squared_distance(cluster_centers):
	def f(features, prediction):

		return float(np.sqrt(euclidean_distance(features, cluster_centers[prediction])))

	return F.udf(f)


def calculate_sse(predictions, cluster_centers):
	return predictions.select(
		calculate_squared_distance(cluster_centers)(F.col('features'), F.col('prediction'))
			.alias('squared_distance')) \
		.agg(F.sum('squared_distance').alias('sse')).collect()[0]['sse']
