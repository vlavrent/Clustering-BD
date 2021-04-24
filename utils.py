import numpy as np
from pyspark.sql import functions as F
from scipy.spatial.distance import cosine, euclidean


def calculate_squared_distance(cluster_centers):
	def f(features, prediction):

		return float(np.sqrt(euclidean(features, cluster_centers[prediction])))

	return F.udf(f)


def calculate_sse(predictions, cluster_centers):
	return predictions.select(
		calculate_squared_distance(cluster_centers)(F.col('features'), F.col('prediction'))
			.alias('squared_distance')) \
		.agg(F.sum('squared_distance').alias('sse')).collect()[0]['sse']
