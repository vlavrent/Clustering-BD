import numpy as np
from pyspark.sql import functions as F
from scipy.spatial.distance import cosine, euclidean
import matplotlib.pyplot as plt


def calculate_squared_distance(cluster_centers):
	def f(features, prediction):

		return float(np.sqrt(euclidean(features, cluster_centers[prediction])))

	return F.udf(f)


def calculate_sse(predictions, cluster_centers):
	return predictions.select(
		calculate_squared_distance(cluster_centers)(F.col('features'), F.col('prediction'))
			.alias('squared_distance')) \
		.agg(F.sum('squared_distance').alias('sse')).collect()[0]['sse']


def visualize_predictions(predictions, saving_path, model_name):
	predictions_list = predictions.select('features', 'prediction').collect()

	x = []
	y = []
	p = []
	for item in predictions_list:
		x.append(item['features'][0])
		y.append(item['features'][1])
		p.append(item['prediction'])

	fig, ax = plt.subplots()

	scatter = ax.scatter(x, y, c=p)

	# produce a legend with the unique colors from the scatter
	legend1 = ax.legend(*scatter.legend_elements(),
						loc="upper right", title="clusters")
	ax.add_artist(legend1)

	fig.savefig(saving_path + model_name + '.png')

