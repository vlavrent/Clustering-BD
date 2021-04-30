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


def create_scatter_plot(x, y, c, save=False, fig_saving_path=None):
	fig, ax = plt.subplots()

	scatter = ax.scatter(x, y, c=c)

	# produce a legend with the unique colors from the scatter
	legend1 = ax.legend(*scatter.legend_elements(),
						loc="upper right", title="noise")
	ax.add_artist(legend1)

	if save:
		fig.savefig(fig_saving_path)
	else:
		plt.show()

	plt.clf()
	plt.close(fig)


def visualize_outliers(dataset, dataset_name):
	data_list = dataset.collect()

	x = []
	y = []
	n = []

	for item in data_list:

		x.append(item['0'])
		y.append(item['1'])
		n.append(item['outlier'])

	create_scatter_plot(x, y, n, save=True, fig_saving_path=dataset_name+"_outliers_scatter_plot.png")


def visualize_predictions(predictions, saving_path, model_name):
	predictions_list = predictions.select('features', 'prediction').collect()

	x = []
	y = []
	p = []
	for item in predictions_list:
		x.append(item['features'][0])
		y.append(item['features'][1])
		p.append(item['prediction'])

	fig_saving_path = saving_path + model_name + '.png'

	create_scatter_plot(x, y, p, save=True, fig_saving_path=fig_saving_path)


