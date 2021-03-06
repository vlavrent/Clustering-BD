import numpy as np
from pyspark.sql import functions as F
from scipy.spatial.distance import cosine, euclidean
import matplotlib.pyplot as plt


def calculate_squared_distance(cluster_centers):
	def f(features, prediction):

		return float(np.sqrt(euclidean(features, cluster_centers[prediction])))

	return F.udf(f)


def calculate_sse(predictions, cluster_centers, prediction_column="prediction"):
	return predictions.select(
		calculate_squared_distance(cluster_centers)(F.col('features'), F.col(prediction_column))
			.alias('squared_distance')) \
		.agg(F.sum('squared_distance').alias('sse')).collect()[0]['sse']


def create_scatter_plot(x, y, c, s,legend_title, save=False, fig_saving_path=None):
	fig, ax = plt.subplots()

	scatter = ax.scatter(x, y, c=c, s=s)

	# produce a legend with the unique colors from the scatter
	legend1 = ax.legend(*scatter.legend_elements(),
						loc="upper right", title=legend_title)
	ax.add_artist(legend1)

	if save:
		fig.savefig(fig_saving_path)
	else:
		plt.show()

	plt.clf()
	plt.close(fig)


def visualize_outliers(dataset, dataset_name, prediction=False):
	data_list = dataset.collect()

	x = []
	y = []
	n = []

	for item in data_list:

		x.append(item['0'])
		y.append(item['1'])
		if prediction:
			n.append(item['prediction'])
		else:
			n.append(item['outlier'])

	s = len(x) * [1]

	if prediction:
		fig_saving_path = dataset_name + "_predicted_outliers_scatter_plot.png"
	else:
		fig_saving_path = dataset_name+"_outliers_scatter_plot.png"
	create_scatter_plot(x, y, n,  s, "outlier", save=True, fig_saving_path=fig_saving_path)


def visualize_predictions(predictions, saving_path, model_name, prediction_column='prediction'):
	predictions_list = predictions.select('features', prediction_column).collect()

	x = []
	y = []
	p = []
	for item in predictions_list:
		x.append(item['features'][0])
		y.append(item['features'][1])
		p.append(item[prediction_column])

	fig_saving_path = saving_path + model_name + '.png'

	s = len(x) * [1]
	create_scatter_plot(x, y, p, s, "clusters", save=True, fig_saving_path=fig_saving_path)


