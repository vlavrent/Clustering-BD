import argparse

from pyspark.sql import functions as F, SparkSession
from pyspark.sql import types as spark_types
from utils import visualize_outliers
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_density_estimation(xx, yy, f, xmin, xmax, ymin, ymax, dataset_name):
	fig = plt.figure(figsize=(8, 8))
	ax = fig.gca()
	ax.set_xlim(xmin, xmax)
	ax.set_ylim(ymin, ymax)
	cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
	ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
	cset = ax.contour(xx, yy, f, colors='k')
	ax.clabel(cset, inline=1, fontsize=10)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	plt.title('2D Gaussian Kernel density estimation')
	plt.savefig(dataset_name + "_density_estimation.png")


def insert_outliers(dataset_path):
	spark = SparkSession.builder.master("local[*]"). \
		appName("insert_outliers"). \
		getOrCreate()

	dataset = spark.read.csv(dataset_path, header=True) \
		.select(F.col("0").cast(spark_types.FloatType()),
				F.col("1").cast(spark_types.FloatType())).withColumn("outlier", F.lit(0))

	X = np.array([[row['0'], row['1']] for row in dataset.collect()])

	# Extract x and y
	x = X[:, 0]
	y = X[:, 1]
	# Define the borders
	deltaX = (max(x) - min(x)) / 10
	deltaY = (max(y) - min(y)) / 10
	xmin = min(x) - deltaX
	xmax = max(x) + deltaX
	ymin = min(y) - deltaY
	ymax = max(y) + deltaY

	# Create meshgrid
	xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

	positions = np.vstack([xx.ravel(), yy.ravel()])
	values = np.vstack([x, y])
	kernel = st.gaussian_kde(values)
	f = np.reshape(kernel(positions).T, xx.shape)
	plot_density_estimation(xx, yy, f, xmin, xmax, ymin, ymax, dataset_path.split("/")[1])

	n = int(dataset.count() * 0.006)
	print("Starting number of outliers: ", n)
	xy_min = [-2, -2]
	xy_max = [2.5, 2]
	np.random.seed(seed=4)
	outlier_data = np.random.uniform(low=xy_min, high=xy_max, size=(n, 2))
	outlier_data_pdfs = kernel.evaluate(outlier_data.T)
	outlier_data = [point for index, point in enumerate(outlier_data)
					if 0.0001 < outlier_data_pdfs[index] < 0.005]
	print("Remaining number of outliers: ", len(outlier_data))
	df = pd.DataFrame(outlier_data, columns=list(["0", "1"]))
	outlier_dataset = spark.createDataFrame(df).withColumn("outlier", F.lit(1))

	dataset_with_outliers = dataset.union(outlier_dataset)

	visualize_outliers(dataset_with_outliers, dataset_path.split("/")[1])


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--dataset_path",
		"-d",
		help="path of the dataset",
		default="Datasets/Data1.csv"
	)
	args = parser.parse_args()
	insert_outliers(args.dataset_path)
