import argparse

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F, SparkSession
from pyspark.sql import types as spark_types
from pyspark.ml.functions import vector_to_array

from utils import visualize_outliers


def post_processing(dataset_path):
	spark = SparkSession.builder.master("local[*]"). \
		appName("post_processing"). \
		getOrCreate()

	dataset_with_predicted_outliers = spark.read.json(dataset_path)

	dataset_with_predicted_outliers.persist()
	dataset_with_predicted_outliers.write.json("dataset_with_predicted_outliers_cure", mode="overwrite")
	tp = dataset_with_predicted_outliers.filter((F.col("prediction") == 1.0) & (F.col("outlier") == 1.0)).count()
	fp = dataset_with_predicted_outliers.filter((F.col("prediction") == 1.0) & (F.col("outlier") == 0.0)).count()
	tn = dataset_with_predicted_outliers.filter((F.col("prediction") == 0.0) & (F.col("outlier") == 0.0)).count()
	fn = dataset_with_predicted_outliers.filter((F.col("prediction") == 0.0) & (F.col("outlier") == 1.0)).count()
	visualize_outliers(dataset_with_predicted_outliers, dataset_path.split("/")[-1], prediction=True)
	dataset_with_predicted_outliers.unpersist()
	accuracy = (tp + tn) / (tp + tn + fp + fn)
	recall_macro = (tp / (tp + fn) + tn / (tn + fp)) / 2
	precision_macro = (tp / (tp + fp) + tn / (tn + fn)) / 2
	f1_macro = (2 * recall_macro * precision_macro) / (precision_macro + recall_macro)
	print("Accuracy: ", accuracy)
	print("Recall (macro): ", recall_macro)
	print("Precision (macro): ", precision_macro)
	print("F1 (macro): ", f1_macro)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--dataset_path",
		"-d",
		help="path of the dataset",
		default="Datasets/Data1_without_outliers"
	)
	args = parser.parse_args()
	post_processing(args.dataset_path)
