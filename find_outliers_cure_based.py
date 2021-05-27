from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F, SparkSession
from pyspark.sql import types as spark_types
import argparse
import numpy as np
from pyspark.ml.clustering import BisectingKMeans
import time
from CURE import calc_representatives
from scipy.spatial import distance
from utils import visualize_outliers

def predict_outliers(representative_points_dict, centers):
    def f(features):

        min_dist = 2000
        for centroid, values in representative_points_dict.items():
            for i in values["representatives"]:
                dist = distance.euclidean(features, i)
                if dist < min_dist:
                    min_dist = dist
                    c = centroid

        centroid_dist = distance.euclidean(features, centers[int(c)])
        threshold = representative_points_dict[c]["mean_dist_center"] \
                    + (2 * representative_points_dict[c]["std_dev_dist_center"])

        if centroid_dist > threshold:
            return 1.0
        else:
            return 0.0

    return F.udf(f, spark_types.DoubleType())


def calc_mean_distance_from_center(centers):
    def f(prediction, representatives):

        dists = []
        for representative in representatives:
            dists.append(distance.euclidean(representative, centers[int(prediction)]))

        return float(np.mean(dists))

    return F.udf(f, spark_types.DoubleType())


def calc_std_dev_distance_from_center(centers):
    def f(prediction, representatives):

        dists = []
        for representative in representatives:
            dists.append(distance.euclidean(representative, centers[int(prediction)]))

        return float(np.std(dists))

    return F.udf(f, spark_types.DoubleType())


def find_outliers(path, threshold, k):

    spark = SparkSession.builder.appName("find_outliers_cure_based").getOrCreate()

    spark.sparkContext.setLogLevel('WARN')
    all_dataset = spark.read.csv(path, header=True).select(F.col("0").cast(spark_types.FloatType()),
                                                  F.col("1").cast(spark_types.FloatType()),
                                                  F.col("outlier").cast(spark_types.DoubleType()))

    assembler = VectorAssembler(
        inputCols=["0", "1"],
        outputCol="features")
    all_dataset = assembler.transform(all_dataset)
    # Export a sample of data
    sampled_dataset = all_dataset.sample(False, 0.3, 7)

    start = time.time()

    # Apply hierarchical clustering in a sample of data
    kmeans = BisectingKMeans().setK(k).setSeed(13) \
        .setFeaturesCol("features") \
        .setPredictionCol("prediction") \
        .setDistanceMeasure('euclidean')

    model = kmeans.fit(sampled_dataset)
    predictions = model.transform(sampled_dataset)

    centers = model.clusterCenters()

    # Find representative points
    pred = predictions.groupBy('prediction').agg(F.collect_list('features').alias('points'))
    pred = pred.sort("prediction")

    representative_points = pred.withColumn("representatives", calc_representatives(
        centers, threshold)(F.col('points'), F.col('prediction')))\
        .select("prediction", "representatives")\
        .withColumn("mean_dist_center",
                    calc_mean_distance_from_center(centers)(F.col("prediction"), F.col("representatives")))\
        .withColumn("std_dev_dist_center",
                    calc_std_dev_distance_from_center(centers)(F.col("prediction"), F.col("representatives")))

    rep = representative_points.collect()

    representative_points_dict = {}
    for cluster in rep:
       representative_points_dict[cluster["prediction"]] = {
           "representatives": cluster["representatives"],
           "mean_dist_center": cluster["mean_dist_center"],
           "std_dev_dist_center": cluster["std_dev_dist_center"]
       }

    dataset_with_predicted_outliers = all_dataset \
        .withColumn("prediction", predict_outliers(representative_points_dict, centers)(F.col("features")))

    dataset_with_predicted_outliers.persist()
    dataset_with_predicted_outliers.write.json("dataset_with_predicted_outliers_cure", mode="overwrite")
    print("Time (seconds): ", time.time() - start)
    tp = dataset_with_predicted_outliers.filter((F.col("prediction") == 1.0) & (F.col("outlier") == 1.0)).count()
    fp = dataset_with_predicted_outliers.filter((F.col("prediction") == 1.0) & (F.col("outlier") == 0.0)).count()
    tn = dataset_with_predicted_outliers.filter((F.col("prediction") == 0.0) & (F.col("outlier") == 0.0)).count()
    fn = dataset_with_predicted_outliers.filter((F.col("prediction") == 0.0) & (F.col("outlier") == 1.0)).count()
    visualize_outliers(dataset_with_predicted_outliers, path.split("/")[-1], prediction=True)
    dataset_with_predicted_outliers.unpersist()
    accuracy = (tp+tn) / (tp+tn+fp+fn)
    recall_macro = (tp/(tp+fn) + tn/(tn+fp))/2
    precision_macro = (tp/(tp+fp) + tn/(tn+fn))/2
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
        default="Datasets/Data1_with_outliers"
    )

    parser.add_argument(
        "--threshold",
        "-th",
        help="Threshold of representative points",
        default=8
    )

    parser.add_argument(
        "--kvalue",
        "-k",
        help="K value for hierarchical clustering",
        default=6
    )

    args = parser.parse_args()

    find_outliers(args.dataset_path, int(args.threshold), int(args.kvalue))