from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F, SparkSession
from pyspark.sql import types as spark_types
import argparse
import numpy as np
from pyspark.ml.clustering import BisectingKMeans
import time
from CURE import calc_representatives
from scipy.spatial import distance


def predict_outliers(represent_points):
    def f(features):

        min = 2000
        for centroid, representative in represent_points.items():
            for i in representative:
                dist = distance.euclidean(features, i)
                if dist < min:
                    min = dist
                    c = centroid

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

    spark = SparkSession.builder.master("local[*]").appName("find_outliers_cure_based").getOrCreate()

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

    print(centers)
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
    representative_points.show(truncate=False)


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
        default=4
    )

    parser.add_argument(
        "--kvalue",
        "-k",
        help="K value for hierarchical clustering",
        default=6
    )

    args = parser.parse_args()

    find_outliers(args.dataset_path, int(args.threshold), int(args.kvalue))