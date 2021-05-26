from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F, SparkSession
from pyspark.sql import types as spark_types
import argparse
import numpy as np
from pyspark.ml.clustering import BisectingKMeans
import time
import scipy.stats as st
from CURE import calc_representatives


def predict_outliers(kernel, mean_representatives_pdf):
    def f(features):
        if kernel.evaluate(features.reshape(1, -1)) < 0.5 * mean_representatives_pdf:
            return 1.0
        else:
            return 0.0

    return F.udf(f, spark_types.DoubleType())


def find_outliers(path, threshold, k):

    spark = SparkSession.builder.master("local[*]").appName("find_outliers_kde").getOrCreate()

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

    representative_points = np.array(pred.withColumn("representatives", calc_representatives(
        centers, threshold)(F.col('points'), F.col('prediction')))\
        .select(F.explode("representatives")).collect())

    representative_points = representative_points.reshape((representative_points.shape[0], -1))

    X = np.array([[row['0'], row['1']] for row in all_dataset.select(["0", "1"]).collect()])
    x = X[:, 0]
    y = X[:, 1]
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    representatives_pdf = kernel.evaluate(np.array(representative_points.T))

    mean_representatives_pdf = np.mean(representatives_pdf)

    spark.sparkContext.broadcast(kernel)
    spark.sparkContext.broadcast(mean_representatives_pdf)
    dataset_with_predicted_outliers = all_dataset\
        .withColumn("prediction", predict_outliers(kernel, mean_representatives_pdf)(F.col("features")))

    dataset_with_predicted_outliers.persist()
    num_of_samples = dataset_with_predicted_outliers.count()
    true_predictions = dataset_with_predicted_outliers.filter(F.col("prediction") == F.col("outlier")).count()
    dataset_with_predicted_outliers.write.json("dataset_with_predicted_outliers_kde_0.5", mode="overwrite")
    print("Accuracy: ", true_predictions/num_of_samples)

    # evaluator = BinaryClassificationEvaluator()
    # evaluator.setLabelCol("outlier").setRawPredictionCol("prediction")
    # print("AUC ROC:", evaluator.evaluate(dataset_with_predicted_outliers))


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