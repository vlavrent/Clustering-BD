from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F, SparkSession
from pyspark.sql import types as spark_types
import argparse
import numpy as np
from collections import defaultdict
from pyspark.ml.clustering import BisectingKMeans
from scipy.spatial import distance
from pyspark.sql.functions import udf
from pyspark.sql.types import *
import time
from matplotlib import pyplot as plt
import pyspark.sql.functions as ps
from pyspark.ml.evaluation import ClusteringEvaluator




def dist(vecA, vecB):
    return np.sqrt(np.power(vecA - vecB, 2).sum())

def representatives(data,center,threshold):
    tempSet = None

    for i in range(1, threshold+1):
        maxDist = -100
        maxPoint = None
        for p in range(0, len(data)):

            if i == 1:
                minDist = dist(data[p], center)
            else:
                X = np.vstack([tempSet,data[p]])
                tmpDist = distance.pdist(X)
                minDist = tmpDist.min()
            if minDist >= maxDist:
                maxDist = minDist
                maxPoint = data[p]
        if tempSet is None:
            tempSet = maxPoint
        else:
            tempSet = np.vstack((tempSet, maxPoint))
    for j in range(len(tempSet)):
        if j==0:
            repPoints = None
        if repPoints is None:
            repPoints = tempSet[j,:] + 0.25 * (center - tempSet[j,:])
        else:
            repPoints = np.vstack((repPoints, tempSet[j,:] + 0.25 * (center - tempSet[j,:])))

    return repPoints.tolist()



def calc_representatives(centers, threshold):
    def f(points, prediction):
        repPoints = representatives(points, centers[int(prediction)], threshold)

        return repPoints
    return F.udf(f, spark_types.ArrayType(spark_types.ArrayType(spark_types.FloatType())))







def Cure(path,threshold,k):


    spark = SparkSession.builder.master("local[*]").appName("kmeans").getOrCreate()

    df = spark.read.csv(path, header=True).select(F.col("0").cast(spark_types.FloatType()), \
                                                  F.col("1").cast(spark_types.FloatType()))

    threshold=6
    k=4
    print("hey")


    #Export a sample of data
    sample = df.sample(False,0.3,7)

    assembler = VectorAssembler(
        inputCols=["0", "1"],
        outputCol="features")

    dataset = assembler.transform(sample)


    start = time.time()
    #Apply hierarchical clustering in a sample of data
    kmeans = BisectingKMeans().setK(k).setSeed(13) \
        .setFeaturesCol("features") \
        .setPredictionCol("prediction") \
        .setDistanceMeasure('euclidean')

    model = kmeans.fit(dataset)
    predictions = model.transform(dataset)

    centers = model.clusterCenters()

    #Find representative points
    pred = predictions.groupBy('prediction').agg(F.collect_list('features').alias('points'))
    pred.persist()
    pred = pred.sort("prediction")
    #pred.show()

    pred = pred.withColumn("representatives", calc_representatives(centers, threshold)(F.col('points'), F.col('prediction')))
    #pred.select("prediction", "representatives").show(truncate=False)



    pred.unpersist()











if __name__ == '__main__':



    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        "-d",
        help="path of the dataset",
        default="Data1.csv"
    )

    parser.add_argument(
        "--threshold",
        "-th",
        help="Threshold of representative points",
        default=2
    )

    parser.add_argument(
        "--kvalue",
        "-k",
        help="K value for hierarchical clustering",
        default=6
    )

    args = parser.parse_args()


    Cure(args.dataset_path,int(args.threshold),int(args.kvalue))