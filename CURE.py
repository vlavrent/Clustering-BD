from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F, SparkSession
from pyspark.sql import types as spark_types
import argparse
import math
import numpy as np
from collections import defaultdict
from pyspark.ml.clustering import BisectingKMeans
import pandas as pd
from scipy.spatial.distance import euclidean
from pyspark.sql.functions import udf
from pyspark.sql.types import *


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

        return repPoints



def assign_points(represent_points):
        def f(x, y):
                min  = 2000
                for centroid, representative in represent_points.items():
                        for i in representative:
                                for j in i:
                                        dist = euclidean([x,y],j)
                                        if dist<min:
                                                min =dist
                                                c = centroid
                return c
        return F.udf(f, IntegerType())





def Cure(path,threshold,k):
   spark = SparkSession.builder.master("local[*]").appName("kmeans").getOrCreate()

        df = spark.read.csv(path, header=True).select(F.col("0").cast(spark_types.FloatType()), \
                                                      F.col("1").cast(spark_types.FloatType()))

        threshold=4
        k=3



        sample = df.sample(False,0.3,7)

        assembler = VectorAssembler(
                inputCols=["0", "1"],
                outputCol="features")

        dataset = assembler.transform(sample)

        #total_time = {}
        silhouette_score ={}


        start = time.time()

        kmeans = BisectingKMeans().setK(k).setSeed(13) \
                .setFeaturesCol("features") \
                .setPredictionCol("prediction") \
                .setDistanceMeasure('euclidean')

        model = kmeans.fit(dataset)
        predictions = model.transform(dataset)
        predictions.persist()

        centers = model.clusterCenters()

        pred = predictions.groupBy('prediction').agg(F.collect_list('features').alias('points'))

        pred = pred.sort("prediction")



        #Collect clustered sample data in main memory
        data_array = pred.rdd.collect()

        points_dist = defaultdict(list)
        for i in range(0,len(centers)):
                rep_Points = representatives(data_array[i][1],centers[i],threshold)
                points_dist[i].append(rep_Points.tolist())

    df = assembler.transform(df)
    #df.show()



    prediction_df = df.withColumn("prediction", assign_points(points_dist)(F.col("0"), F.col("1")))
    prediction_df.show()




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






