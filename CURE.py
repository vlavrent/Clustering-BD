from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F, SparkSession
from pyspark.sql import types as spark_types
import argparse
import math
import numpy as np
from collections import defaultdict
from pyspark.ml.clustering import BisectingKMeans
import pandas as pd
from scipy.spatial import distance
from pyspark.sql.functions import udf, when
from pyspark.sql.types import *
import time
from matplotlib import pyplot as plt
import pyspark.sql.functions as ps
from pyspark.ml.evaluation import ClusteringEvaluator
#from utils import calculate_sse
from tqdm import tqdm


def calculate_squared_distance(cluster_centers):
    def f(features, prediction):

        return float(np.sqrt(distance.euclidean(features, cluster_centers[prediction])))

    return F.udf(f)


def calculate_sse(predictions, cluster_centers):
    return predictions.select(
        calculate_squared_distance(cluster_centers)(F.col('features'), F.col('prediction'))
            .alias('squared_distance')) \
        .agg(F.sum('squared_distance').alias('sse')).collect()[0]['sse']


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




def assign_points(represent_points):
    def f(x, y):
        min  = 2000
        for centroid, representative in represent_points.items():
            for i in representative:
                dist = distance.euclidean([x,y],i)
                if dist<min:
                    min =dist
                    c = centroid
        return c
    return F.udf(f, IntegerType())

def calc_representatives(centers, threshold):
    def f(points, prediction):
        repPoints = representatives(points, centers[int(prediction)], threshold)

        return repPoints
    return F.udf(f, spark_types.ArrayType(spark_types.ArrayType(spark_types.FloatType())))




def calc_merged_representatives(threshold):
    def f(center, points):

        repPoints = representatives(points, center, threshold)




        return repPoints
    return F.udf(f, spark_types.ArrayType(ArrayType(FloatType())))



def merge_prediction():
    def f(merge):

        return merge[0]
    return F.udf(f, spark_types.IntegerType())



def merge_cluster(rep):
    def f(predictions,representatives):


        min_dist = 900
        cluster = [int(predictions),-1]
        for k,v in rep.items():
            if k!=int(predictions):
                for i in v:
                    for j in representatives:
                        dista = distance.euclidean(i,j)
                        if dista<min_dist:
                            min_dist = dista
                            if k<int(predictions):
                                clust = [k,int(predictions)]
                            else: clust = [int(predictions),k]
        if min_dist<0.5:
            cluster = clust



        return cluster

    return F.udf(f,spark_types.ArrayType(spark_types.IntegerType()))

def Clustering_plot(predictions_list,threshold,k):
    x = []
    y = []
    p = []
    for item in predictions_list:
        x.append(item['features'][0])
        y.append(item['features'][1])
        p.append(item['prediction'])


    s = len(x) * [1]
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y, c=p, s=s)
    plt.title("CURE with k"+str(k)+" and "+str(threshold)+" representatives")
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="Clusters")
    ax.add_artist(legend1)
    save=True
    if save:
        fig.savefig("CURE_"+str(k)+"_rep_"+str(threshold)+"_.png")
    else:
        plt.show()
    plt.clf()
    plt.close(fig)



def new_center():
    def f(points):
        s1 = 0
        s2 = 0

        c = []
        for i in points:
            l = len(i)
            for j in i:

                s1 = s1+float(j[0])
                s2 = s2+float(j[1])
        s1 = s1/l
        s2 = s2/l

        c = [s1,s2]

        return c
    return F.udf(f, spark_types.ArrayType(spark_types.FloatType()))


def Cure(path,threshold,k):
    spark = SparkSession.builder.master("local[*]").appName("kmeans").getOrCreate()

    df = spark.read.csv(path, header=True).select(F.col("0").cast(spark_types.FloatType()), \
                                                  F.col("1").cast(spark_types.FloatType()))

    threshold=4
    k=9


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
    predictions.persist()

    centers = model.clusterCenters()

    #Find representative points
    pred = predictions.groupBy('prediction').agg(F.collect_list('features').alias('points'))
    pred = pred.sort("prediction")
    #pred.show()

    pred = pred.withColumn("representatives", calc_representatives(centers, threshold)(F.col('points'), F.col('prediction')))
    #pred.select("prediction", "representatives").show(truncate=False)

    #Collect representative points in main memory
    df = assembler.transform(df)

    rep = pred.select("prediction","representatives").collect()


    points_dist = defaultdict(list)
    for i in rep:
        for j in i[1]:
            points_dist[i[0]].append(j)




    #Merge clusters
    merged = pred.withColumn("merge", merge_cluster(points_dist)(F.col("prediction"),F.col("representatives"))).groupBy('merge').agg(F.collect_list('points').alias('points'))
    merged = merged.withColumn("prediction", merge_prediction()(F.col("merge"))).withColumn("center", new_center()(F.col("points")))
    merged = merged.withColumn("new_points", F.flatten("points"))
    merged = merged.withColumn("representatives", calc_merged_representatives(threshold)(F.col('center'),F.col('new_points')))



    rep2 = merged.select("prediction","representatives").collect()


    dist_points = defaultdict(list)
    for i in rep2:
        for j in i[1]:
            dist_points[i[0]].append(j)
    print(dist_points)



    #Predict clusters for entire dataset
    prediction_df = df.withColumn("prediction", assign_points(dist_points)(F.col("0"), F.col("1")))
    prediction_df.show()
    #prediction_df.write.csv('predictions.csv')


    #Compute execution time of CURE
    end = time.time()
    total_time = end-start
    print("time:"+str(total_time))


    #Create plot CURE presictions
    #predictions_list = prediction_df.select('features', 'prediction').collect()
    #print(predictions_list)
    #prediction_list = predictions_list.rdd.collect()
    #Clustering_plot(predictions_list,threshold,k)



    #Evaluate CURE with silhouette score
    #evaluator = ClusteringEvaluator().setDistanceMeasure('squaredEuclidean')
    #silhouette = evaluator.evaluate(prediction_df)
    #print(silhouette)

    #Evaluate CURE with
    #sse = calculate_sse(prediction_df, centers)
    #print(sse)






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





