from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F, SparkSession
from pyspark.sql import types as spark_types
import argparse
import math
import numpy as np
from collections import defaultdict
from pyspark.ml.clustering import BisectingKMeans


def representatives(data_array,centers):
    centroid = 0
    point1 = []
    points_dist = defaultdict(list)
    for i in data_array:
        maxim = -1
        for j in i[1]:
            x = (j[0]-centers[centroid][0])*(j[0]-centers[centroid][0])
            y = (j[1]-centers[centroid][1])*(j[1]-centers[centroid][1])
            dist = math.sqrt(x+y)
            if maxim<dist:
                maxim = dist
                point = [j[0],j[1]]
        point1.append(point)
        points_dist[centroid].append(point)

        for j in i[1]:
            x = (j[0]-point1[centroid][0])*(j[0]-point1[centroid][0])
            y = (j[1]-point1[centroid][1])*(j[1]-point1[centroid][1])
            dist = math.sqrt(x+y)
            if max2< dist:
                max2 = dist
                point2 = [j[0],j[1]]
        points_dist[centroid].append(point2)

        centroid = centroid + 1
    print(point1)






def Cure(path,thresshold,k):
    spark = SparkSession.builder.master("local[*]").appName("kmeans").getOrCreate()

    df = spark.read.csv(path, header=True).select(F.col("0").cast(spark_types.FloatType()),\
                                                  F.col("1").cast(spark_types.FloatType()))



    sample = df.sample(False,0.0001,12)
    print(sample.count())
    assembler = VectorAssembler(
        inputCols=["0", "1"],
        outputCol="features")

    dataset = assembler.transform(sample)

    kmeans = BisectingKMeans().setK(k).setSeed(13) \
        .setFeaturesCol("features") \
        .setPredictionCol("prediction") \
        .setDistanceMeasure('euclidean')

    model = kmeans.fit(dataset)
    predictions = model.transform(dataset)


    centers = model.clusterCenters()



    pred = predictions.groupBy('prediction').agg(F.collect_list('features').alias('points'))
    pred = pred.sort(F.col('prediction').asc())

    pred.select('points').show()


    #Collect clustered sample data in main memory
    data_array = pred.rdd.collect()




    representatives(data_array,centers)









if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        "-d",
        help="path of the dataset",
        default="Data1.csv"
    )

    parser.add_argument(
        "--thresshold",
        "-th",
        help="Thresshold of representative points",
        default=2
    )

    parser.add_argument(
        "--kvalue",
        "-k",
        help="K value for hierarchical clustering",
        default=6
    )

    args = parser.parse_args()

    Cure(args.dataset_path,int(args.thresshold),int(args.kvalue))






