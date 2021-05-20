# Clustering-BD

This is the repo for our project in the course Mining from Massive Datasets of MSc in Data and Web Science of AUTH

## Installation of requirements

python 3.7 + 

```bash
pip install -r requirements.txt
```

## Usage

### Run cure.py
```bash

```

### Run clustering.py
You should select these parameters: 
* -d dataset path 
* -a name of clustering algorithm {kmeans|bkmeans} 
     bkmeans = bisecting kmeans is hierarchical clustring algorithm 
* -s start value of k for fine tuning
* -e end value of k for fine tuning

```bash
spark-submit --master local[*] --driver-memory 8g clustering.py -d Datasets/Data1.csv -a kmeans -s 2 -e 14

```

### Run insert_outliers.py
You should select these parameters: 
* -d dataset path 
* -s saving path 
* -p percentage for uniform sampling 
* -n number of desired duplicate outliers 

```bash
spark-submit --master local[*] --driver-memory 8g insert_outliers.py -d Datasets/Data1.csv -s Datasets/Data1_with_outliers -p 0.025 -n 20

```

### Run find_outliers_kde.py
You should select these parameters: 
* -d dataset path 
* -th threshold of representatives points
* -k K value for hierarchical clustering


```bash
spark-submit --master local[*] --driver-memory 8g find_outliers_kde.py -d Datasets/Data1_with_outliers -th 8 -k 6

```

### Run find_outliers_cure_based.py
You should select these parameters: 
* -d dataset path 
* -th threshold of representatives points
* -k K value for hierarchical clustering


```bash
spark-submit --master local[*] --driver-memory 8g find_outliers_cure_based.py -d Datasets/Data1_with_outliers -th 8 -k 6

```


