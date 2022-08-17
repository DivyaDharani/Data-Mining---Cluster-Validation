# Data Mining - Cluster Validation

In this project, clusters are formed from the sensor data taken from an Artificial Pancreas System and are validated using different metrics.

This is a continuation of the project https://github.com/DivyaDharani/Data-Mining---Machine-Learning-Model/. Refer to this to know more about meal and no-meal data as well as the extracted features.

## What 
* Features are extracted from meal data (refer to the link mentioned above)
* Meal data are then clustered based on the amount of carbohydrates in each meal

## Clustering Algorithms used
* DBSCAN
* K-Means

Clusters are validated by extracting the ground truth data based on the meal intake amount and then calculating the metrics given below.

## Metrics calculated
* SSE
* Entropy
* Purity
