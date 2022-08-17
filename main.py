import pandas as pd
import warnings
from datetime import datetime, timedelta
import numpy as np
from numpy.fft import fft
import math
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from scipy.stats import entropy

warnings.filterwarnings("ignore")

def get_datasets(insulin_data_file_path, cgm_data_file_path, date_time_format):
    if date_time_format == '':
        date_time_format = '%m/%d/%Y %H:%M:%S'
    insulin_dataset_full = pd.read_csv(insulin_data_file_path, low_memory = False)
    insulin_data = insulin_dataset_full[['Date', 'Time', 'BWZ Carb Input (grams)']]
    cgm_data_set_full = pd.read_csv(cgm_data_file_path, low_memory = False)
    cgm_data = cgm_data_set_full[['Date', 'Time', 'Sensor Glucose (mg/dL)']]
    cgm_data.dropna(inplace = True)
    insulin_data['DateTime'] = pd.to_datetime(insulin_data['Date'] + ' ' + insulin_data['Time'], format = date_time_format)
    cgm_data['DateTime'] = pd.to_datetime(cgm_data['Date'] + " " + cgm_data['Time'], format = date_time_format)
    return insulin_data, cgm_data

def get_meal_start_times_with_carb(insulin_dataset):
    insulin_data_carb = insulin_dataset[insulin_dataset['BWZ Carb Input (grams)'].notna() & insulin_dataset['BWZ Carb Input (grams)'] != 0]
    insulin_data_carb.rename({'DateTime' : 'MealStartDateTime'}, axis = 1, inplace = True)
    insulin_data_carb = insulin_data_carb[['MealStartDateTime', 'BWZ Carb Input (grams)']]
    insulin_data_carb.sort_values(by = 'MealStartDateTime', inplace = True)
    
    meal_start_times_with_carb = [tuple(x) for x in insulin_data_carb.to_numpy()]
    return meal_start_times_with_carb

def get_valid_meal_start_times_with_carb(meal_start_times_with_carb):
    valid_meal_start_times_with_carb = []
    for i in range(len(meal_start_times_with_carb)):
        timestamp, carb = meal_start_times_with_carb[i]
        if i > 0:
            previous_timestamp = meal_start_times_with_carb[i-1][0]
            if previous_timestamp > timestamp - timedelta(hours = 0.5):
                continue

        if i < len(meal_start_times_with_carb) - 1:
            next_timestamp = meal_start_times_with_carb[i+1][0]
            if next_timestamp < timestamp + timedelta(hours = 2):
                continue

        valid_meal_start_times_with_carb.append((timestamp, carb))
    return valid_meal_start_times_with_carb


def extract_meal_and_carb_data(cgm_dataset, valid_meal_start_times_with_carb):
    meal_data = []
    carb_data = []
    for meal_time, carb in valid_meal_start_times_with_carb:
        start_time = meal_time - timedelta(minutes = 30)
        end_time = meal_time + timedelta(hours = 2)
        filtered_data = cgm_dataset[(cgm_dataset['DateTime'] >= start_time) & (cgm_dataset['DateTime'] <= end_time)]
        if len(filtered_data) > 0:
            meal_data.append(list(filtered_data['Sensor Glucose (mg/dL)'].values))
            carb_data.append(carb)
    return meal_data, carb_data

def compute_slope_features(datarow): 
    """
    datarow: list of values
    This method computes differential values for every 3 consecutive points g1, g2, g3 at t1, t2, t3 in this way: 
    slope = (g1+g3-2g2)/(t3-t1)
    Then, at zero-crossing indices, |max-min slope| is calculated and top 3 such values are picked
    """
    slopes = []
    for i in range(len(datarow)-2):
        slopes.append((datarow[i] + datarow[i+2] - 2 * datarow[i+1]) / ((i+2-i) * 5.0)) #one reading per 5 minutes
    
    #plt.plot(slopes)
    #plt.axhline(y=0, color='r')
    
    zero_crossing_indices = np.where(np.diff(np.sign(slopes)))[0]
    #zero-crossing |max-min slope| with indices
    zero_crossing_delta = [(index, abs(slopes[index+1]-slopes[index])) for index in zero_crossing_indices]
    zero_crossing_delta.sort(key = lambda x: x[1], reverse = True)
    return zero_crossing_delta[:3] #top 3 values

def frequency_domain_features(datarow): #computes fft and returns the 2nd, 3rd, and 4th max freq indices
    frequencies = fft(datarow)
    #2nd, 3rd, and 4th max freq indices
    top_frequency_indices = np.argsort(frequencies)[::-1][1:4]
    return top_frequency_indices.tolist()

def extract_features(dataset): #dataset can be list of list values
    cgmMaxAndMinDiff = []
    cgmMaxAndMinTimeDiff = []
    slope_delta_1 = []
    slope_delta_1_loc = []
    slope_delta_2 = []
    slope_delta_2_loc = []
    slope_delta_3 = []
    slope_delta_3_loc = []
    slope_delta_features = [slope_delta_1, slope_delta_2, slope_delta_3]
    slope_delta_loc_features = [slope_delta_1_loc, slope_delta_2_loc, slope_delta_3_loc]
    fft_2 = []
    fft_3 = []
    fft_4 = []
    fft_features = [fft_2, fft_3, fft_4]

    for datarow in dataset:
        maxVal = max(datarow)
        minVal = min(datarow)
        #feature 1
        cgmMaxAndMinDiff.append(maxVal - minVal)

        #features 2-7
        slope_feature_tuples = compute_slope_features(datarow) #(l1, m1), (l2, m2), (l3, m3)
        for i in range(3): 
            if i < len(slope_feature_tuples):
                slope_delta_features[i].append(slope_feature_tuples[i][1])
                slope_delta_loc_features[i].append(slope_feature_tuples[i][0])
            else:
                slope_delta_features[i].append(None)
                slope_delta_loc_features[i].append(None)

        #feature 8
        cgmMaxAndMinTimeDiff.append((datarow.index(maxVal) - datarow.index(minVal)) * 5) #one reading per 5 minutes

        #features 9-11
        top_frequencies = frequency_domain_features(datarow)
        for i in range(3):
            if i < len(top_frequencies):
                fft_features[i].append(top_frequencies[i])
            else:
                fft_features[i].append(None)

    result_df = pd.DataFrame()
    result_df['CGM_Max_Min_Diff'] = cgmMaxAndMinDiff
    result_df['slope_delta_1'] = slope_delta_1
    #result_df['slope_delta_1_loc'] = slope_delta_1_loc
    result_df['slope_delta_2'] = slope_delta_2
    #result_df['slope_delta_2_loc'] = slope_delta_2_loc
    result_df['slope_delta_3'] = slope_delta_3
    #result_df['slope_delta_3_loc'] = slope_delta_3_loc
    result_df['CGM_Max_Min_Time_Diff'] = cgmMaxAndMinTimeDiff
    result_df['fft_2'] = fft_2
    result_df['fft_3'] = fft_3
    result_df['fft_4'] = fft_4
    return result_df

def normalize(df):
    #return (df - df.mean())/(df.max() - df.min())
    return (df - df.min())/((df.max() - df.min()) * 1.0)

def extract_features_and_normalize(meal_data):
    F_meal_data_df = extract_features(meal_data)
    F_meal_data_df = normalize(F_meal_data_df)
    return F_meal_data_df

def dropna_and_get_corresponding_carb_data(F_meal_data_df, carb_data):
    F_meal_data_df['Carb Input'] = carb_data
    F_meal_data_df.dropna(inplace = True)
    carb_data = F_meal_data_df['Carb Input']
    F_meal_data_df.drop(columns = ['Carb Input'], axis = 1, inplace = True)
    return F_meal_data_df, carb_data

def find_bin(val, min_val, bin_size):
    return int((val-min_val)/(bin_size * 1.0))

def extract_ground_truth(carb_data, bin_size = 20):
    ground_truth = []
    min_carb = min(carb_data)
    max_carb = max(carb_data)
    for carb in carb_data:
        ground_truth.append(find_bin(carb, min_carb, bin_size))
    bin_count = math.ceil((max_carb - min_carb) / (bin_size * 1.0))
    return ground_truth, int(bin_count)

def extract_features_and_ground_truth(insulin_data_file_path, cgm_data_file_path, date_time_format = '%m/%d/%Y %H:%M:%S'):
    insulin_dataset, cgm_dataset = get_datasets(insulin_data_file_path, cgm_data_file_path, date_time_format)
    meal_start_times_with_carb = get_meal_start_times_with_carb(insulin_dataset)
    valid_meal_start_times_with_carb = get_valid_meal_start_times_with_carb(meal_start_times_with_carb)
    meal_data, carb_data = extract_meal_and_carb_data(cgm_dataset, valid_meal_start_times_with_carb)
    
    F_meal_data_df = extract_features_and_normalize(meal_data)
    F_meal_data_df, carb_data = dropna_and_get_corresponding_carb_data(F_meal_data_df, carb_data)

    ground_truth, ground_truth_bin_count = extract_ground_truth(carb_data)

    return F_meal_data_df, ground_truth, ground_truth_bin_count

def form_ground_truth_bin_matrix(X, cluster_labels, ground_truth):
    numClusters = len(np.unique(cluster_labels))
    numBins = len(np.unique(ground_truth)) #no. of ground truth values (or bins)
    ground_truth_bin_matrix = np.zeros((numClusters, numBins)) #rows represent clusters; columns represent ground truth bins
    
    for i in range(len(X)):
        data = X[i]
        cluster_index = cluster_labels[i]
        bin_index = ground_truth[i]
        ground_truth_bin_matrix[cluster_index][bin_index] += 1
    return ground_truth_bin_matrix

def compute_entropy_and_purity(ground_truth_bin_matrix, X):
    numClusters = len(ground_truth_bin_matrix)
    numBins = len(ground_truth_bin_matrix[0])
    
    total_entropy = 0
    total_purity = 0
    
    for i in range(numClusters):
        Ci_count = sum(ground_truth_bin_matrix[i]) #number of elements in cluster i
        
        #normalizing each row
        for j in range(numBins):
            ground_truth_bin_matrix[i][j] /= (Ci_count * 1.0)
        
        Ci_entropy = entropy(ground_truth_bin_matrix[i]) #entropy of cluster i
        Ci_purity = max(ground_truth_bin_matrix[i])
        total_entropy += Ci_count * Ci_entropy
        total_purity += Ci_count * Ci_purity
        
    total_entropy /= len(X) * 1.0
    total_purity /= len(X) * 1.0
    
    return total_entropy, total_purity

def get_entropy_and_purity(X, cluster_labels, ground_truth):
    ground_truth_bin_matrix = form_ground_truth_bin_matrix(X, cluster_labels, ground_truth)
    entropy, purity = compute_entropy_and_purity(ground_truth_bin_matrix, X)
    return entropy, purity


def kmeans(X, numClusters, ground_truth):
    N = numClusters
    kmeans = KMeans(n_clusters = N, random_state=2).fit(X)
    kmeans_labels = kmeans.labels_
    kmeans_cluster_centers = kmeans.cluster_centers_
    #SSE
    kmeans_sse = 0
    for i in range(len(X)):
        data = X[i]
        cluster_index = kmeans_labels[i]
        centroid = kmeans_cluster_centers[cluster_index]
        dist = distance.euclidean(data, centroid)
        kmeans_sse += math.pow(dist, 2)

    kmeans_entropy, kmeans_purity = get_entropy_and_purity(X, kmeans_labels, ground_truth)
    return kmeans_sse, kmeans_entropy, kmeans_purity

def dbscan(X, ground_truth):
    dbscan = DBSCAN(eps = 0.60, min_samples = 22).fit(X)
    dbscan_labels = dbscan.labels_

    #Finding cluster mean
    clusters = {}
    for i in range(len(X)):
        data = X[i]
        cluster_index = dbscan_labels[i]
        if cluster_index not in clusters:
            clusters[cluster_index] = []
        clusters[cluster_index].append(data)

    cluster_means = {}
    for key in clusters:
        total = sum(clusters[key]) 
        count = len(clusters[key])
        cluster_means[key] = total/(count * 1.0)

    #DBSCAN SSE
    dbscan_sse = 0
    for i in range(len(X)):
        data = X[i]
        cluster_label = dbscan_labels[i]
        cluster_mean = cluster_means[cluster_label]
        dist = distance.euclidean(data, cluster_mean)
        dbscan_sse += math.pow(dist, 2)


    dbscan_entropy, dbscan_purity = get_entropy_and_purity(X, dbscan_labels, ground_truth)
    return dbscan_sse, dbscan_entropy, dbscan_purity

def retrieve_clusters(data_array, labels):
    clusters = []
    df = pd.DataFrame(data_array)
    df['label'] = labels
    unique_labels = df['label'].unique()
    for label in unique_labels:
        clusters.append(df[df['label'] == label].loc[:, df.columns != 'label'].to_numpy())
    return clusters

def compute_sse_for_clusters(clusters_list):
    sse_list = []
    for cluster in clusters_list:
        cluster_mean = sum(cluster)/(len(cluster) * 1.0)
        sse = 0
        for point in cluster:
            sse += math.pow(distance.euclidean(point, cluster_mean), 2)
        sse_list.append(sse)
    return sse_list

def partition_using_bisecting_kmeans(data_array):
    clustering = KMeans(n_clusters = 2, random_state=0).fit(data_array)
    clusters_list = retrieve_clusters(data_array, clustering.labels_)
    return clusters_list


def dbscan_with_bisecting_k_means(X, ground_truth):
    X_df = pd.DataFrame(X)
    X_df['Ground Truth Bin'] = ground_truth
    X_with_ground_truth = X_df.to_numpy()

    #dbscan = DBSCAN(eps=0.7,min_samples=22).fit(X)
    dbscan = DBSCAN(eps=0.8,min_samples=14).fit(X)
    dbscan_labels = dbscan.labels_

    clusters_list_with_ground_truth = retrieve_clusters(X_with_ground_truth, dbscan_labels)
    clusters_list_without_ground_truth = [cluster[:, :-1] for cluster in clusters_list_with_ground_truth]

    sse_list = compute_sse_for_clusters(clusters_list_without_ground_truth)
    while len(clusters_list_with_ground_truth) < N:
        max_sse_index = np.argsort(sse_list)[-1]
        new_clusters = partition_using_bisecting_kmeans(clusters_list_with_ground_truth[max_sse_index])
        clusters_list_with_ground_truth = clusters_list_with_ground_truth[:max_sse_index] + clusters_list_with_ground_truth[max_sse_index + 1 :] + new_clusters
        clusters_list_without_ground_truth = [cluster[:, :-1] for cluster in clusters_list_with_ground_truth]
        sse_list = compute_sse_for_clusters(clusters_list_without_ground_truth)
    dbscan_sse = sum(sse_list)

    data_array = []
    cluster_labels = []
    data_ground_truth_bins = []
    for i in range(len(clusters_list_with_ground_truth)):
        cluster_data_with_ground_truth = clusters_list_with_ground_truth[i]
        for data in cluster_data_with_ground_truth:
            data_array.append(data[:-1])
            data_ground_truth_bins.append(int(data[-1]))
            cluster_labels.append(i)
    dbscan_entropy, dbscan_purity = get_entropy_and_purity(data_array, cluster_labels, data_ground_truth_bins)
    return dbscan_sse, dbscan_entropy, dbscan_purity


#Final
F_meal_data_df, ground_truth, ground_truth_bin_count = extract_features_and_ground_truth('InsulinData.csv', 'CGMData.csv', '%m/%d/%Y %H:%M:%S')

X = F_meal_data_df.to_numpy()
N = int(ground_truth_bin_count)
kmeans_sse, kmeans_entropy, kmeans_purity = kmeans(X, N, ground_truth)
#dbscan_sse, dbscan_entropy, dbscan_purity = dbscan(X, ground_truth)
dbscan_sse, dbscan_entropy, dbscan_purity = dbscan_with_bisecting_k_means(X, ground_truth)
result_df = pd.DataFrame([[kmeans_sse, dbscan_sse, kmeans_entropy, dbscan_entropy, kmeans_purity, dbscan_purity]])
result_df.to_csv('Results.csv', index = False, header = False)

def calculate_kn_distance(X,k):
    kn_distance = []
    for i in range(len(X)):
        eucl_dist = []
        for j in range(len(X)):
            eucl_dist.append(distance.euclidean(X[i], X[j]))
        eucl_dist.sort()
        kn_distance.append(eucl_dist[k])
    return kn_distance
"""
k_dist = calculate_kn_distance(X, 33)
k_dist.sort()
plt.plot(k_dist)
plt.show()
"""
