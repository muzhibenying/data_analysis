import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import spacy
from sklearn.cluster import DBSCAN

def get_features(csv_path, mode):
    df = pd.read_excel(csv_path, engine = "openpyxl")
    mapbegin = np.array(df.groupby("mapbegin").mapbegin.apply(lambda a: a.iloc[0]).tolist())
    mapbeginlon = np.array(df.groupby("mapbegin").mapbeginlon.min().tolist())
    mapbeginlat = np.array(df.groupby("mapbegin").mapbeginlat.max().tolist())
    taxi_count = np.array(df.groupby("mapbegin").打车次数.sum().tolist())

    # embedding the name of the place
    mapbegin_vector = []
    nlp = spacy.load("zh_core_web_sm")
    for name in mapbegin:
        mapbegin_vector.append(nlp(name).vector)
    mapbegin_vector = np.array(mapbegin_vector)

    # only use the name of the places as features
    if mode == "name":
        features = mapbegin_vector
    
    return features

def distance(feature_1, feature_2, mode):
    if mode == "cosine_similarity":
        return np.sum(feature_1 * feature_2) / (np.sqrt(np.sum(feature_1 * feature_1))) \
                                             * (np.sqrt(np.sum(feature_2 * feature_2)))
    if mode == "L2":
        return np.sum((feature_1 - feature_2) * (feature_1 - feature_2))

def sorted_distance(features, minPts):
    minPts_distances = []
    for feature in features:
        distances = [distance(feature, feature_2, "L2") for feature_2 in features]
        minPts_distances.append(sorted(distances)[minPts])
    plt.plot(sorted(minPts_distances))
    plt.xlabel("Points Sorted According to Distance of " + str(minPts) + "th Nearest Neighbor")
    plt.ylabel(str(minPts) + "th Nearest Neighbor Distance")
    plt.savefig("minPts/" + str(minPts) + "th Nearset Neighbor Distance.png")
    plt.close()


def main():
    """
    The process of clustering places:
    - 1. get features from the data
    - 2. determine the parameters of clustering algorithms
    - 3. use the dbscan algorithms to clustering data
    """
    features = get_features(csv_path = "../../武汉打车.xlsx", mode = "name")
    #for minPts in range(1, 15):
        #sorted_distance(features, minPts)
    clustering = DBSCAN(eps = 15, min_samples = 4, metric = "l2").fit(features)
    print(clustering.labels_)

    

if __name__ == "__main__":
    main()