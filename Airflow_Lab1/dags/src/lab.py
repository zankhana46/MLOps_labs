import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from sklearn.cluster import KMeans  # KMeans code kept for reference
# from kneed import KneeLocator  # KMeans elbow method
from sklearn.cluster import AgglomerativeClustering
import pickle
import os


def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.

    Returns:
        bytes: Serialized data.
    """

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    serialized_data = pickle.dumps(df)
    
    return serialized_data
    

def data_preprocessing(data):

    """
    Deserializes data, performs data preprocessing, and returns serialized clustered data.

    Args:
        data (bytes): Serialized data to be deserialized and processed.

    Returns:
        bytes: Serialized clustered data.
    """
    df = pickle.loads(data)
    df = df.dropna()
    clustering_data = df[["PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "PAYMENTS", "MINIMUM_PAYMENTS"]].head(100)  # Limit to 100 rows
    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)
    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return clustering_serialized_data


def build_save_model(data, filename, use_dbscan=False, dbscan_eps=0.5, dbscan_min_samples=5, use_hierarchical=False, n_clusters=2):
    """
    Builds a clustering model (AgglomerativeClustering), saves it to a file.

    Args:
        data (bytes): Serialized data for clustering.
        filename (str): Name of the file to save the clustering model.
        use_hierarchical (bool): If True, use AgglomerativeClustering.
        n_clusters (int): Number of clusters for AgglomerativeClustering.

    Returns:
        None
    """
    df = pickle.loads(data)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    if use_hierarchical:
        model = AgglomerativeClustering(n_clusters=n_clusters)
        model.fit(df)
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        return None
    else:
        # KMeans code kept for reference
        # kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 300,"random_state": 42,}
        # sse = []
        # for k in range(1, 50):
        #     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        #     kmeans.fit(df)
        #     sse.append(kmeans.inertia_)
        # with open(output_path, 'wb') as f:
        #     pickle.dump(kmeans, f)
        # return sse
        pass  # Remove this if you want to use KMeans again

def load_hierarchical_model(filename):
    """
    Loads a saved AgglomerativeClustering model and predicts clusters for test data.

    Args:
        filename (str): Name of the file containing the saved clustering model.

    Returns:
        array: Predicted cluster labels for the test data.
    """
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, 'rb'))
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    df = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]].dropna().head(10)
    if len(df) < 2:
        raise ValueError("Not enough samples in test data for clustering. At least 2 required.")
    predictions = loaded_model.fit_predict(df)
    n_clusters = len(set(predictions))
    print(f"Number of clusters found by AgglomerativeClustering: {n_clusters}")
    print(f"Cluster labels: {predictions}")
    return predictions
    

# KMeans elbow method code kept for reference
# def load_model_elbow(filename,sse):
#     output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
#     loaded_model = pickle.load(open(output_path, 'rb'))
#     df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
#     kl = KneeLocator(range(1, 50), sse, curve="convex", direction="decreasing")
#     print(f"Optimal no. of clusters: {kl.elbow}")
#     predictions = loaded_model.predict(df)
#     return predictions[0]
