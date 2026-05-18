from sklearn.cluster import KMeans
import joblib
import os

def train_kmeans(X, n_clusters):

    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        random_state=42
    )

    labels = model.fit_predict(X)

    save_model(model)

    return model, labels

def save_model(model):

    os.makedirs("saved_models", exist_ok=True)

    joblib.dump(
        model,
        "saved_models/kmeans.pkl"
    )