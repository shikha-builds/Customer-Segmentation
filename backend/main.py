from fastapi import FastAPI, UploadFile, File
import pandas as pd

from backend.preprocess import preprocess_data
from backend.clustering import train_kmeans

app = FastAPI()

@app.get("/")
def home():

    return {
        "message": "CustomerIQ API Running"
    }

@app.post("/cluster")
async def cluster_data(
    file: UploadFile = File(...)
):

    df = pd.read_csv(file.file)

    features = [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]

    X_scaled = preprocess_data(
        df,
        features
    )

    model, labels = train_kmeans(
        X_scaled,
        5
    )

    df["Cluster"] = labels

    return df.to_dict(orient="records")