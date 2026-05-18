import plotly.express as px
from sklearn.decomposition import PCA
import pandas as pd

def plot_clusters(X, labels):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    df = pd.DataFrame({
        "PCA1": components[:, 0],
        "PCA2": components[:, 1],
        "Cluster": labels.astype(str)
    })

    fig = px.scatter(
        df,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        title="Customer Segments"
    )

    return fig

def elbow_chart(wcss):
    df = pd.DataFrame({
        "Clusters": range(1, 11),
        "WCSS": wcss
    })

    fig = px.line(
        df,
        x="Clusters",
        y="WCSS",
        markers=True,
        title="Elbow Method"
    )

    return fig