import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

# page configuration
st.set_page_config(
    page_title="Smart Customer Profiling",
    layout="wide"
)

# custom button styling
st.markdown("""
<style>
div.stButton > button {
    height: 3em;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
}

div.stDownloadButton > button {
    height: 3em;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# title
st.title("Customer Segmentation Dashboard")

st.write(
    "Analyze customer behavior using Machine Learning clustering and interactive analytics."
)

# file uploader
uploaded_file = st.file_uploader(
    "Upload Customer CSV File",
    type=["csv"]
)

# process file
if uploaded_file is not None:

    try:

        # send file to FastAPI backend
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "text/csv"
            )
        }

        response = requests.post(
            "http://127.0.0.1:8000/cluster",
            files=files
        )

        # check response
        if response.status_code != 200:

            st.error("Backend API Error")

            st.write(response.text)

            st.stop()

        # convert response to dataframe
        data = response.json()

        df = pd.DataFrame(data)

        # dataset preview
        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        # KPI metrics
        st.subheader("Dashboard Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Customers",
            len(df)
        )

        col2.metric(
            "Number of Clusters",
            df["Cluster"].nunique()
        )

        col3.metric(
            "Average Spending Score",
            round(
                df["Spending Score (1-100)"].mean(),
                2
            )
        )

        # PCA visualization
        st.subheader("Customer Segments Visualization")

        features = [
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]

        pca = PCA(n_components=2)

        components = pca.fit_transform(
            df[features]
        )

        plot_df = pd.DataFrame({
            "PCA1": components[:, 0],
            "PCA2": components[:, 1],
            "Cluster": df["Cluster"].astype(str)
        })

        fig = px.scatter(
            plot_df,
            x="PCA1",
            y="PCA2",
            color="Cluster",
            title="Customer Segments"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # cluster table + download button
        col_left, col_right = st.columns([4, 1])

        with col_left:

            st.subheader("Cluster Insights Table")

        with col_right:

            st.write("")
            st.write("")

            csv = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇ Download Dataset",
                data=csv,
                file_name="segmented_customers.csv",
                mime="text/csv",
                use_container_width=True
            )

        # summary table
        summary = df.groupby("Cluster")[
            features
        ].mean()

        st.dataframe(
            summary,
            use_container_width=True
        )

        # final analysis summary
        st.subheader("AI-Powered Business Analysis")

        total_customers = len(df)

        avg_income = round(
            df["Annual Income (k$)"].mean(),
            2
        )

        avg_spending = round(
            df["Spending Score (1-100)"].mean(),
            2
        )

        highest_cluster = summary[
            "Spending Score (1-100)"
        ].idxmax()

        lowest_cluster = summary[
            "Spending Score (1-100)"
        ].idxmin()

        st.markdown(
            f"""
### Dataset Insights

The uploaded dataset contains **{total_customers} customers** segmented into multiple behavioral groups using Machine Learning clustering.

### Key Observations

- Average customer income is **{avg_income}k dollars**
- Average spending score is **{avg_spending}**
- Cluster **{highest_cluster}** represents the highest spending customer group
- Cluster **{lowest_cluster}** represents comparatively lower spending behavior

### Business Impact

The segmentation analysis helps businesses identify:

- high-value customers
- customer spending patterns
- budget-conscious groups
- engagement opportunities

These insights can support:

- personalized marketing
- targeted advertising campaigns
- customer retention strategies
- product recommendation systems

The generated segmented dataset can further be used for advanced analytics and business intelligence workflows.
"""
        )

    except Exception as e:

        st.error(f"Error: {e}")

else:

    st.info(
        "Please upload a CSV file to begin."
    )