# Customer-Segmentation

## It is an end-to-end Machine Learning application that performs customer segmentation using K-Means clustering. The project analyzes customer income and spending behavior to identify different customer groups for targeted business strategies and marketing insights.

## Features
1. Upload custom CSV datasets
2. Automatic customer segmentation using K-Means clustering
3. Interactive dashboard with customer analytics
4. PCA-based cluster visualization
5. KPI metrics and cluster insights
6. Download segmented dataset
7. AI-powered business analysis summary
8. REST API integration using FastAPI
9. Deployable full-stack ML architecture

## Tech Stack
1. Machine Learning
- Python
- Scikit-learn
- K-Means Clustering
- PCA (Principal Component Analysis)
- StandardScaler
2. Data Analysis
- Pandas
- NumPy
3. Visualization
- Plotly
- Matplotlib
- Seaborn
4. Backend
- FastAPI
- Uvicorn
5. Frontend
Streamlit
6. Deployment
Render
Streamlit Cloud

## Dataset
[mall_customers](https://www.kaggle.com/datasets/shwetabh123/mall-customers)
- Features Used:
- Annual Income (k$)
- Spending Score (1-100)

## Project Structure
```
Customer-Segmentation/
│
├── .venv/
│
├── backend/
│   │
│   ├── __pycache__/
│   ├── __init__.py
│   ├── clustering.py
│   ├── insights.py
│   ├── main.py
│   ├── preprocess.py
│   └── visualize.py
│
├── dataset/
│   └── mall_customers.csv
│
├── frontend/
│   └── app.py
│
├── notebook/
│   └── customer_segmentation.ipynb
│
├── saved_models/
│   ├── kmeans.pkl
│   └── segmented_customers.csv
│
├── .env
├── .gitignore
├── README.md
├── render.yaml
├── requirements.txt
└── runtime.txt
```
- Dashboard Functionalities:
1. Upload CSV dataset
2. View dataset preview
3. Display customer KPIs
4. Visualize clusters interactively
5. Analyze customer spending behavior
6. Download segmented dataset
7. Generate AI-powered business insights

- Model Evaluation:
The clustering model was evaluated using:

1. Silhouette Score
2. Elbow Method
3. PCA Cluster Separation

- Future Improvements:
1. OpenAI-powered business recommendations
2. Real-time analytics pipeline
3. Advanced customer profiling
4. Database integration
5. Authentication system
6. Docker support
7. CI/CD pipeline

*Open for collaborations!*
