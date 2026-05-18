from sklearn.preprocessing import StandardScaler

def preprocess_data(df, features):

    X = df[features]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled