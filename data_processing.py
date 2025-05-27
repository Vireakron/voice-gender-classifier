import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(csv_path="data/voice.csv"):
    df = pd.read_csv(csv_path)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])  # male: 1, female: 0

    X = df.drop('label', axis=1)
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test