import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import requests
import joblib
import os

def download_titanic_data(save_path: str):

    if os.path.exists(save_path):
        print(f"File {save_path} already exists. Skipping download.")
        return

    url = "https://hbiostat.org/data/repo/titanic3.csv"

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"downloaded {save_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the dataset: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

def train():
    """
    Функция загружает данные (или использует локальный файл), выполняет предобработку,
    создаёт новые признаки, обучает модель градиентного бустинга и сохраняет артефакты.
    """
    # Пути к файлам
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'Titanic.csv')
    model_path = os.path.join(base_dir, 'models', 'gradient_boost_model.pkl')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
    metrics_path = os.path.join(base_dir, 'models', 'metrics.txt')

    # папка для артефактов, если её нет
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    download_titanic_data(data_path)

    # чение данных
    df = pd.read_csv(data_path)

    cols_to_drop = ['passengerId', 'name', 'ticket', 'fare', 'cabin', 'embarked']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    df['age'] = df['age'].fillna(df['age'].median())
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})

    # создание новых признаков 
    df['familysize'] = df['sibsp'] + df['parch'] + 1
    df['isalone'] = (df['familysize'] == 1).astype(int)
    df['minorage'] = (df['age'] < 18).astype(int)
    df['malerisk'] = ((df['sex'] == 0) & (df['pclass'] == 3) & (df['age'] >= 16) & (df['age'] <= 45) &(df['isalone'] == 1)).astype(int)


    features = ['pclass', 'sex', 'age', 'sibsp', 'parch']
    new_features = ['familysize', 'isalone', 'minorage', 'malerisk']
    all_features = features + new_features
    target = 'survived'

    X = df[all_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.01,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)

    loss = log_loss(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # сохранение модели и скейлера
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # запись метрик в файл
    with open(metrics_path, 'w') as f:
        f.write(f"Log Loss: {loss:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"TN: {cm[0,0]}  FP: {cm[0,1]}\n")
        f.write(f"FN: {cm[1,0]}  TP: {cm[1,1]}\n")

    # логирование в stdout для Airflow
    print("Model training completed.")
    print(f"Metrics saved to {metrics_path}")
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

    return {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }

if __name__ == "__main__":
    train()
