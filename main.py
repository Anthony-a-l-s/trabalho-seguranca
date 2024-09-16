import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregar o conjunto de dados
df = pd.read_csv('./rt_iot2022.csv')

# Remover a coluna 'Unnamed: 0' que é irrelevante para a análise
df.drop(columns=['Unnamed: 0'], inplace=True)

# Tratamento de dados: remover duplicatas e preencher valores ausentes com 0
df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)

# Preprocessamento
def preprocessamento(df):
    # Definir variáveis dependente e independentes
    X = df[['fwd_init_window_size', 'bwd_init_window_size', 'fwd_last_window_size', 'flow_duration', 'fwd_pkts_tot']]
    y = df['Attack_type']

    # Codificar a variável alvo
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Dividir em conjunto de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

    # Normalizar os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, le

# Função para treinar e avaliar modelos
def treinar_modelos(X_train, X_test, y_train, y_test, le):
    # Regressão Logística
    print("=== Regressão Logística ===")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print_classification_results(y_test, y_pred_lr, le)

    # Árvore de Decisão
    print("\n=== Árvore de Decisão ===")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    print_classification_results(y_test, y_pred_dt, le)

    # Floresta Aleatória
    print("\n=== Floresta Aleatória ===")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print_classification_results(y_test, y_pred_rf, le)

    # XGBoost
    print("\n=== XGBoost ===")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    print_classification_results(y_test, y_pred_xgb, le)

    # LightGBM
    print("\n=== LightGBM ===")
    lgb_model = lgb.LGBMClassifier(random_state=42)
    lgb_model.fit(X_train, y_train)
    y_pred_lgb = lgb_model.predict(X_test)
    print_classification_results(y_test, y_pred_lgb, le)

# Função para imprimir resultados da classificação
def print_classification_results(y_test, y_pred, le):
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc * 100:.2f}%")
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

# Preprocessar os dados
X_train, X_test, y_train, y_test, le = preprocessamento(df)

# Treinar e avaliar os modelos
treinar_modelos(X_train, X_test, y_train, y_test, le)
