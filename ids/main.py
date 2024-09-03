import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import GridSearchCV

# Configurações gerais
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.3f' % x)
plt.rcParams["figure.figsize"] = (10, 6)

# Carregando e copiando os dados
df_0 = pd.read_csv("KDDTrain+.txt")
df = df_0.copy()

# Definindo os nomes das colunas
columns = [
    'duração', 'tipo_de_protocolo', 'serviço', 'bandeira', 'bytes_de_origem', 'bytes_de_destino',
    'conexão_com_mesmo_host', 'fragmento_errado', 'urgente', 'quente', 'num_login_falhos', 'login_efetuado',
    'num_comprometidos', 'shell_root', 'tentativa_su', 'num_root', 'num_criações_de_arquivos',
    'num_shells', 'num_arquivos_acessados', 'num_comandos_de_saida', 'é_login_host',
    'é_login_convidado', 'contagem', 'contagem_srv', 'taxa_serror', 'taxa_srv_serror',
    'taxa_rerror', 'taxa_srv_rerror', 'taxa_mesmo_srv', 'taxa_srv_dif',
    'taxa_srv_host_dif', 'contagem_dest_host', 'contagem_srv_dest_host',
    'taxa_mesmo_srv_dest_host', 'taxa_srv_dif_dest_host', 'taxa_mesmo_porta_src_dest_host',
    'taxa_srv_host_dif_dest_host', 'taxa_serror_dest_host', 'taxa_srv_serror_dest_host',
    'taxa_rerror_dest_host', 'taxa_srv_rerror_dest_host', 'ataque', 'nível'
]
df.columns = columns

# Funções auxiliares para análise
def unique_values(df, columns):
    """Imprime valores únicos e suas contagens para colunas específicas no DataFrame."""
    for column_name in columns:
        print(f"Column: {column_name}\n{'-'*30}")
        unique_vals = df[column_name].unique()
        value_counts = df[column_name].value_counts()
        print(f"Unique Values ({len(unique_vals)}): {unique_vals}\n")
        print(f"Value Counts:\n{value_counts}\n{'='*40}\n")