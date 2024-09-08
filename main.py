import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Carregar o conjunto de dados
df = pd.read_csv('./rt_iot2022.csv')

# Remover a coluna 'Unnamed: 0' que é irrelevante para a análise
df.drop(columns=['Unnamed: 0'], inplace=True)

# Exibir as primeiras linhas para garantir que os dados estão corretos
print(df.head())

# Tratamento de dados: remover duplicatas e preencher valores ausentes com 0
df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)

# Filtrar colunas não numéricas e não relevantes para a análise
df_numeric = df.select_dtypes(include=['float64', 'int64'])

# Gráfico de barras: distribuição dos tipos de ataque
attack_count = df['Attack_type'].value_counts()
print("Distribuição dos Tipos de Ataque:")
print(attack_count)

plt.figure(figsize=(10, 6))
sns.barplot(x=attack_count.index, y=attack_count.values)
plt.title('Distribuição dos Tipos de Ataque', fontsize=14)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('Quantidade', fontsize=12)
plt.xlabel('Tipo de Ataque', fontsize=12)
plt.tight_layout()
plt.show()

# Gráfico de pizza: distribuição percentual dos tipos de ataque com legenda e porcentagens
percentages = (attack_count / attack_count.sum()) * 100
print("\nDistribuição Percentual dos Tipos de Ataque:")
print(percentages)

labels_with_percentages = [f'{label} ({percentage:.1f}%)' for label, percentage in zip(attack_count.index, percentages)]

plt.figure(figsize=(8, 8))
plt.pie(attack_count, labels=None, autopct='%1.1f%%', startangle=90, 
        textprops={'fontsize': 10}, pctdistance=0.85)
plt.legend(labels_with_percentages, title="Tipo de Ataque", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
plt.title('Distribuição Percentual dos Tipos de Ataque', fontsize=14)
plt.tight_layout()
plt.show()

# Gráficos de barras para métricas específicas
metrics = ['fwd_init_window_size', 'bwd_init_window_size', 'fwd_last_window_size', 'flow_duration', 'fwd_pkts_tot']
for metric in metrics:
    metric_mean = df.groupby('Attack_type')[metric].mean()
    print(f"\nMédia de {metric.replace('_', ' ').title()} por Tipo de Ataque:")
    print(metric_mean)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=metric_mean.index, y=metric_mean.values)
    plt.title(f'{metric.replace("_", " ").title()} Médio por Tipo de Ataque', fontsize=14)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel('Média', fontsize=12)
    plt.xlabel('Tipo de Ataque', fontsize=12)
    plt.tight_layout()
    plt.show()

# Agrupamento (Clustering) por tipo de ataque usando KMeans
features = ['fwd_init_window_size', 'bwd_init_window_size', 'fwd_last_window_size', 'flow_duration', 'fwd_pkts_tot']
X = df[features]

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar KMeans com 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Exibir clusters gerados
print("\nClusters Gerados:")
print(df['Cluster'].value_counts())

# Visualização com PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Gráfico de dispersão dos clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='Set1', s=100, alpha=0.7)
plt.title('Clusters de Ataques - PCA', fontsize=14)
plt.xlabel('Componente Principal 1', fontsize=12)
plt.ylabel('Componente Principal 2', fontsize=12)
plt.legend(title='Cluster', loc='best')
plt.show()

# Comparação de clusters com os tipos de ataque
print("\nComparação de Clusters com Tipos de Ataque:")
print(df.groupby('Attack_type')['Cluster'].value_counts())

plt.figure(figsize=(10, 6))
sns.countplot(x='Attack_type', hue='Cluster', data=df, palette='Set1')
plt.title('Comparação de Clusters por Tipo de Ataque', fontsize=14)
plt.xticks(rotation=90, fontsize=10)
plt.ylabel('Quantidade', fontsize=12)
plt.xlabel('Tipo de Ataque', fontsize=12)
plt.tight_layout()
plt.show()
