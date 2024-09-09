import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o conjunto de dados
url = 'URL_DO_CONJUNTO_DE_DADOS'
df = pd.read_csv('./rt_iot2022.csv')

# Remover a coluna 'Unnamed: 0' que é irrelevante para a análise
df.drop(columns=['Unnamed: 0'], inplace=True)

# Exibir as primeiras linhas para garantir que os dados estão corretos
print("Primeiras linhas do dataset:")
print(df.head())

# Tratamento de dados: remover duplicatas e preencher valores ausentes com 0
df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)

# Gráfico de barras: distribuição dos tipos de ataque
plt.figure(figsize=(10, 6))
attack_count = df['Attack_type'].value_counts()

# Exibir as contagens dos tipos de ataque
print("\nContagem dos tipos de ataque:")
print(attack_count)

sns.barplot(x=attack_count.index, y=attack_count.values)
plt.title('Distribuição dos Tipos de Ataque', fontsize=14)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('Quantidade', fontsize=12)
plt.xlabel('Tipo de Ataque', fontsize=12)
plt.tight_layout()  # Ajustar layout
plt.show()

# Gráfico de pizza: distribuição percentual dos tipos de ataque com legenda e porcentagens
plt.figure(figsize=(8, 8))
attack_counts = df['Attack_type'].value_counts()

# Calcular as porcentagens manualmente
percentages = (attack_counts / attack_counts.sum()) * 100
labels_with_percentages = [f'{label} ({percentage:.1f}%)' for label, percentage in zip(attack_counts.index, percentages)]

# Exibir as porcentagens dos tipos de ataque
print("\nPercentuais dos tipos de ataque:")
print(percentages)

# Criar o gráfico de pizza sem os rótulos diretamente no gráfico
plt.pie(attack_counts, labels=None, autopct='%1.1f%%', startangle=90, 
        textprops={'fontsize': 10}, pctdistance=0.85)

# Adicionar a legenda ao lado do gráfico com porcentagens
plt.legend(labels_with_percentages, title="Tipo de Ataque", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)

plt.title('Distribuição Percentual dos Tipos de Ataque', fontsize=14)
plt.tight_layout()  # Ajustar layout para evitar corte nos rótulos
plt.show()

# Gráfico de barras: relação entre 'fwd_init_window_size' (tamanho inicial da janela FWD) e tipos de ataque
plt.figure(figsize=(10, 6))
fwd_init_mean = df.groupby('Attack_type')['fwd_init_window_size'].mean()

# Exibir a média de 'fwd_init_window_size' por tipo de ataque
print("\nMédia de 'fwd_init_window_size' por tipo de ataque:")
print(fwd_init_mean)

sns.barplot(x=fwd_init_mean.index, y=fwd_init_mean.values)
plt.title('Tamanho Inicial da Janela (FWD) Médio por Tipo de Ataque', fontsize=14)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('Tamanho Médio', fontsize=12)
plt.xlabel('Tipo de Ataque', fontsize=12)
plt.tight_layout()  # Ajustar layout
plt.show()

# Gráfico de barras: relação entre 'bwd_init_window_size' (tamanho inicial da janela BWD) e tipos de ataque
plt.figure(figsize=(10, 6))
bwd_init_mean = df.groupby('Attack_type')['bwd_init_window_size'].mean()

# Exibir a média de 'bwd_init_window_size' por tipo de ataque
print("\nMédia de 'bwd_init_window_size' por tipo de ataque:")
print(bwd_init_mean)

sns.barplot(x=bwd_init_mean.index, y=bwd_init_mean.values)
plt.title('Tamanho Inicial da Janela (BWD) Médio por Tipo de Ataque', fontsize=14)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('Tamanho Médio', fontsize=12)
plt.xlabel('Tipo de Ataque', fontsize=12)
plt.tight_layout()  # Ajustar layout
plt.show()

# Gráfico de barras: relação entre 'fwd_last_window_size' (último tamanho da janela FWD) e tipos de ataque
plt.figure(figsize=(10, 6))
fwd_last_mean = df.groupby('Attack_type')['fwd_last_window_size'].mean()

# Exibir a média de 'fwd_last_window_size' por tipo de ataque
print("\nMédia de 'fwd_last_window_size' por tipo de ataque:")
print(fwd_last_mean)

sns.barplot(x=fwd_last_mean.index, y=fwd_last_mean.values)
plt.title('Último Tamanho da Janela (FWD) Médio por Tipo de Ataque', fontsize=14)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('Tamanho Médio', fontsize=12)
plt.xlabel('Tipo de Ataque', fontsize=12)
plt.tight_layout()  # Ajustar layout
plt.show()

# Gráfico de barras: relação entre 'flow_duration' (duração do fluxo) e tipos de ataque
plt.figure(figsize=(10, 6))
flow_duration_mean = df.groupby('Attack_type')['flow_duration'].mean()

# Exibir a média de 'flow_duration' por tipo de ataque
print("\nMédia de 'flow_duration' por tipo de ataque:")
print(flow_duration_mean)

sns.barplot(x=flow_duration_mean.index, y=flow_duration_mean.values)
plt.title('Duração Média do Fluxo por Tipo de Ataque', fontsize=14)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('Duração Média', fontsize=12)
plt.xlabel('Tipo de Ataque', fontsize=12)
plt.tight_layout()  # Ajustar layout
plt.show()

# Gráfico de barras: relação entre 'fwd_pkts_tot' (pacotes encaminhados) e tipos de ataque
plt.figure(figsize=(10, 6))
fwd_pkts_mean = df.groupby('Attack_type')['fwd_pkts_tot'].mean()

# Exibir a média de 'fwd_pkts_tot' por tipo de ataque
print("\nMédia de 'fwd_pkts_tot' por tipo de ataque:")
print(fwd_pkts_mean)

sns.barplot(x=fwd_pkts_mean.index, y=fwd_pkts_mean.values)
plt.title('Pacotes Encaminhados Médios por Tipo de Ataque', fontsize=14)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('Pacotes Médios', fontsize=12)
plt.xlabel('Tipo de Ataque', fontsize=12)
plt.tight_layout()  # Ajustar layout
plt.show()