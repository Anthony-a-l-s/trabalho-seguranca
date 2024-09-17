import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Carregar o conjunto de dados
df = pd.read_csv('./rt_iot2022.csv')

# Remover a coluna 'Unnamed: 0' que é irrelevante para a análise
df.drop(columns=['Unnamed: 0'], inplace=True)

# Tratamento de dados: remover duplicatas e preencher valores ausentes com 0
df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)

# Função para gerar insights
def gerar_insights(df):
    # Métricas a serem analisadas
    metrics = ['fwd_init_window_size', 'bwd_init_window_size', 'fwd_last_window_size', 'flow_duration', 'fwd_pkts_tot']
    
    insights = {}
    
    # Contagem de tipos de ataque
    attack_counts = df['Attack_type'].value_counts()
    print("Contagem de tipos de ataque:\n", attack_counts)
    
    # Percentuais de tipos de ataque
    percentages = (attack_counts / attack_counts.sum()) * 100
    print("\nPercentuais dos tipos de ataque:\n", percentages)
    
    # Calcular a média e desvio padrão de cada métrica por tipo de ataque
    for metric in metrics:
        mean_values = df.groupby('Attack_type')[metric].mean()
        std_values = df.groupby('Attack_type')[metric].std()
        
        insights[metric] = {}
        for attack_type in mean_values.index:
            # Considerando desvios acima da média como insights significativos
            mean = mean_values[attack_type]
            std = std_values[attack_type]
            total_mean = df[metric].mean()
            
            # Identificar valores significativamente maiores ou menores que a média geral
            if mean > total_mean + std:
                insights[metric][attack_type] = f"Acima da média ({mean:.2f} vs média geral {total_mean:.2f})"
            elif mean < total_mean - std:
                insights[metric][attack_type] = f"Abaixo da média ({mean:.2f} vs média geral {total_mean:.2f})"
            else:
                insights[metric][attack_type] = f"Próximo à média ({mean:.2f} vs média geral {total_mean:.2f})"
    
    # Exibir insights para cada métrica
    for metric, metric_insights in insights.items():
        print(f"\nInsights para {metric}:")
        for attack_type, insight in metric_insights.items():
            print(f"- {attack_type}: {insight}")

# Função para gerar gráficos
def gerar_graficos(df):
    # Gráfico de barras: distribuição dos tipos de ataque
    plt.figure(figsize=(10, 6))
    attack_count = df['Attack_type'].value_counts()
    sns.barplot(x=attack_count.index, y=attack_count.values)
    plt.title('Distribuição dos Tipos de Ataque', fontsize=14)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel('Quantidade', fontsize=12)
    plt.xlabel('Tipo de Ataque', fontsize=12)
    plt.tight_layout()  # Ajustar layout
    plt.yscale('log')  # Escala logarítmica
    plt.show()

    # Gráfico de pizza: distribuição percentual dos tipos de ataque com legenda e porcentagens
    plt.figure(figsize=(8, 8))
    attack_counts = df['Attack_type'].value_counts()
    percentages = (attack_counts / attack_counts.sum()) * 100
    labels_with_percentages = [f'{label} ({percentage:.1f}%)' for label, percentage in zip(attack_counts.index, percentages)]
    plt.pie(attack_counts, labels=None, autopct='%1.1f%%', startangle=90, 
            textprops={'fontsize': 10}, pctdistance=0.85)
    plt.legend(labels_with_percentages, title="Tipo de Ataque", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.title('Distribuição Percentual dos Tipos de Ataque', fontsize=14)
    plt.tight_layout()  # Ajustar layout para evitar corte nos rótulos
    plt.show()

    # Gráfico de barras: relação entre 'fwd_init_window_size' e tipos de ataque
    plt.figure(figsize=(10, 6))
    fwd_init_mean = df.groupby('Attack_type')['fwd_init_window_size'].mean()
    sns.barplot(x=fwd_init_mean.index, y=fwd_init_mean.values)
    plt.title('Tamanho Inicial da Janela (FWD) Médio por Tipo de Ataque', fontsize=14)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel('Tamanho Médio', fontsize=12)
    plt.xlabel('Tipo de Ataque', fontsize=12)
    plt.yscale('log')  # Escala logarítmica
    plt.tight_layout()  # Ajustar layout
    plt.show()

    # Gráfico de barras: relação entre 'bwd_init_window_size' e tipos de ataque
    plt.figure(figsize=(10, 6))
    bwd_init_mean = df.groupby('Attack_type')['bwd_init_window_size'].mean()
    sns.barplot(x=bwd_init_mean.index, y=bwd_init_mean.values)
    plt.title('Tamanho Inicial da Janela (BWD) Médio por Tipo de Ataque', fontsize=14)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel('Tamanho Médio', fontsize=12)
    plt.xlabel('Tipo de Ataque', fontsize=12)
    plt.yscale('log')  # Escala logarítmica
    plt.tight_layout()  # Ajustar layout
    plt.show()

    # Gráfico de barras: relação entre 'fwd_last_window_size' e tipos de ataque
    plt.figure(figsize=(10, 6))
    fwd_last_mean = df.groupby('Attack_type')['fwd_last_window_size'].mean()
    sns.barplot(x=fwd_last_mean.index, y=fwd_last_mean.values)
    plt.title('Último Tamanho da Janela (FWD) Médio por Tipo de Ataque', fontsize=14)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel('Tamanho Médio', fontsize=12)
    plt.xlabel('Tipo de Ataque', fontsize=12)
    plt.yscale('log')  # Escala logarítmica
    plt.tight_layout()  # Ajustar layout
    plt.show()

    # Gráfico de barras: relação entre 'flow_duration' e tipos de ataque
    plt.figure(figsize=(10, 6))
    flow_duration_mean = df.groupby('Attack_type')['flow_duration'].mean()
    sns.barplot(x=flow_duration_mean.index, y=flow_duration_mean.values)
    plt.title('Duração Média do Fluxo por Tipo de Ataque', fontsize=14)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel('Duração Média', fontsize=12)
    plt.xlabel('Tipo de Ataque', fontsize=12)
    plt.yscale('log')  # Escala logarítmica
    plt.tight_layout()  # Ajustar layout
    plt.show()

    # Gráfico de barras: relação entre 'fwd_pkts_tot' e tipos de ataque
    plt.figure(figsize=(10, 6))
    fwd_pkts_mean = df.groupby('Attack_type')['fwd_pkts_tot'].mean()
    sns.barplot(x=fwd_pkts_mean.index, y=fwd_pkts_mean.values)
    plt.title('Pacotes Encaminhados Médios por Tipo de Ataque', fontsize=14)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel('Pacotes Médios', fontsize=12)
    plt.xlabel('Tipo de Ataque', fontsize=12)
    plt.yscale('log')  # Escala logarítmica
    plt.tight_layout()  # Ajustar layout
    plt.show()

# Executar gráficos
gerar_graficos(df)

# Executar a função para gerar insights
gerar_insights(df)

##
