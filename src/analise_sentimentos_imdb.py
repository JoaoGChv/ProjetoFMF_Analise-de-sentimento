#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projeto FMF: Análise de Sentimentos em Reviews de Filmes
Estrutura AGEMC - Fases A, G e E
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configurações para melhor visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def carregar_dados():
    """
    Fase G (Gathering) - Carregamento dos dados
    """
    print("=" * 60)
    print("FASE G - CARREGAMENTO DOS DADOS")
    print("=" * 60)
    
    try:
        # Carregando o dataset
        df = pd.read_csv('IMDB Dataset.csv')
        print(f"Dataset carregado com sucesso!")
        print(f"Dimensões do dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")
        return df
    except Exception as e:
        print(f"Erro ao carregar o dataset: {e}")
        return None

def informacoes_basicas(df):
    """
    Apresentação das informações básicas do DataFrame
    """
    print("\n" + "=" * 60)
    print("INFORMAÇÕES BÁSICAS DO DATAFRAME")
    print("=" * 60)
    
    print("\n📋 INFORMAÇÕES GERAIS:")
    print("-" * 40)
    print(df.info())
    
    print("\n🔍 PRIMEIRAS 5 LINHAS:")
    print("-" * 40)
    print(df.head())
    
    print("\n📈 ESTATÍSTICAS DESCRITIVAS:")
    print("-" * 40)
    print(df.describe())
    
    print("\n🏷️ COLUNAS DO DATASET:")
    print("-" * 40)
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")

def verificar_dados_faltantes(df):
    """
    Verificação de dados faltantes e sugestão de estratégia
    """
    print("\n" + "=" * 60)
    print("VERIFICAÇÃO DE DADOS FALTANTES")
    print("=" * 60)
    
    # Verificando dados faltantes
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    print("\n📊 DADOS FALTANTES POR COLUNA:")
    print("-" * 40)
    for col in df.columns:
        if missing_data[col] > 0:
            print(f"❌ {col}: {missing_data[col]} valores ({missing_percent[col]:.2f}%)")
        else:
            print(f"✅ {col}: Sem dados faltantes")
    
    # Sugestão de estratégia
    print("\n💡 ESTRATÉGIA SUGERIDA PARA DADOS FALTANTES:")
    print("-" * 40)
    if missing_data.sum() == 0:
        print("✅ Não há dados faltantes no dataset!")
    else:
        print("Para este tipo de problema de análise de sentimentos:")
        print("• Se houver reviews vazios: Remover as linhas (não faz sentido analisar texto vazio)")
        print("• Se houver sentimentos vazios: Remover as linhas (não podemos treinar sem labels)")
        print("• Para outros campos: Avaliar se a remoção impacta significativamente o dataset")

def analise_exploratoria(df):
    """
    Fase E (Exploration) - Análise exploratória dos dados
    """
    print("\n" + "=" * 60)
    print("FASE E - ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("=" * 60)
    
    # Verificando a distribuição das classes de sentimento
    print("\n📊 DISTRIBUIÇÃO DAS CLASSES DE SENTIMENTO:")
    print("-" * 40)
    sentiment_counts = df['sentiment'].value_counts()
    print(sentiment_counts)
    
    # Gráfico de barras para distribuição de sentimentos
    plt.figure(figsize=(10, 6))
    sentiment_counts.plot(kind='bar', color=['#2E8B57', '#DC143C'])
    plt.title('Distribuição de Sentimentos nos Reviews de Filmes', fontsize=16, fontweight='bold')
    plt.xlabel('Sentimento', fontsize=12)
    plt.ylabel('Quantidade de Reviews', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('distribuicao_sentimentos.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Análise de texto para word clouds
    print("\n🔤 ANÁLISE DE TEXTO PARA WORD CLOUDS:")
    print("-" * 40)
    
    # Função para limpar texto
    def limpar_texto(texto):
        if pd.isna(texto):
            return ""
        # Converter para minúsculas e remover caracteres especiais
        texto = re.sub(r'[^a-zA-Z\s]', '', str(texto).lower())
        return texto
    
    # Separando reviews por sentimento
    reviews_positivos = df[df['sentiment'] == 'positive']['review'].apply(limpar_texto)
    reviews_negativos = df[df['sentiment'] == 'negative']['review'].apply(limpar_texto)
    
    # Combinando todos os textos
    todos_textos = ' '.join(df['review'].apply(limpar_texto))
    textos_positivos = ' '.join(reviews_positivos)
    textos_negativos = ' '.join(reviews_negativos)
    
    print(f"📝 Total de palavras (geral): {len(todos_textos.split())}")
    print(f"📝 Total de palavras (positivo): {len(textos_positivos.split())}")
    print(f"📝 Total de palavras (negativo): {len(textos_negativos.split())}")
    
    # Criando word clouds
    print("\n☁️ GERANDO WORD CLOUDS...")
    
    # Word Cloud Geral
    wordcloud_geral = WordCloud(width=800, height=400, background_color='white', 
                               max_words=100, colormap='viridis').generate(todos_textos)
    
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Word Cloud Geral
    plt.subplot(2, 2, 1)
    plt.imshow(wordcloud_geral, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud - Todos os Reviews', fontsize=14, fontweight='bold')
    
    # Subplot 2: Word Cloud Positivo
    if textos_positivos.strip():
        wordcloud_positivo = WordCloud(width=400, height=300, background_color='white',
                                     max_words=50, colormap='Greens').generate(textos_positivos)
        plt.subplot(2, 2, 2)
        plt.imshow(wordcloud_positivo, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Reviews Positivos', fontsize=14, fontweight='bold')
    
    # Subplot 3: Word Cloud Negativo
    if textos_negativos.strip():
        wordcloud_negativo = WordCloud(width=400, height=300, background_color='white',
                                     max_words=50, colormap='Reds').generate(textos_negativos)
        plt.subplot(2, 2, 3)
        plt.imshow(wordcloud_negativo, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Reviews Negativos', fontsize=14, fontweight='bold')
    
    # Subplot 4: Estatísticas adicionais
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Estatísticas do dataset
    stats_text = f"""
    📊 ESTATÍSTICAS DO DATASET
    
    Total de Reviews: {len(df):,}
    
    Distribuição:
    • Positivos: {sentiment_counts.get('positive', 0):,} 
      ({sentiment_counts.get('positive', 0)/len(df)*100:.1f}%)
    • Negativos: {sentiment_counts.get('negative', 0):,}
      ({sentiment_counts.get('negative', 0)/len(df)*100:.1f}%)
    
    Comprimento médio dos reviews:
    • Positivos: {reviews_positivos.str.len().mean():.0f} caracteres
    • Negativos: {reviews_negativos.str.len().mean():.0f} caracteres
    """
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('wordclouds_analise.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Análise de comprimento dos reviews
    print("\n📏 ANÁLISE DE COMPRIMENTO DOS REVIEWS:")
    print("-" * 40)
    
    df['comprimento_review'] = df['review'].str.len()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    df.boxplot(column='comprimento_review', by='sentiment', ax=plt.gca())
    plt.title('Distribuição do Comprimento dos Reviews por Sentimento')
    plt.suptitle('')  # Remove o título automático
    
    plt.subplot(1, 2, 2)
    df['comprimento_review'].hist(bins=50, alpha=0.7, color='skyblue')
    plt.title('Distribuição Geral do Comprimento dos Reviews')
    plt.xlabel('Comprimento (caracteres)')
    plt.ylabel('Frequência')
    
    plt.tight_layout()
    plt.savefig('analise_comprimento.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📏 Comprimento médio dos reviews: {df['comprimento_review'].mean():.0f} caracteres")
    print(f"📏 Comprimento mediano dos reviews: {df['comprimento_review'].median():.0f} caracteres")
    print(f"📏 Comprimento mínimo: {df['comprimento_review'].min():.0f} caracteres")
    print(f"📏 Comprimento máximo: {df['comprimento_review'].max():.0f} caracteres")

def resumo_fase_aplicacao():
    """
    Resumo da Fase A (Aplicação) - Declaração da pergunta principal
    """
    print("\n" + "=" * 60)
    print("FASE A - APLICAÇÃO (PROBLEMA DE NEGÓCIO)")
    print("=" * 60)
    
    print("\n🎯 PERGUNTA PRINCIPAL DO PROJETO FMF:")
    print("-" * 40)
    print("Qual modelo de machine learning é mais eficaz para prever")
    print("o sentimento (positivo ou negativo) de um review de filme?")
    
    print("\n📋 CONTEXTO DO PROBLEMA:")
    print("-" * 40)
    print("• Objetivo: Classificar automaticamente reviews de filmes como positivos ou negativos")
    print("• Aplicação: Análise de sentimentos em plataformas de streaming, críticas de filmes")
    print("• Benefício: Automatizar a análise de feedback dos usuários")
    print("• Dataset: IMDB Movie Reviews com labels de sentimento")
    
    print("\n🔍 CRITÉRIOS DE AVALIAÇÃO:")
    print("-" * 40)
    print("• Acurácia: Capacidade de classificar corretamente os sentimentos")
    print("• Precisão e Recall: Balanceamento entre falsos positivos e negativos")
    print("• F1-Score: Métrica balanceada entre precisão e recall")
    print("• Velocidade: Tempo de processamento para novos reviews")

def main():
    """
    Função principal que executa todas as fases
    """
    print("🚀 INICIANDO PROJETO FMF: ANÁLISE DE SENTIMENTOS IMDB")
    print("=" * 60)
    
    # Fase G - Carregamento dos dados
    df = carregar_dados()
    if df is None:
        return
    
    # Informações básicas
    informacoes_basicas(df)
    
    # Verificação de dados faltantes
    verificar_dados_faltantes(df)
    
    # Fase E - Análise exploratória
    analise_exploratoria(df)
    
    # Fase A - Resumo da aplicação
    resumo_fase_aplicacao()
    
    print("\n" + "=" * 60)
    print("✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("=" * 60)
    print("\n📁 Arquivos gerados:")
    print("• distribuicao_sentimentos.png")
    print("• wordclouds_analise.png")
    print("• analise_comprimento.png")
    
    print("\n🔄 Próximos passos sugeridos:")
    print("• Fase M (Modelagem): Implementar diferentes algoritmos de ML")
    print("• Fase C (Comunicação): Apresentar resultados e conclusões")
    print("• Implementar técnicas de pré-processamento de texto")
    print("• Avaliar diferentes features para o modelo")

if __name__ == "__main__":
    main() 