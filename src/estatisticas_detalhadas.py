#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estatísticas Detalhadas - Dataset IMDB
Análise complementar com informações adicionais
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

def analise_detalhada():
    """
    Análise detalhada com estatísticas adicionais
    """
    print("=" * 60)
    print("ANÁLISE DETALHADA - DATASET IMDB")
    print("=" * 60)
    
    # Carregando dados
    df = pd.read_csv('IMDB Dataset.csv')
    
    # 1. Análise de palavras por review
    print("\nANÁLISE DE PALAVRAS POR REVIEW:")
    print("-" * 40)
    
    def contar_palavras(texto):
        if pd.isna(texto):
            return 0
        return len(str(texto).split())
    
    df['num_palavras'] = df['review'].apply(contar_palavras)
    
    print(f"Média de palavras por review: {df['num_palavras'].mean():.1f}")
    print(f"Mediana de palavras por review: {df['num_palavras'].median():.1f}")
    print(f"Desvio padrão: {df['num_palavras'].std():.1f}")
    print(f"Mínimo: {df['num_palavras'].min()}")
    print(f"Máximo: {df['num_palavras'].max()}")
    
    # Por sentimento
    positivos = df[df['sentiment'] == 'positive']
    negativos = df[df['sentiment'] == 'negative']
    
    print(f"\nPOSITIVOS:")
    print(f"   Média: {positivos['num_palavras'].mean():.1f}")
    print(f"   Mediana: {positivos['num_palavras'].median():.1f}")
    
    print(f"\nNEGATIVOS:")
    print(f"   Média: {negativos['num_palavras'].mean():.1f}")
    print(f"   Mediana: {negativos['num_palavras'].median():.1f}")
    
    # 2. Análise de reviews duplicados
    print("\nANÁLISE DE REVIEWS DUPLICADOS:")
    print("-" * 40)
    
    duplicados = df[df.duplicated(subset=['review'], keep=False)]
    print(f"Total de reviews duplicados: {len(duplicados)}")
    print(f"Percentual de duplicados: {(len(duplicados)/len(df)*100):.2f}%")
    
    if len(duplicados) > 0:
        print("\nEXEMPLOS DE REVIEWS DUPLICADOS:")
        for i, (idx, row) in enumerate(duplicados.head(3).iterrows()):
            print(f"   {i+1}. Review: {row['review'][:100]}...")
            print(f"      Sentimento: {row['sentiment']}")
            print()
    
    # 3. Análise de sentenças por review
    print("\nANÁLISE DE SENTENÇAS POR REVIEW:")
    print("-" * 40)
    
    def contar_sentencas(texto):
        if pd.isna(texto):
            return 0
        # Conta sentenças baseado em pontuação
        sentencas = re.split(r'[.!?]+', str(texto))
        return len([s for s in sentencas if s.strip()])
    
    df['num_sentencas'] = df['review'].apply(contar_sentencas)
    
    print(f"Média de sentenças por review: {df['num_sentencas'].mean():.1f}")
    print(f"Mediana de sentenças por review: {df['num_sentencas'].median():.1f}")
    print(f"Mínimo: {df['num_sentencas'].min()}")
    print(f"Máximo: {df['num_sentencas'].max()}")
    
    # 4. Análise de palavras mais comuns
    print("\nANÁLISE DE PALAVRAS MAIS COMUNS:")
    print("-" * 40)
    
    def limpar_e_contar_palavras(texto):
        if pd.isna(texto):
            return []
        # Limpar texto e contar palavras
        texto_limpo = re.sub(r'[^a-zA-Z\s]', '', str(texto).lower())
        return texto_limpo.split()
    
    # Palavras gerais
    todas_palavras = []
    for texto in df['review']:
        todas_palavras.extend(limpar_e_contar_palavras(texto))
    
    contador_geral = Counter(todas_palavras)
    
    print("TOP 10 PALAVRAS MAIS COMUNS (GERAL):")
    for i, (palavra, count) in enumerate(contador_geral.most_common(10), 1):
        print(f"   {i:2d}. {palavra:15s}: {count:,}")
    
    # Palavras por sentimento
    palavras_positivas = []
    palavras_negativas = []
    
    for _, row in positivos.iterrows():
        palavras_positivas.extend(limpar_e_contar_palavras(row['review']))
    
    for _, row in negativos.iterrows():
        palavras_negativas.extend(limpar_e_contar_palavras(row['review']))
    
    contador_positivo = Counter(palavras_positivas)
    contador_negativo = Counter(palavras_negativas)
    
    print("\nTOP 5 PALAVRAS MAIS COMUNS (POSITIVOS):")
    for i, (palavra, count) in enumerate(contador_positivo.most_common(5), 1):
        print(f"   {i}. {palavra:15s}: {count:,}")
    
    print("\nTOP 5 PALAVRAS MAIS COMUNS (NEGATIVOS):")
    for i, (palavra, count) in enumerate(contador_negativo.most_common(5), 1):
        print(f"   {i}. {palavra:15s}: {count:,}")
    
    # 5. Visualizações adicionais
    print("\nGERANDO VISUALIZAÇÕES ADICIONAIS...")
    
    # Gráfico de distribuição de palavras por sentimento
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Distribuição de palavras
    plt.subplot(2, 3, 1)
    df.boxplot(column='num_palavras', by='sentiment', ax=plt.gca())
    plt.title('Distribuição de Palavras por Sentimento')
    plt.suptitle('')
    
    # Subplot 2: Distribuição de sentenças
    plt.subplot(2, 3, 2)
    df.boxplot(column='num_sentencas', by='sentiment', ax=plt.gca())
    plt.title('Distribuição de Sentenças por Sentimento')
    plt.suptitle('')
    
    # Subplot 3: Histograma de palavras
    plt.subplot(2, 3, 3)
    df['num_palavras'].hist(bins=50, alpha=0.7, color='lightblue')
    plt.title('Distribuição de Palavras por Review')
    plt.xlabel('Número de Palavras')
    plt.ylabel('Frequência')
    
    # Subplot 4: Scatter plot palavras vs sentenças
    plt.subplot(2, 3, 4)
    plt.scatter(df['num_palavras'], df['num_sentencas'], alpha=0.5, s=1)
    plt.xlabel('Número de Palavras')
    plt.ylabel('Número de Sentenças')
    plt.title('Correlação: Palavras vs Sentenças')
    
    # Subplot 5: Média de palavras por sentimento
    plt.subplot(2, 3, 5)
    medias_palavras = df.groupby('sentiment')['num_palavras'].mean()
    medias_palavras.plot(kind='bar', color=['#2E8B57', '#DC143C'])
    plt.title('Média de Palavras por Sentimento')
    plt.ylabel('Média de Palavras')
    plt.xticks(rotation=0)
    
    # Subplot 6: Média de sentenças por sentimento
    plt.subplot(2, 3, 6)
    medias_sentencas = df.groupby('sentiment')['num_sentencas'].mean()
    medias_sentencas.plot(kind='bar', color=['#2E8B57', '#DC143C'])
    plt.title('Média de Sentenças por Sentimento')
    plt.ylabel('Média de Sentenças')
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('estatisticas_detalhadas.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Resumo final
    print("\n" + "=" * 60)
    print("RESUMO DAS ESTATÍSTICAS DETALHADAS")
    print("=" * 60)
    
    print(f"\nESTATÍSTICAS GERAIS:")
    print(f"   • Total de reviews: {len(df):,}")
    print(f"   • Reviews únicos: {df['review'].nunique():,}")
    print(f"   • Reviews duplicados: {len(duplicados):,}")
    print(f"   • Total de palavras: {len(todas_palavras):,}")
    print(f"   • Vocabulário único: {len(set(todas_palavras)):,}")
    
    print(f"\nESTATÍSTICAS POR REVIEW:")
    print(f"   • Palavras (média): {df['num_palavras'].mean():.1f}")
    print(f"   • Palavras (mediana): {df['num_palavras'].median():.1f}")
    print(f"   • Sentenças (média): {df['num_sentencas'].mean():.1f}")
    print(f"   • Sentenças (mediana): {df['num_sentencas'].median():.1f}")
    print(f"   • Caracteres (média): {df['review'].str.len().mean():.0f}")
    print(f"   • Caracteres (mediana): {df['review'].str.len().median():.0f}")
    
    print(f"\nDISTRIBUIÇÃO POR SENTIMENTO:")
    print(f"   • Positivos: {len(positivos):,} ({len(positivos)/len(df)*100:.1f}%)")
    print(f"   • Negativos: {len(negativos):,} ({len(negativos)/len(df)*100:.1f}%)")
    
    print(f"\n✅ Arquivo gerado: estatisticas_detalhadas.png")

if __name__ == "__main__":
    analise_detalhada() 