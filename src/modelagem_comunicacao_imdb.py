#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projeto FMF: Modelagem e Comunicação - Fases M e C
Implementação dos modelos de machine learning e comunicação dos resultados
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# Configurações para melhor visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def carregar_e_preparar_dados():
    """
    Carregamento e preparação dos dados para modelagem
    """
    print("=" * 60)
    print("CARREGAMENTO E PREPARAÇÃO DOS DADOS")
    print("=" * 60)
    
    # Carregando o dataset
    df = pd.read_csv('IMDB Dataset.csv')
    print(f"Dataset carregado: {len(df)} reviews")
    
    # Verificando balanceamento
    print(f"\nDistribuição original:")
    print(df['sentiment'].value_counts())
    
    # Balanceamento dos dados (se necessário)
    positive_reviews = df[df['sentiment'] == 'positive']
    negative_reviews = df[df['sentiment'] == 'negative']
    
    # Usando a classe menor para balanceamento
    min_class_size = min(len(positive_reviews), len(negative_reviews))
    
    try:
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=42)
        
        # Preparando dados para o sampler
        X = df[['review']]
        y = df['sentiment']
        
        # Aplicando undersampling
        X_resampled, y_resampled = rus.fit_resample(X, y)
        df_balanced = pd.DataFrame({'review': X_resampled['review'], 'sentiment': y_resampled})
        
        print("RandomUnderSampler aplicado com sucesso")
    except:
        print("Erro no RandomUnderSampler, usando método alternativo...")
        
        # Método alternativo: amostragem manual
        positive_sample = positive_reviews.sample(n=min_class_size, random_state=42)
        negative_sample = negative_reviews.sample(n=min_class_size, random_state=42)
        df_balanced = pd.concat([positive_sample, negative_sample]).reset_index(drop=True)
        
        print("Método alternativo aplicado com sucesso")
    
    print(f"\nDistribuição após balanceamento:")
    print(df_balanced['sentiment'].value_counts())
    
    return df_balanced

def dividir_dados(df):
    """
    Divisão dos dados em treino e teste
    """
    print("\n" + "=" * 60)
    print("DIVISÃO DOS DADOS EM TREINO E TESTE")
    print("=" * 60)
    
    # Divisão dos dados
    X = df['review']
    y = df['sentiment']
    
    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dados de treino: {len(train_x)} reviews")
    print(f"Dados de teste: {len(test_x)} reviews")
    
    print(f"\nDistribuição dos dados de treino:")
    print(train_y.value_counts())
    
    print(f"\nDistribuição dos dados de teste:")
    print(test_y.value_counts())
    
    return train_x, test_x, train_y, test_y

def vetorizar_texto(train_x, test_x):
    """
    Vetorização do texto usando CountVectorizer e TF-IDF
    """
    print("\n" + "=" * 60)
    print("VETORIZAÇÃO DO TEXTO")
    print("=" * 60)
    
    # CountVectorizer
    print("\nAplicando CountVectorizer...")
    count_vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    print("CountVectorizer converte texto em matriz de contagem de palavras")
    
    # TF-IDF Vectorizer
    print("\nAplicando TF-IDF Vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    print("TF-IDF considera frequência e importância das palavras")
    
    # Aplicando vetorização
    train_x_count = count_vectorizer.fit_transform(train_x)
    test_x_count = count_vectorizer.transform(test_x)
    
    train_x_tfidf = tfidf_vectorizer.fit_transform(train_x)
    test_x_tfidf = tfidf_vectorizer.transform(test_x)
    
    print(f"\nMatriz de treino: {train_x_count.shape}")
    print(f"Matriz de teste: {test_x_count.shape}")
    print(f"Vocabulário: {len(count_vectorizer.vocabulary_)} palavras")
    
    return train_x_count, test_x_count, train_x_tfidf, test_x_tfidf

def treinar_modelos(train_x, test_x, train_y, test_y):
    """
    Treinamento dos modelos de machine learning
    """
    print("\n" + "=" * 60)
    print("TREINAMENTO DOS MODELOS")
    print("=" * 60)
    
    # Definindo os modelos
    modelos = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    resultados = {}
    
    print("\nTreinando modelos...")
    
    for nome, modelo in modelos.items():
        print(f"\nTreinando {nome}...")
        
        # Treinando o modelo
        modelo.fit(train_x, train_y)
        
        # Fazendo predições
        predicoes = modelo.predict(test_x)
        
        # Calculando métricas
        acuracia = accuracy_score(test_y, predicoes)
        f1 = f1_score(test_y, predicoes, pos_label='positive')
        
        resultados[nome] = {
            'modelo': modelo,
            'acuracia': acuracia,
            'f1_score': f1,
            'predicoes': predicoes
        }
        
        print(f"   Acurácia: {acuracia:.4f}")
        print(f"   F1-Score: {f1:.4f}")
    
    return resultados

def comparar_modelos(resultados):
    """
    Comparação dos modelos treinados
    """
    print("\n" + "=" * 60)
    print("COMPARAÇÃO DOS MODELOS")
    print("=" * 60)
    
    print("\nTabela comparativa dos modelos:")
    print("-" * 50)
    print(f"{'Modelo':<20} {'Acurácia':<12} {'F1-Score':<12}")
    print("-" * 50)
    
    melhor_modelo = None
    melhor_f1 = 0
    
    for nome, resultado in resultados.items():
        print(f"{nome:<20} {resultado['acuracia']:<12.4f} {resultado['f1_score']:<12.4f}")
        
        if resultado['f1_score'] > melhor_f1:
            melhor_f1 = resultado['f1_score']
            melhor_modelo = (nome, resultado)
    
    print("-" * 50)
    print(f"Melhor modelo: {melhor_modelo[0]}")
    
    return melhor_modelo

def relatorio_detalhado(melhor_modelo, test_y):
    """
    Relatório detalhado do melhor modelo
    """
    print("\n" + "=" * 60)
    print(f"RELATÓRIO DETALHADO - {melhor_modelo[0]}")
    print("=" * 60)
    
    nome, resultado = melhor_modelo
    modelo = resultado['modelo']
    predicoes = resultado['predicoes']
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(test_y, predicoes))
    
    # Matriz de Confusão
    print("\nMatriz de Confusão:")
    cm = confusion_matrix(test_y, predicoes)
    print(cm)
    
    # Visualização da matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Matriz de Confusão - {nome}')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predito')
    plt.tight_layout()
    plt.savefig('matriz_confusao.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return modelo

def otimizar_hiperparametros(melhor_modelo, train_x, test_x, train_y, test_y):
    """
    Otimização de hiperparâmetros do melhor modelo
    """
    print("\n" + "=" * 60)
    print("OTIMIZAÇÃO DE HIPERPARÂMETROS")
    print("=" * 60)
    
    nome, resultado = melhor_modelo
    modelo_base = resultado['modelo']
    
    # Definindo parâmetros para otimização
    if isinstance(modelo_base, LogisticRegression):
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    elif isinstance(modelo_base, MultinomialNB):
        param_grid = {
            'alpha': [0.1, 0.5, 1.0, 2.0]
        }
    elif isinstance(modelo_base, DecisionTreeClassifier):
        param_grid = {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    else:
        print("Modelo não suporta otimização automática")
        return modelo_base
    
    # Grid Search
    grid_search = GridSearchCV(
        modelo_base, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(train_x, train_y)
    
    # Resultados da otimização
    print(f"\nMelhores parâmetros:")
    print(grid_search.best_params_)
    
    print(f"\nMelhor score (CV): {grid_search.best_score_:.4f}")
    
    # Testando o modelo otimizado
    modelo_otimizado = grid_search.best_estimator_
    predicoes_otimizadas = modelo_otimizado.predict(test_x)
    
    acuracia_otimizada = accuracy_score(test_y, predicoes_otimizadas)
    f1_otimizado = f1_score(test_y, predicoes_otimizadas, pos_label='positive')
    
    print(f"\nResultados do modelo otimizado:")
    print(f"Acurácia: {acuracia_otimizada:.4f}")
    print(f"F1-Score: {f1_otimizado:.4f}")
    
    return modelo_otimizado, acuracia_otimizada, f1_otimizado

def salvar_resultados(resultados, melhor_modelo, acuracia_otimizada, f1_otimizado):
    """
    Salvando os resultados em arquivo
    """
    print("\n" + "=" * 60)
    print("SALVANDO RESULTADOS")
    print("=" * 60)
    
    with open('resultados_modelagem.txt', 'w', encoding='utf-8') as f:
        f.write("RELATÓRIO DE MODELAGEM - PROJETO FMF\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("COMPARAÇÃO DOS MODELOS:\n")
        f.write("-" * 30 + "\n")
        for nome, resultado in resultados.items():
            f.write(f"{nome}: Acurácia={resultado['acuracia']:.4f}, F1={resultado['f1_score']:.4f}\n")
        
        f.write(f"\nMELHOR MODELO: {melhor_modelo[0]}\n")
        f.write(f"Acurácia: {melhor_modelo[1]['acuracia']:.4f}\n")
        f.write(f"F1-Score: {melhor_modelo[1]['f1_score']:.4f}\n")
        
        f.write(f"\nMODELO OTIMIZADO:\n")
        f.write(f"Acurácia: {acuracia_otimizada:.4f}\n")
        f.write(f"F1-Score: {f1_otimizado:.4f}\n")
    
    print("Resultados salvos em 'resultados_modelagem.txt'")

def main():
    """
    Função principal que executa todo o pipeline de modelagem
    """
    print("INICIANDO FASES M E C - PROJETO FMF")
    print("=" * 60)
    
    # Carregamento e preparação dos dados
    df = carregar_e_preparar_dados()
    
    # Divisão dos dados
    train_x, test_x, train_y, test_y = dividir_dados(df)
    
    # Vetorização
    train_x_count, test_x_count, train_x_tfidf, test_x_tfidf = vetorizar_texto(train_x, test_x)
    
    # Usando TF-IDF para treinamento (geralmente melhor para análise de sentimentos)
    train_x_vectorized = train_x_tfidf
    test_x_vectorized = test_x_tfidf
    
    # Treinamento dos modelos
    resultados = treinar_modelos(train_x_vectorized, test_x_vectorized, train_y, test_y)
    
    # Comparação dos modelos
    melhor_modelo = comparar_modelos(resultados)
    
    # Relatório detalhado
    modelo_final = relatorio_detalhado(melhor_modelo, test_y)
    
    # Otimização de hiperparâmetros
    modelo_otimizado, acuracia_otimizada, f1_otimizado = otimizar_hiperparametros(
        melhor_modelo, train_x_vectorized, test_x_vectorized, train_y, test_y
    )
    
    # Salvando resultados
    salvar_resultados(resultados, melhor_modelo, acuracia_otimizada, f1_otimizado)
    
    print("\n" + "=" * 60)
    print("FASES M E C CONCLUÍDAS")
    print("=" * 60)
    
    print("\nArquivos gerados:")
    print("• resultados_modelagem.txt")
    print("• matriz_confusao.png")
    
    print("\nResumo final:")
    print(f"• Melhor modelo: {melhor_modelo[0]}")
    print(f"• Acurácia: {melhor_modelo[1]['acuracia']:.4f}")
    print(f"• F1-Score: {melhor_modelo[1]['f1_score']:.4f}")

if __name__ == "__main__":
    main() 