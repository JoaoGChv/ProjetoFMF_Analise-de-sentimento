#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projeto FMF: Análise de Sentimentos IMDB - Fases M e C
Implementação das fases de Modelagem e Comunicação seguindo o artigo original
com melhorias para robustez e reprodutibilidade
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# Configurações para melhor visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def carregar_e_preparar_dados():
    """
    Fase G - Carregamento e preparação dos dados seguindo o artigo
    """
    print("=" * 60)
    print("FASE G - CARREGAMENTO E PREPARAÇÃO DOS DADOS")
    print("=" * 60)
    
    # Carregando o dataset original
    print("\nCARREGANDO DATASET ORIGINAL...")
    df_review = pd.read_csv('IMDB Dataset.csv')
    print(f"Dataset carregado: {len(df_review)} reviews")
    
    # Removendo duplicatas (melhoria sugerida)
    print("\nREMOVENDO DUPLICATAS...")
    df_review_clean = df_review.drop_duplicates(subset=['review'])
    print(f"Reviews únicos: {len(df_review_clean)} (removidos {len(df_review) - len(df_review_clean)} duplicatas)")
    
    # Criando amostra desbalanceada como no artigo
    print("\nCRIANDO AMOSTRA DESBALANCEADA (como no artigo)...")
    df_positive = df_review_clean[df_review_clean['sentiment']=='positive'][:9000]
    df_negative = df_review_clean[df_review_clean['sentiment']=='negative'][:1000]
    
    df_review_imb = pd.concat([df_positive, df_negative])
    print(f"Dataset desbalanceado criado: {len(df_review_imb)} reviews")
    print(f"   • Positivos: {len(df_positive)}")
    print(f"   • Negativos: {len(df_negative)}")
    
    # Verificando distribuição
    print("\nDISTRIBUIÇÃO DO DATASET DESBALANCEADO:")
    print(df_review_imb['sentiment'].value_counts())
    
    return df_review_imb

def balancear_dados(df_review_imb):
    """
    Balanceamento dos dados usando RandomUnderSampler (como no artigo)
    """
    print("\n" + "=" * 60)
    print("BALANCEAMENTO DOS DADOS")
    print("=" * 60)
    
    print("\n⚖️ APLICANDO RANDOMUNDERSAMPLER...")
    
    try:
        # Método do artigo
        rus = RandomUnderSampler(random_state=0)
        df_review_bal, df_review_bal['sentiment'] = rus.fit_resample(
            df_review_imb[['review']], df_review_imb['sentiment']
        )
        print("✅ RandomUnderSampler aplicado com sucesso!")
        
    except IndexError:
        # Método alternativo caso ocorra erro (como mencionado no artigo)
        print("⚠️ Erro no RandomUnderSampler, usando método alternativo...")
        length_negative = len(df_review_imb[df_review_imb['sentiment']=='negative'])
        df_review_positive = df_review_imb[df_review_imb['sentiment']=='positive'].sample(n=length_negative, random_state=0)
        df_review_non_positive = df_review_imb[~(df_review_imb['sentiment']=='positive')]
        
        df_review_bal = pd.concat([df_review_positive, df_review_non_positive])
        df_review_bal.reset_index(drop=True, inplace=True)
        print("✅ Método alternativo aplicado com sucesso!")
    
    print("\n📈 DISTRIBUIÇÃO APÓS BALANCEAMENTO:")
    print(df_review_bal['sentiment'].value_counts())
    
    return df_review_bal

def dividir_dados(df_review_bal):
    """
    Divisão dos dados em treino e teste (como no artigo)
    """
    print("\n" + "=" * 60)
    print("DIVISÃO DOS DADOS EM TREINO E TESTE")
    print("=" * 60)
    
    # Split como no artigo (33% para teste)
    print("\n✂️ DIVIDINDO DADOS (67% treino, 33% teste)...")
    train, test = train_test_split(df_review_bal, test_size=0.33, random_state=42, stratify=df_review_bal['sentiment'])
    
    # Definindo variáveis independentes e dependentes
    train_x, train_y = train['review'], train['sentiment']
    test_x, test_y = test['review'], test['sentiment']
    
    print(f"📊 Dados de treino: {len(train_x)} reviews")
    print(f"📊 Dados de teste: {len(test_x)} reviews")
    
    print("\n📈 DISTRIBUIÇÃO DOS DADOS DE TREINO:")
    print(train_y.value_counts())
    
    print("\n📈 DISTRIBUIÇÃO DOS DADOS DE TESTE:")
    print(test_y.value_counts())
    
    return train_x, train_y, test_x, test_y

def vetorizar_texto(train_x, test_x, metodo='count'):
    """
    Vetorização do texto usando CountVectorizer ou TfidfVectorizer
    """
    print("\n" + "=" * 60)
    print("VETORIZAÇÃO DO TEXTO")
    print("=" * 60)
    
    if metodo == 'count':
        print("\n🔤 APLICANDO COUNTVECTORIZER...")
        vectorizer = CountVectorizer(max_features=5000, stop_words='english')
        print("📝 CountVectorizer converte texto em matriz de contagem de palavras")
        print("   • max_features=5000: Limita a 5000 palavras mais frequentes")
        print("   • stop_words='english': Remove palavras comuns (the, and, etc.)")
        
    elif metodo == 'tfidf':
        print("\n🔤 APLICANDO TF-IDF VECTORIZER...")
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        print("📝 TF-IDF considera frequência e importância das palavras")
        print("   • Palavras raras recebem peso maior")
        print("   • Palavras muito comuns recebem peso menor")
    
    # Aplicando vetorização
    train_x_vectorized = vectorizer.fit_transform(train_x)
    test_x_vectorized = vectorizer.transform(test_x)
    
    print(f"\n📊 Matriz de treino: {train_x_vectorized.shape}")
    print(f"📊 Matriz de teste: {test_x_vectorized.shape}")
    print(f"📊 Vocabulário: {len(vectorizer.vocabulary_)} palavras")
    
    return vectorizer, train_x_vectorized, test_x_vectorized

def treinar_modelos(train_x_vectorized, train_y, test_x_vectorized, test_y):
    """
    Treinamento dos modelos de Machine Learning
    """
    print("\n" + "=" * 60)
    print("FASE M - MODELAGEM")
    print("=" * 60)
    
    # Definindo os modelos (como mencionados no artigo)
    modelos = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    resultados = {}
    
    print("\n🤖 TREINANDO MODELOS...")
    
    for nome, modelo in modelos.items():
        print(f"\n📈 Treinando {nome}...")
        
        # Treinamento
        modelo.fit(train_x_vectorized, train_y)
        
        # Previsões
        y_pred = modelo.predict(test_x_vectorized)
        
        # Métricas
        acuracia = accuracy_score(test_y, y_pred)
        f1 = f1_score(test_y, y_pred, pos_label='positive')
        
        resultados[nome] = {
            'modelo': modelo,
            'acuracia': acuracia,
            'f1_score': f1,
            'y_pred': y_pred
        }
        
        print(f"   ✅ Acurácia: {acuracia:.4f}")
        print(f"   ✅ F1-Score: {f1:.4f}")
    
    return resultados

def avaliar_modelos(resultados, test_y):
    """
    Avaliação detalhada dos modelos
    """
    print("\n" + "=" * 60)
    print("FASE C - COMUNICAÇÃO DOS RESULTADOS")
    print("=" * 60)
    
    # Tabela comparativa
    print("\n📊 TABELA COMPARATIVA DOS MODELOS:")
    print("-" * 60)
    print(f"{'Modelo':<20} {'Acurácia':<12} {'F1-Score':<12}")
    print("-" * 60)
    
    for nome, resultado in resultados.items():
        print(f"{nome:<20} {resultado['acuracia']:<12.4f} {resultado['f1_score']:<12.4f}")
    
    # Encontrando o melhor modelo
    melhor_modelo = max(resultados.items(), key=lambda x: x[1]['f1_score'])
    print(f"\n🏆 MELHOR MODELO: {melhor_modelo[0]}")
    print(f"   • Acurácia: {melhor_modelo[1]['acuracia']:.4f}")
    print(f"   • F1-Score: {melhor_modelo[1]['f1_score']:.4f}")
    
    # Relatório detalhado do melhor modelo
    print(f"\n📋 RELATÓRIO DETALHADO - {melhor_modelo[0]}:")
    print("-" * 40)
    print(classification_report(test_y, melhor_modelo[1]['y_pred']))
    
    # Matriz de confusão
    print(f"\n📊 MATRIZ DE CONFUSÃO - {melhor_modelo[0]}:")
    cm = confusion_matrix(test_y, melhor_modelo[1]['y_pred'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Matriz de Confusão - {melhor_modelo[0]}')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Previsto')
    plt.tight_layout()
    plt.savefig('matriz_confusao.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return melhor_modelo

def otimizar_modelo(vectorizer, train_x, train_y, test_x, test_y):
    """
    Otimização do melhor modelo usando GridSearchCV
    """
    print("\n" + "=" * 60)
    print("OTIMIZAÇÃO DO MODELO")
    print("=" * 60)
    
    print("\n🔧 APLICANDO GRIDSEARCHCV PARA OTIMIZAÇÃO...")
    
    # Vetorizando dados para otimização
    train_x_vectorized = vectorizer.fit_transform(train_x)
    test_x_vectorized = vectorizer.transform(test_x)
    
    # Definindo parâmetros para otimização
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    # GridSearchCV com Logistic Regression
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        param_grid,
        cv=5,
        scoring=lambda est, X, y: f1_score(y, est.predict(X), pos_label='positive'),
        n_jobs=-1
    )
    
    print("⏳ Executando GridSearchCV...")
    grid_search.fit(train_x_vectorized, train_y)
    
    print(f"\n🏆 MELHORES PARÂMETROS:")
    print(grid_search.best_params_)
    
    print(f"\n📊 MELHOR SCORE (CV): {grid_search.best_score_:.4f}")
    
    # Avaliando modelo otimizado
    y_pred_otimizado = grid_search.predict(test_x_vectorized)
    acuracia_otimizada = accuracy_score(test_y, y_pred_otimizado)
    f1_otimizado = f1_score(test_y, y_pred_otimizado, pos_label='positive')
    
    print(f"\n📈 RESULTADOS DO MODELO OTIMIZADO:")
    print(f"   • Acurácia: {acuracia_otimizada:.4f}")
    print(f"   • F1-Score: {f1_otimizado:.4f}")
    
    return grid_search, acuracia_otimizada, f1_otimizado

def salvar_resultados(resultados, melhor_modelo, acuracia_otimizada, f1_otimizado):
    """
    Salvando resultados em arquivo
    """
    print("\n" + "=" * 60)
    print("SALVANDO RESULTADOS")
    print("=" * 60)
    
    # Criando relatório
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
    
    print("✅ Resultados salvos em 'resultados_modelagem.txt'")

def main():
    """
    Função principal que executa todas as fases M e C
    """
    print("🚀 INICIANDO FASES M E C - PROJETO FMF")
    print("=" * 60)
    
    # Fase G - Carregamento e preparação
    df_review_imb = carregar_e_preparar_dados()
    
    # Balanceamento dos dados
    df_review_bal = balancear_dados(df_review_imb)
    
    # Divisão dos dados
    train_x, train_y, test_x, test_y = dividir_dados(df_review_bal)
    
    # Vetorização (usando CountVectorizer como no artigo)
    vectorizer, train_x_vectorized, test_x_vectorized = vetorizar_texto(train_x, test_x, metodo='count')
    
    # Fase M - Treinamento dos modelos
    resultados = treinar_modelos(train_x_vectorized, train_y, test_x_vectorized, test_y)
    
    # Fase C - Avaliação dos modelos
    melhor_modelo = avaliar_modelos(resultados, test_y)
    
    # Otimização do modelo
    modelo_otimizado, acuracia_otimizada, f1_otimizado = otimizar_modelo(
        vectorizer, train_x, train_y, test_x, test_y
    )
    
    # Salvando resultados
    salvar_resultados(resultados, melhor_modelo, acuracia_otimizada, f1_otimizado)
    
    print("\n" + "=" * 60)
    print("✅ FASES M E C CONCLUÍDAS COM SUCESSO!")
    print("=" * 60)
    
    print("\n📁 Arquivos gerados:")
    print("• matriz_confusao.png")
    print("• resultados_modelagem.txt")
    
    print("\n🎯 RESUMO FINAL:")
    print(f"• Melhor modelo: {melhor_modelo[0]}")
    print(f"• Acurácia: {melhor_modelo[1]['acuracia']:.4f}")
    print(f"• F1-Score: {melhor_modelo[1]['f1_score']:.4f}")
    print(f"• Modelo otimizado F1-Score: {f1_otimizado:.4f}")

if __name__ == "__main__":
    main() 