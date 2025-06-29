#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comunicação Avançada - Análise de Sentimentos IMDB
Implementação de visualizações profissionais e análise detalhada de erros
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Configurações para melhor visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def carregar_dados_e_modelo():
    """
    Carregamento dos dados e modelo treinado
    """
    print("=" * 60)
    print("CARREGAMENTO DOS DADOS E MODELO")
    print("=" * 60)
    
    # Carregando dados
    df = pd.read_csv('IMDB Dataset.csv')
    print(f"Dataset carregado: {len(df)} reviews")
    
    # Simulando um modelo treinado (em produção, carregaria o modelo salvo)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    # Pré-processamento básico
    def limpar_texto(texto):
        if pd.isna(texto):
            return ""
        texto = re.sub(r'[^a-zA-Z\s]', '', str(texto).lower())
        return texto
    
    df['review_limpo'] = df['review'].apply(limpar_texto)
    
    # Divisão dos dados
    X = df['review_limpo']
    y = df['sentiment']
    
    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Vetorização
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    train_x_vectorized = vectorizer.fit_transform(train_x)
    test_x_vectorized = vectorizer.transform(test_x)
    
    # Treinamento do modelo
    modelo = LogisticRegression(random_state=42, max_iter=1000)
    modelo.fit(train_x_vectorized, train_y)
    
    # Predições
    predicoes = modelo.predict(test_x_vectorized)
    
    print(f"Modelo treinado e predições geradas")
    
    return test_x, test_y, predicoes, modelo, vectorizer

def matriz_confusao_avancada(test_y, predicoes):
    """
    Matriz de confusão com visualização profissional
    """
    print("\n" + "=" * 60)
    print("MATRIZ DE CONFUSÃO AVANÇADA")
    print("=" * 60)
    
    # Calculando matriz de confusão
    cm = confusion_matrix(test_y, predicoes)
    
    # Criando visualização profissional
    plt.figure(figsize=(12, 10))
    
    # Heatmap principal
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Quantidade de Reviews'})
    
    plt.title('Matriz de Confusão - Análise de Sentimentos IMDB\nComunicação Avançada', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Sentimento Predito', fontsize=14, fontweight='bold')
    plt.ylabel('Sentimento Real', fontsize=14, fontweight='bold')
    
    # Adicionando anotações detalhadas
    tn, fp, fn, tp = cm.ravel()
    
    # Calculando métricas
    precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0
    acuracia = (tp + tn) / (tp + tn + fp + fn)
    
    # Texto com métricas
    texto_metricas = f"""
    MÉTRICAS DETALHADAS:
    
    Verdadeiros Negativos (TN): {tn}
    Falsos Positivos (FP): {fp}
    Falsos Negativos (FN): {fn}
    Verdadeiros Positivos (TP): {tp}
    
    Acurácia: {acuracia:.4f}
    Precisão: {precisao:.4f}
    Recall: {recall:.4f}
    Especificidade: {especificidade:.4f}
    """
    
    plt.figtext(0.02, 0.02, texto_metricas, fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('matriz_confusao_avancada.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Métricas detalhadas:")
    print(f"  Verdadeiros Negativos (TN): {tn}")
    print(f"  Falsos Positivos (FP): {fp}")
    print(f"  Falsos Negativos (FN): {fn}")
    print(f"  Verdadeiros Positivos (TP): {tp}")
    print(f"  Acurácia: {acuracia:.4f}")
    print(f"  Precisão: {precisao:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Especificidade: {especificidade:.4f}")
    
    return tn, fp, fn, tp

def analise_erros_detalhada(test_x, test_y, predicoes):
    """
    Análise detalhada dos erros do modelo
    """
    print("\n" + "=" * 60)
    print("ANÁLISE DE ERROS DO MODELO")
    print("=" * 60)
    
    # Criando DataFrame com resultados
    df_resultados = pd.DataFrame({
        'review': test_x,
        'sentimento_real': test_y,
        'sentimento_predito': predicoes
    })
    
    # Identificando erros
    df_resultados['erro'] = df_resultados['sentimento_real'] != df_resultados['sentimento_predito']
    df_resultados['tipo_erro'] = 'Correto'
    
    # Falsos positivos (negativo classificado como positivo)
    mask_fp = (df_resultados['sentimento_real'] == 'negative') & (df_resultados['sentimento_predito'] == 'positive')
    df_resultados.loc[mask_fp, 'tipo_erro'] = 'Falso Positivo'
    
    # Falsos negativos (positivo classificado como negativo)
    mask_fn = (df_resultados['sentimento_real'] == 'positive') & (df_resultados['sentimento_predito'] == 'negative')
    df_resultados.loc[mask_fn, 'tipo_erro'] = 'Falso Negativo'
    
    # Estatísticas de erros
    print(f"Estatísticas de erros:")
    print(f"  Total de reviews: {len(df_resultados)}")
    print(f"  Reviews corretos: {len(df_resultados[df_resultados['erro'] == False])}")
    print(f"  Reviews com erro: {len(df_resultados[df_resultados['erro'] == True])}")
    print(f"  Taxa de erro: {len(df_resultados[df_resultados['erro'] == True]) / len(df_resultados):.4f}")
    
    # Análise de falsos positivos
    falsos_positivos = df_resultados[df_resultados['tipo_erro'] == 'Falso Positivo']
    print(f"\nFalsos Positivos (negativo classificado como positivo): {len(falsos_positivos)}")
    
    if len(falsos_positivos) > 0:
        print("Exemplos de falsos positivos:")
        for i, (_, row) in enumerate(falsos_positivos.head(2).iterrows()):
            print(f"Exemplo {i+1}:")
            print(f"  Review: {row['review'][:200]}...")
            print("Análise do erro:")
            print("  Possível causa: Sarcasmo, linguagem ambígua ou negação complexa")
            print()
    
    # Análise de falsos negativos
    falsos_negativos = df_resultados[df_resultados['tipo_erro'] == 'Falso Negativo']
    print(f"Falsos Negativos (positivo classificado como negativo): {len(falsos_negativos)}")
    
    if len(falsos_negativos) > 0:
        print("Exemplos de falsos negativos:")
        for i, (_, row) in enumerate(falsos_negativos.head(2).iterrows()):
            print(f"Exemplo {i+1}:")
            print(f"  Review: {row['review'][:200]}...")
            print("Análise do erro:")
            print("  Possível causa: Palavras negativas em contexto positivo ou negação mal interpretada")
            print()
    
    return df_resultados

def analise_features_importancia(modelo, vectorizer):
    """
    Análise de importância das features
    """
    print("\n" + "=" * 60)
    print("ANÁLISE DE IMPORTÂNCIA DAS FEATURES")
    print("=" * 60)
    
    # Obtendo feature importances
    if hasattr(modelo, 'coef_'):
        importances = np.abs(modelo.coef_[0])
        feature_names = vectorizer.get_feature_names_out()
    else:
        print("Modelo não suporta análise de features")
        return
    
    # Top 20 features
    indices = np.argsort(importances)[::-1][:20]
    
    print(f"Top 20 features mais importantes:")
    print("-" * 40)
    for i, idx in enumerate(indices, 1):
        print(f"{i:2d}. {feature_names[idx]:<15} - {importances[idx]:.4f}")
    
    # Visualização
    plt.figure(figsize=(15, 10))
    
    # Top 10
    plt.subplot(2, 1, 1)
    top_10_indices = indices[:10]
    plt.barh(range(len(top_10_indices)), importances[top_10_indices])
    plt.yticks(range(len(top_10_indices)), [feature_names[i] for i in top_10_indices])
    plt.xlabel('Importância')
    plt.title('Top 10 Features Mais Importantes')
    plt.gca().invert_yaxis()
    
    # 11-20
    plt.subplot(2, 1, 2)
    top_11_20_indices = indices[10:20]
    plt.barh(range(len(top_11_20_indices)), importances[top_11_20_indices])
    plt.yticks(range(len(top_11_20_indices)), [feature_names[i] for i in top_11_20_indices])
    plt.xlabel('Importância')
    plt.title('Features 11-20 Mais Importantes')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('analise_features_importancia.png', dpi=300, bbox_inches='tight')
    plt.show()

def analise_frequencia_palavras(test_x, test_y):
    """
    Análise de frequência de palavras por classe
    """
    print("\n" + "=" * 60)
    print("ANÁLISE DE FREQUÊNCIA DE PALAVRAS")
    print("=" * 60)
    
    # Separando reviews por sentimento
    reviews_positivos = test_x[test_y == 'positive']
    reviews_negativos = test_x[test_y == 'negative']
    
    # Função para contar palavras
    def contar_palavras(reviews):
        todas_palavras = []
        for review in reviews:
            palavras = review.lower().split()
            todas_palavras.extend(palavras)
        return Counter(todas_palavras)
    
    # Contando palavras
    palavras_positivas = contar_palavras(reviews_positivos)
    palavras_negativas = contar_palavras(reviews_negativos)
    
    # Top 15 palavras por classe
    top_positivas = palavras_positivas.most_common(15)
    top_negativas = palavras_negativas.most_common(15)
    
    print(f"Top 15 palavras em reviews positivos:")
    print("-" * 40)
    for palavra, count in top_positivas:
        print(f"{palavra:<15} - {count}")
    
    print(f"\nTop 15 palavras em reviews negativos:")
    print("-" * 40)
    for palavra, count in top_negativas:
        print(f"{palavra:<15} - {count}")
    
    # Visualização
    plt.figure(figsize=(15, 10))
    
    # Reviews positivos
    plt.subplot(2, 1, 1)
    palavras, contagens = zip(*top_positivas)
    plt.barh(range(len(palavras)), contagens, color='green', alpha=0.7)
    plt.yticks(range(len(palavras)), palavras)
    plt.xlabel('Frequência')
    plt.title('Palavras Mais Frequentes - Reviews Positivos')
    plt.gca().invert_yaxis()
    
    # Reviews negativos
    plt.subplot(2, 1, 2)
    palavras, contagens = zip(*top_negativas)
    plt.barh(range(len(palavras)), contagens, color='red', alpha=0.7)
    plt.yticks(range(len(palavras)), palavras)
    plt.xlabel('Frequência')
    plt.title('Palavras Mais Frequentes - Reviews Negativos')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('frequencia_palavras_por_classe.png', dpi=300, bbox_inches='tight')
    plt.show()

def salvar_relatorio(df_resultados, tn, fp, fn, tp):
    """
    Salvando relatório de comunicação avançada
    """
    print("\n" + "=" * 60)
    print("SALVANDO RELATÓRIO")
    print("=" * 60)
    
    with open('relatorio_comunicacao_avancada.txt', 'w', encoding='utf-8') as f:
        f.write("RELATÓRIO DE COMUNICAÇÃO AVANÇADA - ANÁLISE DE SENTIMENTOS IMDB\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("MATRIZ DE CONFUSÃO:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Verdadeiros Negativos (TN): {tn}\n")
        f.write(f"Falsos Positivos (FP): {fp}\n")
        f.write(f"Falsos Negativos (FN): {fn}\n")
        f.write(f"Verdadeiros Positivos (TP): {tp}\n\n")
        
        f.write("ANÁLISE DE ERROS:\n")
        f.write("-" * 18 + "\n")
        f.write(f"Total de reviews: {len(df_resultados)}\n")
        f.write(f"Reviews corretos: {len(df_resultados[df_resultados['erro'] == False])}\n")
        f.write(f"Reviews com erro: {len(df_resultados[df_resultados['erro'] == True])}\n")
        f.write(f"Taxa de erro: {len(df_resultados[df_resultados['erro'] == True]) / len(df_resultados):.4f}\n\n")
        
        f.write("TIPOS DE ERRO:\n")
        f.write("-" * 13 + "\n")
        f.write(f"Falsos Positivos: {len(df_resultados[df_resultados['tipo_erro'] == 'Falso Positivo'])}\n")
        f.write(f"Falsos Negativos: {len(df_resultados[df_resultados['tipo_erro'] == 'Falso Negativo'])}\n\n")
        
        f.write("INSIGHTS:\n")
        f.write("-" * 8 + "\n")
        f.write("• Sarcasmo e ironia são desafios para o modelo\n")
        f.write("• Negação complexa pode causar erros\n")
        f.write("• Contexto ambíguo requer análise mais sofisticada\n")
        f.write("• Palavras negativas em contexto positivo podem confundir o modelo\n")
    
    print("Relatório salvo em 'relatorio_comunicacao_avancada.txt'")

def main():
    """
    Função principal
    """
    print("INICIANDO COMUNICAÇÃO AVANÇADA")
    print("=" * 60)
    
    # Carregamento dos dados e modelo
    test_x, test_y, predicoes, modelo, vectorizer = carregar_dados_e_modelo()
    
    # Matriz de confusão avançada
    tn, fp, fn, tp = matriz_confusao_avancada(test_y, predicoes)
    
    # Análise de erros
    df_resultados = analise_erros_detalhada(test_x, test_y, predicoes)
    
    # Análise de features
    analise_features_importancia(modelo, vectorizer)
    
    # Análise de frequência de palavras
    analise_frequencia_palavras(test_x, test_y)
    
    # Salvando relatório
    salvar_relatorio(df_resultados, tn, fp, fn, tp)
    
    print("\n" + "=" * 60)
    print("COMUNICAÇÃO AVANÇADA CONCLUÍDA")
    print("=" * 60)
    
    print("\nArquivos gerados:")
    print("• matriz_confusao_avancada.png")
    print("• analise_features_importancia.png")
    print("• frequencia_palavras_por_classe.png")
    print("• relatorio_comunicacao_avancada.txt")

if __name__ == "__main__":
    main() 