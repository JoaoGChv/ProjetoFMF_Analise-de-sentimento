#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projeto ORIGINAL MELHORADO: Análise de Sentimentos IMDB
Implementação com 8 originalidades inovadoras incluindo validação estatística
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download de recursos NLTK (se necessário)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Configurações para melhor visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def carregar_dados():
    """
    Carregamento dos dados
    """
    print("=" * 60)
    print("CARREGAMENTO DOS DADOS")
    print("=" * 60)
    
    df = pd.read_csv('IMDB Dataset.csv')
    print(f"Dataset carregado: {len(df)} reviews")
    print(f"Distribuição: {df['sentiment'].value_counts().to_dict()}")
    
    return df

def preprocessamento_avancado(df):
    """
    Pipeline avançado de pré-processamento com lematização
    """
    print("\n" + "=" * 60)
    print("PRÉ-PROCESSAMENTO AVANÇADO")
    print("=" * 60)
    
    # Inicializando lematizador
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def limpar_texto(texto):
        if pd.isna(texto):
            return ""
        
        # Converter para minúsculas
        texto = str(texto).lower()
        
        # Remover números
        texto = re.sub(r'\d+', '', texto)
        
        # Remover pontuação
        texto = re.sub(r'[^\w\s]', '', texto)
        
        # Tokenização
        tokens = texto.split()
        
        # Remover stopwords e palavras < 3 letras
        tokens = [token for token in tokens if token not in stop_words and len(token) >= 3]
        
        # Lematização
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    # Aplicando pré-processamento
    df['review_limpo'] = df['review'].apply(limpar_texto)
    
    print("Pré-processamento concluído (lematização, limpeza, etc.)")
    
    return df

def preparar_modelagem(df):
    """
    Preparação para modelagem
    """
    print("\n" + "=" * 60)
    print("PREPARAÇÃO PARA MODELAGEM")
    print("=" * 60)
    
    # Divisão dos dados
    X = df['review_limpo']
    y = df['sentiment']
    
    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Divisão dos dados:")
    print(f"  Treino: {len(train_x)} reviews")
    print(f"  Teste: {len(test_x)} reviews")
    
    # Vetorização TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    train_x_vectorized = vectorizer.fit_transform(train_x)
    test_x_vectorized = vectorizer.transform(test_x)
    
    print(f"Vetorização:")
    print(f"  Vocabulário: {len(vectorizer.vocabulary_)} palavras")
    print(f"  Matriz treino: {train_x_vectorized.shape}")
    print(f"  Matriz teste: {test_x_vectorized.shape}")
    
    return train_x_vectorized, test_x_vectorized, train_y, test_y, vectorizer

def modelagem_com_cross_validation(train_x, test_x, train_y, test_y):
    """
    Modelagem com cross-validation robusta
    """
    print("\n" + "=" * 60)
    print("MODELAGEM COM CROSS-VALIDATION ROBUSTA")
    print("=" * 60)
    
    # Definindo modelos
    modelos = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    resultados = {}
    cv_scores_acc = {}
    cv_scores_f1 = {}
    
    # Cross-validation robusta
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for nome, modelo in modelos.items():
        print(f"\n{nome}:")
        
        # Cross-validation
        cv_acc_scores = []
        cv_f1_scores = []
        
        for train_idx, val_idx in cv.split(train_x, train_y):
            X_train_fold = train_x[train_idx]
            y_train_fold = train_y.iloc[train_idx]
            X_val_fold = train_x[val_idx]
            y_val_fold = train_y.iloc[val_idx]
            
            modelo.fit(X_train_fold, y_train_fold)
            pred_fold = modelo.predict(X_val_fold)
            
            cv_acc_scores.append(accuracy_score(y_val_fold, pred_fold))
            cv_f1_scores.append(f1_score(y_val_fold, pred_fold, pos_label='positive'))
        
        # Treinamento no conjunto completo
        modelo.fit(train_x, train_y)
        pred_teste = modelo.predict(test_x)
        
        # Métricas
        acc_teste = accuracy_score(test_y, pred_teste)
        f1_teste = f1_score(test_y, pred_teste, pos_label='positive')
        
        # Armazenando resultados
        resultados[nome] = {
            'modelo': modelo,
            'acc_teste': acc_teste,
            'f1_teste': f1_teste,
            'predicoes': pred_teste
        }
        
        cv_scores_acc[nome] = np.array(cv_acc_scores)
        cv_scores_f1[nome] = np.array(cv_f1_scores)
        
        print(f"   CV Acurácia: {cv_scores_acc[nome].mean():.4f} ± {cv_scores_acc[nome].std():.4f}")
        print(f"   CV F1-Score: {cv_scores_f1[nome].mean():.4f} ± {cv_scores_f1[nome].std():.4f}")
        print(f"   Teste Acurácia: {acc_teste:.4f}")
        print(f"   Teste F1-Score: {f1_teste:.4f}")
    
    return resultados, cv_scores_acc, cv_scores_f1

def validacao_estatistica(resultados, cv_scores_acc, cv_scores_f1):
    """
    Validação estatística com testes t
    """
    print("\n" + "=" * 60)
    print("VALIDAÇÃO ESTATÍSTICA")
    print("=" * 60)
    
    # Encontrando o melhor modelo
    melhor_modelo = max(resultados.items(), key=lambda x: x[1]['f1_teste'])
    melhor_nome = melhor_modelo[0]
    
    print(f"Melhor modelo: {melhor_nome}")
    print(f"Acurácia: {melhor_modelo[1]['acc_teste']:.4f}")
    print(f"F1-Score: {melhor_modelo[1]['f1_teste']:.4f}")
    
    # Testes t para comparar com outros modelos
    print(f"\nTestes t (comparando {melhor_nome} com outros modelos):")
    print("-" * 50)
    
    significancia = {}
    
    for nome, resultado in resultados.items():
        if nome != melhor_nome:
            # Teste t para acurácia
            t_stat_acc, p_value_acc = stats.ttest_ind(
                cv_scores_acc[melhor_nome], cv_scores_acc[nome]
            )
            
            # Teste t para F1-Score
            t_stat_f1, p_value_f1 = stats.ttest_ind(
                cv_scores_f1[melhor_nome], cv_scores_f1[nome]
            )
            
            significancia[nome] = {
                'acc_p_value': p_value_acc,
                'f1_p_value': p_value_f1,
                'acc_significativo': p_value_acc < 0.05,
                'f1_significativo': p_value_f1 < 0.05
            }
            
            print(f"{melhor_nome} vs {nome}:")
            print(f"   Acurácia: p-value = {p_value_acc:.4f} {'(significativo)' if p_value_acc < 0.05 else '(não significativo)'}")
            print(f"   F1-Score: p-value = {p_value_f1:.4f} {'(significativo)' if p_value_f1 < 0.05 else '(não significativo)'}")
    
    return melhor_modelo, significancia

def otimizar_melhor_modelo(melhor_modelo, train_x, test_x, train_y, test_y):
    """
    Otimização de hiperparâmetros do melhor modelo
    """
    print("\n" + "=" * 60)
    print("OTIMIZAÇÃO DE HIPERPARÂMETROS")
    print("=" * 60)
    
    nome, resultado = melhor_modelo
    modelo_base = resultado['modelo']
    
    # Definindo parâmetros para otimização
    if isinstance(modelo_base, SVC):
        param_grid = {
            'C': [1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    elif isinstance(modelo_base, LogisticRegression):
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    elif isinstance(modelo_base, RandomForestClassifier):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
    else:
        print("Modelo não suporta otimização automática")
        return melhor_modelo
    
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
    
    acc_otimizada = accuracy_score(test_y, predicoes_otimizadas)
    f1_otimizado = f1_score(test_y, predicoes_otimizadas, pos_label='positive')
    
    print(f"\nResultados do modelo otimizado:")
    print(f"Acurácia: {acc_otimizada:.4f}")
    print(f"F1-Score: {f1_otimizado:.4f}")
    
    # Criando novo resultado otimizado
    resultado_otimizado = {
        'modelo': modelo_otimizado,
        'acc_teste': acc_otimizada,
        'f1_teste': f1_otimizado,
        'predicoes': predicoes_otimizadas
    }
    
    return (f"{nome} (Otimizado)", resultado_otimizado)

def analise_features(melhor_modelo, vectorizer):
    """
    Análise de features para interpretabilidade
    """
    print("\n" + "=" * 60)
    print("ANÁLISE DE FEATURES")
    print("=" * 60)
    
    nome, resultado = melhor_modelo
    modelo = resultado['modelo']
    
    # Obtendo feature importances
    if hasattr(modelo, 'feature_importances_'):
        # Para Random Forest
        importances = modelo.feature_importances_
        feature_names = vectorizer.get_feature_names_out()
    elif hasattr(modelo, 'coef_'):
        # Para Logistic Regression
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
    plt.title(f'Top 10 Features Mais Importantes - {nome}')
    plt.gca().invert_yaxis()
    
    # 11-20
    plt.subplot(2, 1, 2)
    top_11_20_indices = indices[10:20]
    plt.barh(range(len(top_11_20_indices)), importances[top_11_20_indices])
    plt.yticks(range(len(top_11_20_indices)), [feature_names[i] for i in top_11_20_indices])
    plt.xlabel('Importância')
    plt.title(f'Features 11-20 Mais Importantes - {nome}')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('analise_features_importancia.png', dpi=300, bbox_inches='tight')
    plt.show()

def comunicacao_avancada(melhor_modelo, test_y):
    """
    Comunicação avançada com visualizações profissionais
    """
    print("\n" + "=" * 60)
    print("COMUNICAÇÃO AVANÇADA")
    print("=" * 60)
    
    nome, resultado = melhor_modelo
    predicoes = resultado['predicoes']
    
    # Matriz de confusão profissional
    cm = confusion_matrix(test_y, predicoes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Matriz de Confusão - {nome}', fontsize=16, fontweight='bold')
    plt.ylabel('Valor Real', fontsize=12)
    plt.xlabel('Valor Predito', fontsize=12)
    
    # Adicionando métricas
    tn, fp, fn, tp = cm.ravel()
    precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    stats_text = f"""
    Métricas Detalhadas:
    
    Verdadeiros Negativos (TN): {tn}
    Falsos Positivos (FP): {fp}
    Falsos Negativos (FN): {fn}
    Verdadeiros Positivos (TP): {tp}
    
    Precisão: {precisao:.2%}
    Recall: {recall:.2%}
    Especificidade: {especificidade:.2%}
    """
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('matriz_confusao_avancada.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Métricas do modelo {nome}:")
    print(f"  Verdadeiros Negativos: {tn}")
    print(f"  Falsos Positivos: {fp}")
    print(f"  Falsos Negativos: {fn}")
    print(f"  Verdadeiros Positivos: {tp}")
    print(f"  Precisão: {precisao:.2%}")
    print(f"  Recall: {recall:.2%}")
    print(f"  Especificidade: {especificidade:.2%}")

def salvar_resultados(resultados, melhor_modelo):
    """
    Salvando resultados em arquivo
    """
    print("\n" + "=" * 60)
    print("SALVANDO RESULTADOS")
    print("=" * 60)
    
    with open('resultados_projeto_original.txt', 'w', encoding='utf-8') as f:
        f.write("RELATÓRIO PROJETO ORIGINAL - ANÁLISE DE SENTIMENTOS IMDB\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("ORIGINALIDADES IMPLEMENTADAS:\n")
        f.write("-" * 30 + "\n")
        f.write("1. Pipeline avançado de pré-processamento\n")
        f.write("2. Novos modelos (SVM e Random Forest)\n")
        f.write("3. Otimização de hiperparâmetros\n")
        f.write("4. Validação estatística\n")
        f.write("5. Cross-validation robusta\n")
        f.write("6. Análise de features\n")
        f.write("7. Visualização profissional\n")
        f.write("8. Análise detalhada de erros\n\n")
        
        f.write("COMPARAÇÃO DE TODOS OS MODELOS:\n")
        f.write("-" * 35 + "\n")
        for nome, resultado in resultados.items():
            f.write(f"{nome}: Acurácia={resultado['acc_teste']:.4f}, F1={resultado['f1_teste']:.4f}\n")
        
        f.write(f"\nCAMPEÃO GERAL: {melhor_modelo[0]}\n")
        f.write(f"Acurácia: {melhor_modelo[1]['acc_teste']:.4f}\n")
        f.write(f"F1-Score: {melhor_modelo[1]['f1_teste']:.4f}\n\n")
        
        f.write("PRÉ-PROCESSAMENTO AVANÇADO:\n")
        f.write("-" * 30 + "\n")
        f.write("• Conversão para minúsculas\n")
        f.write("• Remoção de números\n")
        f.write("• Remoção de pontuação\n")
        f.write("• Remoção de stopwords\n")
        f.write("• Remoção de palavras < 3 letras\n")
        f.write("• Lemmatização\n")
    
    print("Resultados salvos em 'resultados_projeto_original.txt'")

def main():
    """
    Função principal
    """
    print("INICIANDO PROJETO ORIGINAL MELHORADO")
    print("=" * 60)
    
    # Carregamento dos dados
    df = carregar_dados()
    
    # Pré-processamento avançado
    df = preprocessamento_avancado(df)
    
    # Preparação para modelagem
    train_x, test_x, train_y, test_y, vectorizer = preparar_modelagem(df)
    
    # Modelagem com cross-validation
    resultados, cv_scores_acc, cv_scores_f1 = modelagem_com_cross_validation(
        train_x, test_x, train_y, test_y
    )
    
    # Validação estatística
    melhor_modelo, significancia = validacao_estatistica(resultados, cv_scores_acc, cv_scores_f1)
    
    # Otimização do melhor modelo
    melhor_modelo_otimizado = otimizar_melhor_modelo(
        melhor_modelo, train_x, test_x, train_y, test_y
    )
    
    # Adicionando modelo otimizado aos resultados
    resultados[melhor_modelo_otimizado[0]] = melhor_modelo_otimizado[1]
    
    # Análise de features
    analise_features(melhor_modelo_otimizado, vectorizer)
    
    # Comunicação avançada
    comunicacao_avancada(melhor_modelo_otimizado, test_y)
    
    # Salvando resultados
    salvar_resultados(resultados, melhor_modelo_otimizado)
    
    print("\n" + "=" * 60)
    print("PROJETO ORIGINAL MELHORADO CONCLUÍDO")
    print("=" * 60)
    
    print("\nArquivos gerados:")
    print("• resultados_projeto_original.txt")
    print("• matriz_confusao_avancada.png")
    print("• analise_features_importancia.png")
    
    print(f"\nMelhor modelo: {melhor_modelo_otimizado[0]}")
    print(f"Acurácia: {melhor_modelo_otimizado[1]['acc_teste']:.4f}")
    print(f"F1-Score: {melhor_modelo_otimizado[1]['f1_teste']:.4f}")

if __name__ == "__main__":
    main() 