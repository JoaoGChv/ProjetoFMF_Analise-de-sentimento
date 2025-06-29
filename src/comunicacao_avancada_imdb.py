#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projeto ORIGINAL: Comunicação Avançada - Fase C
Implementação com originalidades na comunicação:
1. Visualização da Matriz de Confusão com heatmap
2. Análise de Erros do Modelo (Falsos Positivos/Negativos)
3. Visualização de Importância das Palavras
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configurações para visualizações profissionais
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

def carregar_modelo_e_dados():
    """
    Carrega o modelo otimizado e dados necessários
    """
    print("=" * 60)
    print("CARREGANDO MODELO OTIMIZADO E DADOS")
    print("=" * 60)
    
    # Importando o modelo treinado do projeto original
    from projeto_original_imdb import (
        carregar_e_preparar_dados, balancear_dados, dividir_dados, 
        vetorizar_texto, treinar_modelos_fmf, treinar_novos_modelos,
        encontrar_melhor_modelo, otimizar_hiperparametros
    )
    
    print("\nCARREGANDO DADOS...")
    df_review_imb = carregar_e_preparar_dados()
    df_review_bal = balancear_dados(df_review_imb)
    train_x, train_y, test_x, test_y = dividir_dados(df_review_bal)
    
    print("\nVETORIZANDO DADOS...")
    vectorizer, train_x_vectorized, test_x_vectorized = vetorizar_texto(train_x, test_x, metodo='count')
    
    print("\nTREINANDO MODELOS...")
    resultados_fmf = treinar_modelos_fmf(train_x_vectorized, train_y, test_x_vectorized, test_y)
    resultados_novos = treinar_novos_modelos(train_x_vectorized, train_y, test_x_vectorized, test_y)
    
    print("\nIDENTIFICANDO MELHOR MODELO...")
    melhor_modelo, todos_resultados = encontrar_melhor_modelo(resultados_fmf, resultados_novos)
    
    print("\nOTIMIZANDO HIPERPARÂMETROS...")
    modelo_otimizado, acuracia_otimizada, f1_otimizado = otimizar_hiperparametros(
        melhor_modelo, vectorizer, train_x, train_y, test_x, test_y
    )
    
    # Fazendo previsões com o modelo otimizado
    y_pred_otimizado = modelo_otimizado.predict(test_x_vectorized)
    
    return {
        'modelo_otimizado': modelo_otimizado,
        'vectorizer': vectorizer,
        'test_x': test_x,
        'test_y': test_y,
        'y_pred': y_pred_otimizado,
        'acuracia': acuracia_otimizada,
        'f1_score': f1_otimizado
    }

def visualizar_matriz_confusao(test_y, y_pred):
    """
    Originalidade 3: Visualização da Matriz de Confusão com heatmap
    """
    print("\n" + "=" * 60)
    print("ORIGINALIDADE 3 - VISUALIZAÇÃO DA MATRIZ DE CONFUSÃO")
    print("=" * 60)
    
    # Calculando matriz de confusão
    cm = confusion_matrix(test_y, y_pred, labels=['negative', 'positive'])
    
    # Criando heatmap profissional
    plt.figure(figsize=(10, 8))
    
    # Definindo rótulos para as classes
    labels = ['Negativo', 'Positivo']
    
    # Criando heatmap com seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Quantidade de Reviews'})
    
    plt.title('Matriz de Confusão - Modelo SVM Otimizado\nAnálise de Sentimentos IMDB', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Previsão do Modelo', fontsize=14, fontweight='bold')
    plt.ylabel('Sentimento Real', fontsize=14, fontweight='bold')
    
    # Adicionando anotações explicativas
    plt.text(-0.4, -0.5, 'Verdadeiros Negativos (TN)', fontsize=12, fontweight='bold', color='darkblue')
    plt.text(0.6, -0.5, 'Falsos Positivos (FP)', fontsize=12, fontweight='bold', color='darkred')
    plt.text(-0.4, 0.5, 'Falsos Negativos (FN)', fontsize=12, fontweight='bold', color='darkred')
    plt.text(0.6, 0.5, 'Verdadeiros Positivos (TP)', fontsize=12, fontweight='bold', color='darkblue')
    
    plt.tight_layout()
    plt.savefig('matriz_confusao_avancada.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculando métricas detalhadas
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📊 MÉTRICAS DETALHADAS:")
    print(f"   • Verdadeiros Negativos (TN): {tn}")
    print(f"   • Falsos Positivos (FP): {fp}")
    print(f"   • Falsos Negativos (FN): {fn}")
    print(f"   • Verdadeiros Positivos (TP): {tp}")
    
    print(f"\n📈 CÁLCULOS DERIVADOS:")
    print(f"   • Precisão: {tp/(tp+fp):.4f}")
    print(f"   • Recall: {tp/(tp+fn):.4f}")
    print(f"   • Especificidade: {tn/(tn+fp):.4f}")
    
    return cm, (tn, fp, fn, tp)

def analisar_erros_modelo(test_x, test_y, y_pred, vectorizer, modelo_otimizado):
    """
    Originalidade 4: Análise de Erros do Modelo
    """
    print("\n" + "=" * 60)
    print("ORIGINALIDADE 4 - ANÁLISE DE ERROS DO MODELO")
    print("=" * 60)
    
    # Criando DataFrame para análise
    df_analise = pd.DataFrame({
        'review': test_x,
        'sentimento_real': test_y,
        'previsao': y_pred
    })
    
    # Identificando erros
    df_analise['erro'] = df_analise['sentimento_real'] != df_analise['previsao']
    df_analise['tipo_erro'] = 'Correto'
    
    # Falsos Positivos (negativo real, positivo previsto)
    mask_fp = (df_analise['sentimento_real'] == 'negative') & (df_analise['previsao'] == 'positive')
    df_analise.loc[mask_fp, 'tipo_erro'] = 'Falso Positivo'
    
    # Falsos Negativos (positivo real, negativo previsto)
    mask_fn = (df_analise['sentimento_real'] == 'positive') & (df_analise['previsao'] == 'negative')
    df_analise.loc[mask_fn, 'tipo_erro'] = 'Falso Negativo'
    
    # Estatísticas dos erros
    print(f"\n📊 ESTATÍSTICAS DE ERROS:")
    print(df_analise['tipo_erro'].value_counts())
    
    # Análise de Falsos Positivos
    print("\n" + "🔴" * 50)
    print("FALSOS POSITIVOS - Reviews Negativos Classificados como Positivos")
    print("🔴" * 50)
    
    falsos_positivos = df_analise[df_analise['tipo_erro'] == 'Falso Positivo']
    
    if len(falsos_positivos) >= 2:
        for i, (idx, row) in enumerate(falsos_positivos.head(2).iterrows()):
            print(f"\n📝 EXEMPLO {i+1}:")
            print(f"Review: {row['review'][:200]}...")
            print(f"Sentimento Real: {row['sentimento_real']}")
            print(f"Previsão do Modelo: {row['previsao']}")
            
            # Análise do erro
            print("🔍 ANÁLISE DO ERRO:")
            if 'good' in row['review'].lower() or 'great' in row['review'].lower():
                print("   • Possível causa: Palavras positivas isoladas em contexto negativo")
            elif 'but' in row['review'].lower() or 'however' in row['review'].lower():
                print("   • Possível causa: Estrutura de negação complexa")
            elif len(row['review'].split()) < 20:
                print("   • Possível causa: Review muito curto, contexto limitado")
            else:
                print("   • Possível causa: Sarcasmo ou linguagem ambígua")
    else:
        print("⚠️ Não há falsos positivos suficientes para análise")
    
    # Análise de Falsos Negativos
    print("\n" + "🔵" * 50)
    print("FALSOS NEGATIVOS - Reviews Positivos Classificados como Negativos")
    print("🔵" * 50)
    
    falsos_negativos = df_analise[df_analise['tipo_erro'] == 'Falso Negativo']
    
    if len(falsos_negativos) >= 2:
        for i, (idx, row) in enumerate(falsos_negativos.head(2).iterrows()):
            print(f"\n📝 EXEMPLO {i+1}:")
            print(f"Review: {row['review'][:200]}...")
            print(f"Sentimento Real: {row['sentimento_real']}")
            print(f"Previsão do Modelo: {row['previsao']}")
            
            # Análise do erro
            print("🔍 ANÁLISE DO ERRO:")
            if 'bad' in row['review'].lower() or 'terrible' in row['review'].lower():
                print("   • Possível causa: Palavras negativas em contexto positivo")
            elif 'not' in row['review'].lower() or 'no' in row['review'].lower():
                print("   • Possível causa: Negação que não foi capturada corretamente")
            elif len(row['review'].split()) < 20:
                print("   • Possível causa: Review muito curto, contexto limitado")
            else:
                print("   • Possível causa: Linguagem neutra ou ambígua")
    else:
        print("⚠️ Não há falsos negativos suficientes para análise")
    
    return df_analise

def visualizar_importancia_palavras(vectorizer, modelo_otimizado, test_x, test_y):
    """
    Originalidade 5: Visualização de Importância das Palavras
    """
    print("\n" + "=" * 60)
    print("ORIGINALIDADE 5 - VISUALIZAÇÃO DE IMPORTÂNCIA DAS PALAVRAS")
    print("=" * 60)
    
    # Verificando se o modelo suporta feature_importances_ ou coef_
    if hasattr(modelo_otimizado.best_estimator_, 'coef_'):
        # Para SVM e Logistic Regression
        coeficientes = modelo_otimizado.best_estimator_.coef_[0]
        feature_names = vectorizer.get_feature_names_out()
        
        print("✅ Modelo SVM suporta análise de coeficientes!")
        
        # Criando DataFrame com coeficientes
        df_coef = pd.DataFrame({
            'palavra': feature_names,
            'coeficiente': coeficientes
        })
        
        # Ordenando por importância
        df_coef = df_coef.sort_values('coeficiente', ascending=False)
        
        # Palavras mais importantes para positivo e negativo
        palavras_positivas = df_coef.head(15)
        palavras_negativas = df_coef.tail(15)
        
        # Criando visualização
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Palavras positivas
        sns.barplot(data=palavras_positivas, x='coeficiente', y='palavra', ax=ax1, palette='Greens')
        ax1.set_title('15 Palavras Mais Importantes para Sentimento POSITIVO', 
                      fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Coeficiente (Importância)', fontsize=12)
        ax1.set_ylabel('Palavra', fontsize=12)
        
        # Palavras negativas
        sns.barplot(data=palavras_negativas, x='coeficiente', y='palavra', ax=ax2, palette='Reds')
        ax2.set_title('15 Palavras Mais Importantes para Sentimento NEGATIVO', 
                      fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('Coeficiente (Importância)', fontsize=12)
        ax2.set_ylabel('Palavra', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('importancia_palavras_svm.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 TOP 5 PALAVRAS POSITIVAS:")
        for i, (_, row) in enumerate(palavras_positivas.head().iterrows()):
            print(f"   {i+1}. '{row['palavra']}' (coef: {row['coeficiente']:.4f})")
        
        print(f"\n📊 TOP 5 PALAVRAS NEGATIVAS:")
        for i, (_, row) in enumerate(palavras_negativas.head().iterrows()):
            print(f"   {i+1}. '{row['palavra']}' (coef: {row['coeficiente']:.4f})")
        
        return df_coef
        
    elif hasattr(modelo_otimizado.best_estimator_, 'feature_importances_'):
        # Para Random Forest
        importancias = modelo_otimizado.best_estimator_.feature_importances_
        feature_names = vectorizer.get_feature_names_out()
        
        print("✅ Modelo Random Forest suporta análise de feature importance!")
        
        # Implementação similar para Random Forest
        # (código seria similar, mas usando feature_importances_)
        
    else:
        print("⚠️ Este modelo não suporta análise direta de importância de features")
        print("   • SVM com kernel RBF não permite extração direta de coeficientes")
        print("   • Considere usar SVM linear ou Logistic Regression para esta análise")
        
        # Alternativa: Análise baseada em frequência de palavras
        print("\n🔍 ANÁLISE ALTERNATIVA: Frequência de Palavras por Classe")
        
        # Separando reviews por classe
        reviews_positivos = test_x[test_y == 'positive']
        reviews_negativos = test_x[test_y == 'negative']
        
        # Vetorizando separadamente
        X_pos = vectorizer.transform(reviews_positivos)
        X_neg = vectorizer.transform(reviews_negativos)
        
        # Calculando frequência média
        freq_pos = np.mean(X_pos.toarray(), axis=0)
        freq_neg = np.mean(X_neg.toarray(), axis=0)
        
        # Diferença de frequência
        diff_freq = freq_pos - freq_neg
        
        # Criando DataFrame
        df_freq = pd.DataFrame({
            'palavra': vectorizer.get_feature_names_out(),
            'freq_positiva': freq_pos,
            'freq_negativa': freq_neg,
            'diferenca': diff_freq
        })
        
        # Ordenando por diferença
        df_freq = df_freq.sort_values('diferenca', ascending=False)
        
        # Visualizando
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Palavras mais frequentes em positivos
        palavras_pos = df_freq.head(15)
        sns.barplot(data=palavras_pos, x='diferenca', y='palavra', ax=ax1, palette='Greens')
        ax1.set_title('15 Palavras Mais Frequentes em Reviews POSITIVOS', 
                      fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Diferença de Frequência (Pos - Neg)', fontsize=12)
        
        # Palavras mais frequentes em negativos
        palavras_neg = df_freq.tail(15)
        sns.barplot(data=palavras_neg, x='diferenca', y='palavra', ax=ax2, palette='Reds')
        ax2.set_title('15 Palavras Mais Frequentes em Reviews NEGATIVOS', 
                      fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('Diferença de Frequência (Pos - Neg)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('frequencia_palavras_por_classe.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df_freq

def gerar_relatorio_final(resultados):
    """
    Gerando relatório final da comunicação avançada
    """
    print("\n" + "=" * 60)
    print("GERANDO RELATÓRIO FINAL")
    print("=" * 60)
    
    with open('relatorio_comunicacao_avancada.txt', 'w', encoding='utf-8') as f:
        f.write("RELATÓRIO COMUNICAÇÃO AVANÇADA - PROJETO ORIGINAL\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("ORIGINALIDADES IMPLEMENTADAS:\n")
        f.write("-" * 30 + "\n")
        f.write("3. Visualização da Matriz de Confusão com heatmap\n")
        f.write("4. Análise de Erros do Modelo (Falsos Positivos/Negativos)\n")
        f.write("5. Visualização de Importância das Palavras\n\n")
        
        f.write(f"RESULTADOS DO MODELO OTIMIZADO:\n")
        f.write("-" * 35 + "\n")
        f.write(f"Acurácia: {resultados['acuracia']:.4f}\n")
        f.write(f"F1-Score: {resultados['f1_score']:.4f}\n\n")
        
        f.write("ARQUIVOS GERADOS:\n")
        f.write("-" * 20 + "\n")
        f.write("• matriz_confusao_avancada.png\n")
        f.write("• importancia_palavras_svm.png (ou frequencia_palavras_por_classe.png)\n")
        f.write("• relatorio_comunicacao_avancada.txt\n\n")
        
        f.write("INSIGHTS DA ANÁLISE:\n")
        f.write("-" * 20 + "\n")
        f.write("• Visualização profissional da matriz de confusão\n")
        f.write("• Análise detalhada dos erros do modelo\n")
        f.write("• Identificação das palavras mais importantes\n")
        f.write("• Compreensão dos pontos fortes e fracos do modelo\n")
    
    print("✅ Relatório salvo em 'relatorio_comunicacao_avancada.txt'")

def main():
    """
    Função principal da comunicação avançada
    """
    print("🚀 INICIANDO COMUNICAÇÃO AVANÇADA - PROJETO ORIGINAL")
    print("=" * 60)
    
    # Carregando modelo e dados
    resultados = carregar_modelo_e_dados()
    
    # Originalidade 3: Matriz de Confusão
    cm, metricas = visualizar_matriz_confusao(resultados['test_y'], resultados['y_pred'])
    
    # Originalidade 4: Análise de Erros
    df_analise = analisar_erros_modelo(
        resultados['test_x'], resultados['test_y'], 
        resultados['y_pred'], resultados['vectorizer'], 
        resultados['modelo_otimizado']
    )
    
    # Originalidade 5: Importância das Palavras
    df_importancia = visualizar_importancia_palavras(
        resultados['vectorizer'], resultados['modelo_otimizado'],
        resultados['test_x'], resultados['test_y']
    )
    
    # Gerando relatório final
    gerar_relatorio_final(resultados)
    
    print("\n" + "=" * 60)
    print("✅ COMUNICAÇÃO AVANÇADA CONCLUÍDA COM SUCESSO!")
    print("=" * 60)
    
    print("\n📁 Arquivos gerados:")
    print("• matriz_confusao_avancada.png")
    print("• importancia_palavras_svm.png (ou frequencia_palavras_por_classe.png)")
    print("• relatorio_comunicacao_avancada.txt")
    
    print("\n🎯 RESUMO DAS ORIGINALIDADES:")
    print("3. Visualização profissional da matriz de confusão")
    print("4. Análise detalhada de erros do modelo")
    print("5. Visualização de importância das palavras")

if __name__ == "__main__":
    main() 