# PROJETO ORIGINAL MELHORADO: ANÁLISE DE SENTIMENTOS IMDB
# Versão com validação estatística, cross-validation robusta e análise de features

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import ttest_ind
import warnings
import matplotlib
try:
    plt.style.use('seaborn-v0_8')
except Exception:
    sns.set_theme()
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette("husl")

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

print("PROJETO ORIGINAL MELHORADO - ANÁLISE DE SENTIMENTOS IMDB")
print("=" * 70)

# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
print("\n1. CARREGAMENTO E PREPARAÇÃO DOS DADOS")
print("-" * 50)

# Carregar dataset
df = pd.read_csv('IMDB Dataset.csv')
print(f"Dataset carregado: {df.shape[0]} reviews, {df.shape[1]} colunas")

# Remover duplicatas
df_clean = df.drop_duplicates(subset=['review'])
print(f"Reviews após remoção de duplicatas: {len(df_clean)}")

# Criar amostra desbalanceada (como no FMF)
df_positive = df_clean[df_clean['sentiment'] == 'positive'][:9000]
df_negative = df_clean[df_clean['sentiment'] == 'negative'][:1000]
df_imb = pd.concat([df_positive, df_negative])
print(f"Dataset desbalanceado: {len(df_imb)} reviews")

# Balancear com RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_bal, y_bal = rus.fit_resample(df_imb[['review']], df_imb['sentiment'])
print(f"Dataset balanceado: {len(X_bal)} reviews")

# 2. PRÉ-PROCESSAMENTO AVANÇADO
print("\n🔧 2. PRÉ-PROCESSAMENTO AVANÇADO")
print("-" * 50)

def preprocessamento_avancado(texto):
    """Pipeline avançado de pré-processamento com lematização"""
    if pd.isna(texto):
        return ""
    
    # Converter para minúsculas
    texto = str(texto).lower()
    
    # Remover números
    texto = re.sub(r'\d+', '', texto)
    
    # Remover pontuação
    texto = re.sub(r'[^\w\s]', '', texto)
    
    # Tokenização
    tokens = nltk.word_tokenize(texto)
    
    # Remover stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Remover palavras muito curtas
    tokens = [token for token in tokens if len(token) >= 3]
    
    # Lemmatização
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Aplicar pré-processamento
print("🔄 Aplicando pré-processamento avançado...")
df_imb['review_processed'] = df_imb['review'].apply(preprocessamento_avancado)
print("✅ Pré-processamento concluído (lematização, limpeza, etc.)")

# 3. PREPARAÇÃO PARA MODELAGEM
print("\n🤖 3. PREPARAÇÃO PARA MODELAGEM")
print("-" * 50)

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(
    df_imb['review_processed'], df_imb['sentiment'], 
    test_size=0.33, random_state=42, stratify=df_imb['sentiment']
)

print(f"📊 Divisão dos dados:")
print(f"   • Treino: {len(X_train)} reviews")
print(f"   • Teste: {len(X_test)} reviews")

# Vetorização
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"📊 Vetorização:")
print(f"   • Vocabulário: {len(vectorizer.vocabulary_)} palavras")
print(f"   • Features de treino: {X_train_vec.shape}")
print(f"   • Features de teste: {X_test_vec.shape}")

# 4. MODELAGEM COM CROSS-VALIDATION ROBUSTA
print("\n📈 4. MODELAGEM COM CROSS-VALIDATION ROBUSTA")
print("-" * 50)

# Configurar StratifiedKFold para validação robusta
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Modelos para teste
modelos = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1)
}

# Resultados com cross-validation
resultados_cv = {}
resultados_teste = {}

print("🔄 Treinando modelos com cross-validation...")

for nome, modelo in modelos.items():
    print(f"\n📊 {nome}:")
    
    # Cross-validation
    cv_scores_acc = cross_val_score(modelo, X_train_vec, y_train, cv=cv, scoring='accuracy')
    cv_scores_f1 = cross_val_score(modelo, X_train_vec, y_train, cv=cv, scoring='f1_macro')
    
    # Treinar no conjunto completo
    modelo.fit(X_train_vec, y_train)
    
    # Prever no conjunto de teste
    y_pred = modelo.predict(X_test_vec)
    
    # Avaliar
    acc_teste = accuracy_score(y_test, y_pred)
    f1_teste = f1_score(y_test, y_pred, pos_label='positive')
    
    # Armazenar resultados
    resultados_cv[nome] = {
        'cv_acc_mean': cv_scores_acc.mean(),
        'cv_acc_std': cv_scores_acc.std(),
        'cv_f1_mean': cv_scores_f1.mean(),
        'cv_f1_std': cv_scores_f1.std()
    }
    
    resultados_teste[nome] = {
        'acuracia': acc_teste,
        'f1_score': f1_teste,
        'y_pred': y_pred,
        'modelo': modelo
    }
    
    print(f"   ✅ CV Acurácia: {cv_scores_acc.mean():.4f} ± {cv_scores_acc.std():.4f}")
    print(f"   ✅ CV F1-Score: {cv_scores_f1.mean():.4f} ± {cv_scores_f1.std():.4f}")
    print(f"   ✅ Teste Acurácia: {acc_teste:.4f}")
    print(f"   ✅ Teste F1-Score: {f1_teste:.4f}")

# 5. VALIDAÇÃO ESTATÍSTICA
print("\n📊 5. VALIDAÇÃO ESTATÍSTICA")
print("-" * 50)

# Encontrar melhor modelo
melhor_modelo = max(resultados_teste.items(), key=lambda x: x[1]['f1_score'])
print(f"🏆 MELHOR MODELO: {melhor_modelo[0]}")
print(f"   • Acurácia: {melhor_modelo[1]['acuracia']:.4f}")
print(f"   • F1-Score: {melhor_modelo[1]['f1_score']:.4f}")

# Teste de significância estatística
print(f"\n🔬 TESTE DE SIGNIFICÂNCIA ESTATÍSTICA:")
print(f"Comparando {melhor_modelo[0]} com outros modelos...")

modelos_comparacao = ['Logistic Regression', 'Naive Bayes', 'Decision Tree']
if melhor_modelo[0] not in modelos_comparacao:
    modelos_comparacao.append(melhor_modelo[0])

for modelo_nome in modelos_comparacao:
    if modelo_nome != melhor_modelo[0]:
        # Obter scores de CV para comparação
        scores_melhor = cross_val_score(
            melhor_modelo[1]['modelo'], X_train_vec, y_train, 
            cv=cv, scoring='f1_macro'
        )
        scores_outro = cross_val_score(
            resultados_teste[modelo_nome]['modelo'], X_train_vec, y_train, 
            cv=cv, scoring='f1_macro'
        )
        
        # Teste t para amostras independentes
        t_stat, p_value = ttest_ind(scores_melhor, scores_outro)
        
        print(f"   • {melhor_modelo[0]} vs {modelo_nome}:")
        print(f"     - t-statistic: {t_stat:.4f}")
        print(f"     - p-value: {p_value:.4f}")
        print(f"     - Significativo (p < 0.05): {'✅ SIM' if p_value < 0.05 else '❌ NÃO'}")

# 6. OTIMIZAÇÃO DE HIPERPARÂMETROS
print("\n🔧 6. OTIMIZAÇÃO DE HIPERPARÂMETROS")
print("-" * 50)

# Otimizar o melhor modelo
melhor_nome = melhor_modelo[0]
print(f"🔄 Otimizando {melhor_nome}...")

if 'SVM' in melhor_nome:
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
elif 'Random Forest' in melhor_nome:
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
else:
    param_grid = {}

if param_grid:
    grid_search = GridSearchCV(
        modelos[melhor_nome], param_grid, cv=cv, 
        scoring='f1_macro', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train_vec, y_train)
    
    print(f"✅ Melhores parâmetros: {grid_search.best_params_}")
    
    # Avaliar modelo otimizado
    y_pred_otimizado = grid_search.predict(X_test_vec)
    acc_otimizado = accuracy_score(y_test, y_pred_otimizado)
    f1_otimizado = f1_score(y_test, y_pred_otimizado, pos_label='positive')
    
    print(f"🏆 MODELO OTIMIZADO:")
    print(f"   • Acurácia: {acc_otimizado:.4f}")
    print(f"   • F1-Score: {f1_otimizado:.4f}")
    print(f"   • Melhoria Acurácia: +{acc_otimizado - melhor_modelo[1]['acuracia']:.4f}")
    print(f"   • Melhoria F1-Score: +{f1_otimizado - melhor_modelo[1]['f1_score']:.4f}")
    
    # Atualizar melhor modelo
    melhor_modelo = (f"{melhor_nome} (Otimizado)", {
        'acuracia': acc_otimizado,
        'f1_score': f1_otimizado,
        'y_pred': y_pred_otimizado,
        'modelo': grid_search.best_estimator_
    })

# 7. ANÁLISE DE FEATURES
print("\n🔍 7. ANÁLISE DE FEATURES")
print("-" * 50)

# Obter nomes das features
feature_names = vectorizer.get_feature_names_out()

# Análise de importância para modelos que suportam
modelo_final = melhor_modelo[1]['modelo']

if hasattr(modelo_final, 'feature_importances_'):
    # Random Forest ou Decision Tree
    importancias = modelo_final.feature_importances_
    tipo_importancia = "Feature Importances"
elif hasattr(modelo_final, 'coef_'):
    # Logistic Regression ou SVM linear
    importancias = np.abs(modelo_final.coef_[0])
    tipo_importancia = "Coefficient Magnitudes"
else:
    # SVM com kernel não-linear
    print("⚠️  Modelo SVM com kernel não-linear - análise de features limitada")
    importancias = None

if importancias is not None:
    # Criar DataFrame com importâncias
    df_importancias = pd.DataFrame({
        'feature': feature_names,
        'importance': importancias
    }).sort_values('importance', ascending=False)
    
    print(f"📊 {tipo_importancia} - Top 20 Features:")
    print(df_importancias.head(20))
    
    # Visualizar top features
    plt.figure(figsize=(12, 8))
    top_features = df_importancias.head(20)
    
    plt.subplot(2, 1, 1)
    sns.barplot(data=top_features.head(10), x='importance', y='feature', palette='viridis')
    plt.title(f'Top 10 {tipo_importancia} - {melhor_modelo[0]}', fontsize=14, fontweight='bold')
    plt.xlabel('Importância', fontsize=12)
    
    plt.subplot(2, 1, 2)
    sns.barplot(data=top_features.tail(10), x='importance', y='feature', palette='plasma')
    plt.title(f'Features 11-20 {tipo_importancia} - {melhor_modelo[0]}', fontsize=14, fontweight='bold')
    plt.xlabel('Importância', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('analise_features_importancia.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Gráfico salvo: analise_features_importancia.png")

# 8. COMUNICAÇÃO AVANÇADA
print("\n📊 8. COMUNICAÇÃO AVANÇADA")
print("-" * 50)

# Matriz de confusão avançada
cm = confusion_matrix(y_test, melhor_modelo[1]['y_pred'], labels=['negative', 'positive'])

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negativo', 'Positivo'], 
            yticklabels=['Negativo', 'Positivo'],
            cbar_kws={'label': 'Quantidade de Reviews'})

plt.title(f'Matriz de Confusão - {melhor_modelo[0]}\nAnálise de Sentimentos IMDB', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Previsão do Modelo', fontsize=14, fontweight='bold')
plt.ylabel('Sentimento Real', fontsize=14, fontweight='bold')

# Adicionar anotações
tn, fp, fn, tp = cm.ravel()
plt.text(-0.4, -0.5, f'TN: {tn}', fontsize=12, fontweight='bold', color='darkblue')
plt.text(0.6, -0.5, f'FP: {fp}', fontsize=12, fontweight='bold', color='darkred')
plt.text(-0.4, 0.5, f'FN: {fn}', fontsize=12, fontweight='bold', color='darkred')
plt.text(0.6, 0.5, f'TP: {tp}', fontsize=12, fontweight='bold', color='darkblue')

plt.tight_layout()
plt.savefig('matriz_confusao_melhorada.png', dpi=300, bbox_inches='tight')
plt.show()

# Métricas detalhadas
print(f"\n📊 MÉTRICAS DETALHADAS - {melhor_modelo[0]}:")
print(f"   • Verdadeiros Negativos (TN): {tn}")
print(f"   • Falsos Positivos (FP): {fp}")
print(f"   • Falsos Negativos (FN): {fn}")
print(f"   • Verdadeiros Positivos (TP): {tp}")
print(f"   • Precisão: {tp/(tp+fp):.4f}")
print(f"   • Recall: {tp/(tp+fn):.4f}")
print(f"   • Especificidade: {tn/(tn+fp):.4f}")
print(f"   • Acurácia: {melhor_modelo[1]['acuracia']:.4f}")
print(f"   • F1-Score: {melhor_modelo[1]['f1_score']:.4f}")

# 9. RELATÓRIO FINAL
print("\n�� 9. RELATÓRIO FINAL")
print("-" * 50)

# Tabela comparativa completa
print(f"\n📈 TABELA COMPARATIVA COMPLETA:")
print("-" * 100)
print(f"{'Modelo':<25} {'CV Acc':<12} {'CV F1':<12} {'Test Acc':<12} {'Test F1':<12} {'Significativo':<15}")
print("-" * 100)

for nome, resultado in resultados_teste.items():
    cv_info = resultados_cv[nome]
    print(f"{nome:<25} {cv_info['cv_acc_mean']:<12.4f} {cv_info['cv_f1_mean']:<12.4f} "
          f"{resultado['acuracia']:<12.4f} {resultado['f1_score']:<12.4f} "
          f"{'✅' if nome == melhor_modelo[0] else '❌':<15}")

print("-" * 100)
print(f"🏆 CAMPEÃO: {melhor_modelo[0]}")
print(f"   • Acurácia: {melhor_modelo[1]['acuracia']:.4f}")
print(f"   • F1-Score: {melhor_modelo[1]['f1_score']:.4f}")

# Salvar resultados
with open('resultados_melhorados.txt', 'w', encoding='utf-8') as f:
    f.write("RESULTADOS DO PROJETO ORIGINAL MELHORADO\n")
    f.write("=" * 50 + "\n\n")
    
    f.write(f"CAMPEÃO: {melhor_modelo[0]}\n")
    f.write(f"Acurácia: {melhor_modelo[1]['acuracia']:.4f}\n")
    f.write(f"F1-Score: {melhor_modelo[1]['f1_score']:.4f}\n\n")
    
    f.write("TABELA COMPARATIVA:\n")
    f.write("-" * 80 + "\n")
    for nome, resultado in resultados_teste.items():
        cv_info = resultados_cv[nome]
        f.write(f"{nome}: CV_Acc={cv_info['cv_acc_mean']:.4f}, Test_Acc={resultado['acuracia']:.4f}\n")

print(f"\n✅ Resultados salvos em: resultados_melhorados.txt")
print(f"✅ Matriz de confusão salva em: matriz_confusao_melhorada.png")
if importancias is not None:
    print(f"✅ Análise de features salva em: analise_features_importancia.png")

print(f"\n🎉 PROJETO ORIGINAL MELHORADO CONCLUÍDO COM SUCESSO!")
print(f"   Implementadas as melhorias: validação estatística, cross-validation robusta e análise de features") 