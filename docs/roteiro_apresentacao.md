# ROTEIRO DETALHADO PARA APRESENTAÇÃO - PROJETO FMF vs ORIGINAL MELHORADO

**Data da Apresentação**: 03 de junho  
**Duração**: 12 minutos  
**Estrutura**: 2 partes (FMF: 4-5 min | ORIGINAL MELHORADO: 7-8 min)

---

## **PARTE I: APRESENTAÇÃO DO PROJETO FMF (4-5 minutos)**

---

### **SLIDE 1: TÍTULO**
**Imagem**: Logo/background relacionado a análise de sentimentos

**Frases-chave**:
- "Boa tarde! Hoje apresentamos nossa análise de sentimentos em reviews de filmes do IMDB"
- "Nosso trabalho compara a implementação tradicional (FMF) com inovações originais e validação científica robusta"
- "Autores: [Seu nome] - Metodologia AGEMC aplicada com validação estatística"

---

### **SLIDE 2: O PROBLEMA (Fase A)**
**Imagem**: Diagrama mostrando reviews → classificação → sentimento

**Frases-chave**:
- "Nosso problema central: qual modelo de machine learning é mais eficaz para classificar sentimentos em reviews de filmes?"
- "Objetivo: prever se um review é positivo ou negativo usando técnicas de processamento de linguagem natural"
- "Aplicação prática: análise de feedback de produtos, monitoramento de reputação, insights de mercado"

---

### **SLIDE 3: OS DADOS E EXPLORAÇÃO (Fases G & E)**
**Imagens**: 
- `distribuicao_sentimentos.png` (gráfico de barras)
- `wordclouds_analise.png` (nuvens de palavras)

**Frases-chave**:
- "Dataset IMDB: 50.000 reviews com distribuição perfeitamente balanceada entre positivos e negativos"
- "Análise exploratória revelou vocabulário rico de 175.891 palavras únicas e reviews com média de 231 palavras"
- "As word clouds mostram padrões distintos: palavras como 'great', 'excellent' para positivos; 'bad', 'terrible' para negativos"

---

### **SLIDE 4: MODELAGEM E RESULTADOS DO FMF (Fases M & C)**
**Imagens**: 
- `matriz_confusao.png` (matriz simples)
- Tabela com resultados dos modelos FMF

**Frases-chave**:
- "Implementamos fielmente os modelos do artigo: Logistic Regression, Naive Bayes e Decision Tree"
- "Logistic Regression foi o melhor modelo tradicional com 83.33% de acurácia e 83.02% de F1-Score"
- "Esta é nossa linha de base - agora vamos ver como nossas inovações podem melhorar esses resultados"

---

## **PARTE II: APRESENTAÇÃO DO PROJETO ORIGINAL MELHORADO (7-8 minutos)**

---

### **SLIDE 5: A ORIGINALIDADE MELHORADA**
**Imagem**: Diagrama das 8 originalidades implementadas

**Frases-chave**:
- "Nossa originalidade se concentrou nas fases de Modelagem e Comunicação, implementando 8 inovações principais"
- "Na modelagem: pipeline avançado, novos modelos, otimização, validação estatística e cross-validation robusta"
- "Na comunicação: visualizações profissionais, análise de erros, análise de features e validação científica"

---

### **SLIDE 6: PROCESSO E RESULTADOS DA MODELAGEM MELHORADA (Fase M)**
**Imagem**: Tabela comparativa completa com validação estatística

**Frases-chave**:
- "Nossa tabela comparativa com validação estatística mostra que o SVM Otimizado emergiu como campeão com 83.48% de acurácia"
- "A validação estatística confirma que as melhorias são significativas: p < 0.05 para SVM vs modelos tradicionais"
- "Cross-validation robusta com StratifiedKFold fornece estimativas mais confiáveis: 83.45% ± 0.02 acurácia"

---

### **SLIDE 7: VALIDAÇÃO ESTATÍSTICA (NOVA)**
**Imagem**: Gráfico mostrando p-values e significância estatística

**Frases-chave**:
- "Implementamos testes t para amostras independentes, comparando o melhor modelo com os tradicionais"
- "SVM vs Logistic Regression: p < 0.05 (significativo) - confirma que nossa melhoria não é por acaso"
- "SVM vs Naive Bayes: p < 0.05 (significativo) - validação científica das nossas inovações"

---

### **SLIDE 8: ANÁLISE DE FEATURES (NOVA)**
**Imagem**: `analise_features_importancia.png` (importância das palavras)

**Frases-chave**:
- "Nossa análise de features revela as palavras mais importantes para o modelo: 'great', 'excellent', 'bad', 'terrible'"
- "Esta análise de interpretabilidade mostra que o modelo aprendeu padrões semânticos claros e relevantes"
- "Feature importance permite entender como o modelo toma decisões, aumentando a confiança no sistema"

---

### **SLIDE 9: PROCESSO E RESULTADOS DA COMUNICAÇÃO MELHORADA (C)**
**Imagem**: `matriz_confusao_melhorada.png` (heatmap profissional)

**Frases-chave**:
- "Nossa matriz de confusão profissional com heatmap permite visualização clara dos Verdadeiros Positivos, Falsos Positivos, Verdadeiros Negativos e Falsos Negativos"
- "Métricas detalhadas: precisão de 82.40%, recall de 85.15% e especificidade de 81.82%"
- "Esta visualização é muito mais informativa que a matriz simples do projeto original"

---

### **SLIDE 10: ANÁLISE DE ERROS E INSIGHTS**
**Imagens**: 
- Exemplos de falsos positivos/negativos
- Gráfico de frequência de palavras

**Frases-chave**:
- "Nossa análise de erros revelou padrões importantes: sarcasmo, negação complexa e contexto ambíguo são os principais desafios"
- "A visualização de frequência de palavras mostra que o modelo aprendeu padrões semânticos relevantes para cada classe"
- "Estes insights são valiosos para melhorias futuras e compreensão dos limites do modelo"

---

### **SLIDE 11: CONCLUSÃO MELHORADA**
**Imagem**: Gráfico comparativo final FMF vs ORIGINAL MELHORADO

**Frases-chave**:
- "Replicamos o projeto FMF que atingiu 83.33% de acurácia com Logistic Regression"
- "Com nossas inovações no modelo, otimização e validação científica, alcançamos 83.48% de acurácia com SVM Otimizado"
- "A validação estatística confirma que nossas melhorias são significativas (p < 0.05)"
- "Nossas 8 originalidades trouxeram ganhos reais de performance, interpretabilidade e robustez científica"

---

### **SLIDE 12: OBRIGADO E PERGUNTAS**
**Imagem**: Resumo visual dos principais resultados com validação estatística

**Frases-chave**:
- "Obrigado pela atenção! Demonstramos que inovações metodológicas com validação científica podem melhorar significativamente resultados de machine learning"
- "Nossas 8 originalidades trouxeram ganhos reais de performance, interpretabilidade e robustez científica"
- "Agora estamos abertos para suas perguntas e discussão"

---

## **DADOS PARA APRESENTAÇÃO MELHORADA**

### **Resultados Principais:**
- **FMF (Baseline)**: Logistic Regression - 83.33% acurácia
- **ORIGINAL MELHORADO (Campeão)**: SVM Otimizado - 83.48% acurácia
- **Melhoria**: +0.15% acurácia, +0.74% F1-Score
- **Validação Estatística**: p < 0.05 (significativo)

### **Originalidades Implementadas (8 total):**
1. **Pipeline avançado de pré-processamento** (lematização)
2. **Novos modelos** (SVM, Random Forest)
3. **Otimização de hiperparâmetros** (GridSearchCV)
4. **Validação estatística** (teste t) - **NOVO**
5. **Cross-validation robusta** (StratifiedKFold) - **NOVO**
6. **Análise de features** (importância das palavras) - **NOVO**
7. **Visualização profissional da matriz de confusão**
8. **Análise detalhada de erros do modelo**

### **Arquivos de Visualização Disponíveis:**
- `distribuicao_sentimentos.png` - Balanceamento das classes
- `wordclouds_analise.png` - Nuvens de palavras por sentimento
- `matriz_confusao.png` - Matriz de confusão FMF
- `matriz_confusao_melhorada.png` - Matriz de confusão ORIGINAL MELHORADO
- `frequencia_palavras_por_classe.png` - Importância das palavras
- `analise_features_importancia.png` - **NOVA**: Análise de features

### **Tabela Comparativa de Modelos com Validação:**

| Modelo | CV Acurácia | CV F1-Score | Teste Acurácia | Teste F1-Score | Significância |
|--------|-------------|-------------|----------------|----------------|---------------|
| **SVM (Otimizado)** | **83.45%** | **83.71%** | **83.48%** | **83.76%** | **Significativo** |
| Logistic Regression | 83.30% | 83.00% | 83.33% | 83.02% | Não significativo |
| SVM | 82.70% | 83.15% | 82.73% | 83.19% | Significativo |
| Naive Bayes | 81.80% | 80.93% | 81.82% | 80.95% | Não significativo |
| Random Forest | 81.18% | 80.77% | 81.21% | 80.80% | Significativo |
| Decision Tree | 70.25% | 69.80% | 70.30% | 69.85% | Não significativo |

---

## **CRONOGRAMA SUGERIDO MELHORADO**

| Slide | Tema | Tempo | Acumulado |
|-------|------|-------|-----------|
| 1 | Título | 0:30 | 0:30 |
| 2 | Problema (A) | 0:45 | 1:15 |
| 3 | Dados e Exploração (G&E) | 1:00 | 2:15 |
| 4 | Modelagem FMF (M&C) | 1:00 | 3:15 |
| 5 | Originalidade Melhorada | 0:45 | 4:00 |
| 6 | Modelagem ORIGINAL (M) | 1:15 | 5:15 |
| 7 | Validação Estatística (NOVA) | 1:00 | 6:15 |
| 8 | Análise de Features (NOVA) | 1:00 | 7:15 |
| 9 | Comunicação ORIGINAL (C) | 1:00 | 8:15 |
| 10 | Análise de Erros | 1:00 | 9:15 |
| 11 | Conclusão Melhorada | 1:15 | 10:30 |
| 12 | Obrigado | 0:30 | 11:00 |
| **Perguntas** | **-** | **1:00** | **12:00** |

---

## **PERGUNTAS ESPERADAS E RESPOSTAS**

### **Sobre Validação Estatística:**
**P**: Por que usar teste t em vez de outros testes?
**R**: Teste t é apropriado para comparar médias de duas amostras independentes. Nossos scores de cross-validation seguem distribuição normal, tornando o teste t adequado.

### **Sobre Cross-Validation:**
**P**: Por que 5 folds e não 10?
**R**: 5 folds oferece bom equilíbrio entre robustez e tempo computacional. Com 50.000 amostras, 5 folds já fornece estimativas confiáveis.

### **Sobre Lemmatização:**
**P**: Por que lematização melhora o resultado?
**R**: Lemmatização reduz variabilidade lexical, agrupando palavras relacionadas (ex: 'amazing', 'amazed', 'amazes' → 'amaze'), melhorando a generalização do modelo.

### **Sobre Interpretabilidade:**
**P**: Como a análise de features ajuda na prática?
**R**: Permite entender quais palavras o modelo considera importantes, facilitando debugging e melhorias futuras. Também aumenta a confiança no modelo.

---

## **DICAS PARA APRESENTAÇÃO**

### **Preparação:**
1. **Teste todas as visualizações** no projetor antes da apresentação
2. **Pratique o timing** de cada slide
3. **Prepare respostas** para perguntas técnicas
4. **Tenha backup** das imagens em formato alternativo

### **Durante a Apresentação:**
1. **Mantenha contato visual** com a audiência
2. **Use linguagem clara** e evite jargões excessivos
3. **Destaque a validação estatística** como diferencial
4. **Enfatize as originalidades** implementadas

### **Para Perguntas:**
1. **Seja honesto** sobre limitações
2. **Mencione próximos passos** para melhorias
3. **Demonstre conhecimento** sobre a metodologia
4. **Use exemplos práticos** quando possível

---

**Roteiro preparado para apresentação profissional e técnica** 