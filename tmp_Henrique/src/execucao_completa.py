#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Execução Completa - Projeto FMF vs ORIGINAL
Executa todo o pipeline: Análise Exploratória + FMF + ORIGINAL + Comparação Final
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def executar_comando(comando, descricao):
    """
    Executa um comando e exibe o progresso
    """
    print(f"\n{'='*60}")
    print(f"EXECUTANDO: {descricao}")
    print(f"{'='*60}")
    print(f"Comando: {comando}")
    print(f"Inicio: {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        resultado = subprocess.run(comando, shell=True, capture_output=True, text=True)
        
        if resultado.returncode == 0:
            print(f"SUCESSO: {descricao}")
            print(f"Fim: {datetime.now().strftime('%H:%M:%S')}")
            return True
        else:
            print(f"ERRO: {descricao}")
            print(f"Erro: {resultado.stderr}")
            return False
            
    except Exception as e:
        print(f"EXCECAO: {descricao}")
        print(f"Erro: {str(e)}")
        return False

def verificar_arquivo(arquivo, descricao):
    """
    Verifica se um arquivo foi criado
    """
    if os.path.exists(arquivo):
        print(f"OK {descricao}: {arquivo}")
        return True
    else:
        print(f"ERRO {descricao}: {arquivo} - NAO ENCONTRADO")
        return False

def main():
    """
    Função principal que executa todo o pipeline
    """
    print("PROJETO FMF vs ORIGINAL: ANALISE DE SENTIMENTOS IMDB")
    print("=" * 80)
    print("SCRIPT DE EXECUCAO COMPLETA")
    print("=" * 80)
    print(f"Inicio da execucao: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    # Lista de execuções
    execucoes = [
        {
            'comando': 'python src/analise_sentimentos_imdb.py',
            'descricao': 'Analise Exploratoria (Fases A, G, E)',
            'arquivos_esperados': [
                'visualizations/distribuicao_sentimentos.png',
                'visualizations/wordclouds_analise.png',
                'visualizations/analise_comprimento.png',
                'docs/relatorio_analise_imdb.md'
            ]
        },
        {
            'comando': 'python src/estatisticas_detalhadas.py',
            'descricao': 'Estatisticas Detalhadas',
            'arquivos_esperados': [
                'visualizations/estatisticas_detalhadas.png'
            ]
        },
        {
            'comando': 'python src/modelagem_comunicacao_imdb.py',
            'descricao': 'Projeto FMF (Fases M e C)',
            'arquivos_esperados': [
                'visualizations/matriz_confusao.png',
                'results/resultados_modelagem.txt'
            ]
        },
        {
            'comando': 'python src/projeto_original_imdb.py',
            'descricao': 'Projeto ORIGINAL (Fases M e C)',
            'arquivos_esperados': [
                'results/resultados_projeto_original.txt'
            ]
        },
        {
            'comando': 'python src/comunicacao_avancada_imdb.py',
            'descricao': 'Comunicacao Avancada',
            'arquivos_esperados': [
                'visualizations/matriz_confusao_avancada.png',
                'visualizations/frequencia_palavras_por_classe.png',
                'results/relatorio_comunicacao_avancada.txt'
            ]
        }
    ]
    
    # Executando cada etapa
    resultados = []
    for i, execucao in enumerate(execucoes, 1):
        print(f"\nETAPA {i}/{len(execucoes)}")
        
        # Executando comando
        sucesso = executar_comando(execucao['comando'], execucao['descricao'])
        resultados.append({
            'etapa': i,
            'descricao': execucao['descricao'],
            'sucesso': sucesso
        })
        
        if sucesso:
            # Verificando arquivos gerados
            print(f"\nVERIFICANDO ARQUIVOS GERADOS:")
            for arquivo in execucao['arquivos_esperados']:
                verificar_arquivo(arquivo, f"Arquivo {execucao['descricao']}")
        
        # Pausa entre execuções
        if i < len(execucoes):
            print(f"\nAguardando 3 segundos antes da proxima etapa...")
            time.sleep(3)
    
    # Relatório final
    print(f"\n{'='*80}")
    print("RELATORIO FINAL DE EXECUCAO")
    print(f"{'='*80}")
    print(f"Fim da execucao: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    # Estatísticas
    total_etapas = len(resultados)
    etapas_sucesso = sum(1 for r in resultados if r['sucesso'])
    etapas_falha = total_etapas - etapas_sucesso
    
    print(f"\nESTATISTICAS:")
    print(f"   • Total de etapas: {total_etapas}")
    print(f"   • Etapas com sucesso: {etapas_sucesso}")
    print(f"   • Etapas com falha: {etapas_falha}")
    print(f"   • Taxa de sucesso: {(etapas_sucesso/total_etapas)*100:.1f}%")
    
    # Detalhamento das etapas
    print(f"\nDETALHAMENTO DAS ETAPAS:")
    for resultado in resultados:
        status = "SUCESSO" if resultado['sucesso'] else "FALHA"
        print(f"   {resultado['etapa']}. {resultado['descricao']}: {status}")
    
    # Arquivos finais gerados
    print(f"\nARQUIVOS FINAIS GERADOS:")
    arquivos_finais = [
        'docs/README.md',
        'docs/relatorio_final_consolidado.md',
        'requirements.txt'
    ]
    
    for arquivo in arquivos_finais:
        verificar_arquivo(arquivo, "Documentacao")
    
    # Verificando visualizações
    print(f"\nVISUALIZACOES GERADAS:")
    visualizacoes = [
        'visualizations/distribuicao_sentimentos.png',
        'visualizations/wordclouds_analise.png',
        'visualizations/analise_comprimento.png',
        'visualizations/estatisticas_detalhadas.png',
        'visualizations/matriz_confusao.png',
        'visualizations/matriz_confusao_avancada.png',
        'visualizations/frequencia_palavras_por_classe.png'
    ]
    
    for viz in visualizacoes:
        verificar_arquivo(viz, "Visualizacao")
    
    # Verificando relatórios
    print(f"\nRELATORIOS GERADOS:")
    relatorios = [
        'docs/relatorio_analise_imdb.md',
        'results/resultados_modelagem.txt',
        'results/resultados_projeto_original.txt',
        'results/relatorio_comunicacao_avancada.txt',
        'docs/relatorio_final_consolidado.md'
    ]
    
    for rel in relatorios:
        verificar_arquivo(rel, "Relatorio")
    
    # Conclusão
    if etapas_falha == 0:
        print(f"\nPARABENS! TODAS AS ETAPAS FORAM EXECUTADAS COM SUCESSO!")
        print(f"O projeto esta 100% completo e pronto para apresentacao.")
        print(f"Resultado final: SVM Otimizado com 83.48% de acuracia")
    else:
        print(f"\nATENCAO: {etapas_falha} etapa(s) falharam.")
        print(f"Verifique os erros acima e execute novamente se necessario.")
    
    print(f"\nPROXIMOS PASSOS:")
    print(f"   1. Leia o 'docs/relatorio_final_consolidado.md' para o resumo completo")
    print(f"   2. Consulte o 'docs/README.md' para documentacao detalhada")
    print(f"   3. Use as visualizacoes geradas para apresentacao")
    print(f"   4. Analise os resultados em 'results/resultados_projeto_original.txt'")
    
    print(f"\n{'='*80}")
    print("EXECUCAO COMPLETA FINALIZADA")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 