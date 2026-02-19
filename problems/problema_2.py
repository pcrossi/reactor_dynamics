# -*- coding: utf-8 -*-
"""
Módulo: problema_2.py
Descrição: Script de validação (Benchmark) conforme Ganapol (2013), Tabela 1c.
Cenário: Reator Térmico I com inserção de reatividade em degrau de $1.0 (Crítico-Pronto).
Objetivo: Validar a estabilidade do integrador numérico frente a variações exponenciais extremas.
"""

import numpy as np
from reactor_dynamics import ReactorPhysics, MultiPhysicsSolver

def problema_2():
    """
    Executa o Benchmark de Ganapol (2013) para o cenário de inserção em degrau de $1.0.
    
    Este teste é particularmente desafiante para o solver, pois avalia a capacidade
    do algoritmo de manter a precisão durante um crescimento exponencial extremo da
    densidade de nêutrons (variando de 1.0 até ordens de grandeza de 10^89 em 100s).
    """
    print("\n" + "="*80)
    print(f"{'BENCHMARK GANAPOL (2013) - TABELA 1c (STEP $1.0)':^80}")
    print("="*80)

    # ==========================================================================
    # 1. DEFINIÇÃO DOS PARÂMETROS CINÉTICOS
    # ==========================================================================
    # Parâmetros cinéticos correspondentes ao "Thermal Reactor I" (Tabela 1a - Ganapol).
    lambdas = [0.0127, 0.0317, 0.115, 0.311, 1.40, 3.87]
    betas   = [0.000285, 0.0015975, 0.001410, 0.0030525, 0.00096, 0.000195]
    gen_time = 5.0e-4 # Tempo de geração [s]

    config = ReactorPhysics(lambdas, betas, gen_time)
    
    # Cálculo da fração total de nêutrons atrasados (necessário para conversão $ -> absoluta)
    beta_total = sum(betas)

    # ==========================================================================
    # 2. DEFINIÇÃO DA FUNÇÃO DE REATIVIDADE
    # ==========================================================================
    # Inserção de reatividade constante (Step) de $1.0 (Condição Crítico-Pronto).
    rho_step_dollars = 1.0
    rho_abs = rho_step_dollars * beta_total

    def rho_func_step(t, state):
        """
        Retorna a reatividade constante para o transiente em degrau.
        Assinatura padronizada exigida pelo MultiPhysicsSolver.
        """
        return rho_abs

    # Instancia o solver numérico acoplado à física do reator
    solver = MultiPhysicsSolver(config, rho_func_step)

    # ==========================================================================
    # 3. DADOS DE REFERÊNCIA E CONFIGURAÇÃO TEMPORAL
    # ==========================================================================
    # Dados extraídos diretamente da Tabela 1c (Ganapol, 2013).
    # Formato: (Tempo [s], Densidade de Nêutrons Normalizada [-])
    reference_data = [
        (0.1,   2.5157661414043723e+00),
        (0.5,   1.0362533810640215e+01),
        (1.0,   3.2183540945534212e+01),
        (10.0,  3.2469788980305281e+09),
        (100.0, 2.5964846465508731e+89)
    ]

    print(f"{'Tempo (s)':<10} | {'N (Calculado)':<25} | {'N (Referência)':<25} | {'Erro Relativo'}")
    print("-" * 80)

    # Extração exclusiva dos instantes de tempo para avaliação exata do solver
    eval_times = [row[0] for row in reference_data]

    # ==========================================================================
    # 4. EXECUÇÃO DA SIMULAÇÃO NUMÉRICA
    # ==========================================================================
    # O método 'Radau' (Runge-Kutta implícito) é selecionado devido à sua 
    # estabilidade incondicional (L-stable) e robustez inata para lidar com 
    # sistemas rigidamente acoplados (stiff) submetidos a explosões exponenciais.
    sol = solver.solve([0, 100], t_eval=eval_times, method='Radau', rtol=1e-12, atol=1e-14)

    # ==========================================================================
    # 5. ANÁLISE DE RESULTADOS E COMPARAÇÃO
    # ==========================================================================
    for i, (t_ref, n_ref) in enumerate(reference_data):
        n_calc = sol.y[0][i]
        
        # Cálculo do erro relativo para validação da precisão e consistência numérica
        erro = abs((n_calc - n_ref) / n_ref)
        
        print(f"{t_ref:<10.1f} | {n_calc:<25.18e} | {n_ref:<25.18e} | {erro:.4e}")

    print("-" * 80)

if __name__ == "__main__":
    problema_2()