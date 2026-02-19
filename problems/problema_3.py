# -*- coding: utf-8 -*-
"""
Módulo: problema_3.py
Descrição: Script de validação (Benchmark) conforme Ganapol (2013), Tabela 2b.
Cenário: Reator Térmico II sujeito a uma inserção de reatividade em rampa de 0.1 $/s.
Objetivo: Validar a precisão do solver em transientes com reatividade dependente do tempo
          e transição para o regime supercrítico pronto.
"""

import numpy as np
from reactor_dynamics import ReactorPhysics, MultiPhysicsSolver    

def problema_3():
    """
    Executa o Benchmark de Ganapol (2013) para o caso de inserção em rampa.
    
    O cenário simula a evolução da densidade de nêutrons sob uma taxa de inserção
    constante de 0.1 $/s, cobrindo a transição de subcrítico para supercrítico 
    pronto (atingido em t=10s, quando rho = $1.0).
    """
    print("\n" + "="*80)
    print(f"{'BENCHMARK GANAPOL (2013) - TABLE 2b (RAMP 0.1 $/s)':^80}")
    print("="*80)

    # ==========================================================================
    # 1. DEFINIÇÃO DOS PARÂMETROS CINÉTICOS
    # ==========================================================================
    # Parâmetros correspondentes ao "Thermal Reactor II" (Tabela 2a - Ganapol).
    lambdas = [0.0127, 0.0317, 0.115, 0.311, 1.40, 3.87]
    betas   = [0.000266, 0.001491, 0.001316, 0.002849, 0.000896, 0.000182]
    gen_time = 2.0e-5 # Tempo de geração [s]

    config = ReactorPhysics(lambdas, betas, gen_time)
    
    # Cálculo da fração total de nêutrons atrasados (Beta Efetivo) para conversão de unidades.
    beta_total = sum(betas)

    # ==========================================================================
    # 2. DEFINIÇÃO DA FUNÇÃO DE REATIVIDADE
    # ==========================================================================
    # Inserção de reatividade em rampa: d(rho)/dt = 0.1 $/s.
    # A função converte o valor em dólares para reatividade absoluta:
    # rho_abs(t) = (0.1 * t) * beta_total
    def rho_func_ramp(t, state):
        return 0.1 * t * beta_total

    # Inicialização do solver numérico com acoplamento físico
    solver = MultiPhysicsSolver(config, rho_func_ramp)

    # ==========================================================================
    # 3. DADOS DE REFERÊNCIA E CONFIGURAÇÃO TEMPORAL
    # ==========================================================================
    # Dados extraídos da Tabela 2b (Ganapol, 2013).
    # O benchmark destaca o crescimento exponencial severo após t=10s.
    reference_data = [
        (2.0,   1.338200050e+00),
        (4.0,   2.228441897e+00),
        (6.0,   5.582052449e+00),
        (8.0,   4.278629573e+01),
        (10.0,  4.511636239e+05),  # Limiar de Criticalidade Pronta (rho = 1.0 $)
        (11.0,  1.792213607e+16)   # Excursão Supercrítica Pronta (rho = 1.1 $)
    ]

    print(f"{'Tempo (s)':<10} | {'N (Calculado)':<25} | {'N (Referência)':<25} | {'Erro Relativo'}")
    print("-" * 80)

    # Extração dos pontos de avaliação
    eval_times = [row[0] for row in reference_data]

    # ==========================================================================
    # 4. EXECUÇÃO DA SIMULAÇÃO
    # ==========================================================================
    # Utiliza-se o método implícito 'Radau' devido à rigidez (stiffness) do sistema,
    # especialmente durante a excursão supercrítica pronta onde a escala de tempo
    # é dominada pelo tempo de geração (gen_time).
    sol = solver.solve([0, 11], t_eval=eval_times, method='Radau', rtol=1e-12, atol=1e-14)

    # ==========================================================================
    # 5. ANÁLISE COMPARATIVA
    # ==========================================================================
    for i, (t_ref, n_ref) in enumerate(reference_data):
        n_calc = sol.y[0][i]
        
        # Calculo do erro relativo para verificação de conformidade
        erro = abs((n_calc - n_ref) / n_ref)
        
        print(f"{t_ref:<10.1f} | {n_calc:<25.18e} | {n_ref:<25.18e} | {erro:.4e}")

    print("-" * 80)
    print("Nota: O aumento de ordens de magnitude entre 10s e 11s caracteriza o regime supercrítico pronto.")

if __name__ == "__main__":
    problema_3()