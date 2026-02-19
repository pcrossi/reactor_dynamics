# -*- coding: utf-8 -*-
"""
Módulo: problema_4.py
Descrição: Script de validação (Benchmark) conforme Ganapol (2013), Seção 4.2.1.
Cenário: Reator Térmico III com inserção de reatividade em degrau e realimentação adiabática.
Objetivo: Validar o acoplamento multi-física (Neutrônica + Energia) e a resposta do solver
          a mecanismos de feedback negativo de temperatura.
"""

import numpy as np
import matplotlib.pyplot as plt
from reactor_dynamics import ReactorPhysics, MultiPhysicsSolver

def problema_4():
    """
    Executa o Benchmark de Ganapol (2013) para o caso de realimentação adiabática.
    
    O modelo físico acopla a cinética pontual a uma equação de energia simplificada:
    dW/dt = n(t), onde a reatividade é dada por rho(t) = rho_0 - B * W(t).
    
    Este teste avalia:
    1. A precisão da integração temporal acoplada (Tabela 4a).
    2. A capacidade de capturar picos de potência limitados pelo feedback (Tabela 4c).
    3. O comportamento assintótico em escala log-log (Figura 3).
    """
    print("\n" + "="*80)
    print(f"{'BENCHMARK GANAPOL (2013) - STEP W/ ADIABATIC FEEDBACK':^80}")
    print("="*80)

    # ==========================================================================
    # 1. DEFINIÇÃO DOS PARÂMETROS FÍSICOS
    # ==========================================================================
    # Parâmetros cinéticos do "Thermal Reactor III" (Tabela 4b - Ganapol)
    lambdas = [0.0124, 0.0305, 0.111, 0.301, 1.13, 3.00]
    betas   = [0.00021, 0.00141, 0.00127, 0.00255, 0.00074, 0.00027]
    gen_time = 5.0e-5 # Tempo de geração [s]
    
    config = ReactorPhysics(lambdas, betas, gen_time)
    beta_total = sum(betas)
    
    # Coeficiente de Realimentação de Temperatura (Feedback) [unid. inversa de energia]
    B_coeff = 2.5e-6

    # ==========================================================================
    # 2. MODELO DE ENERGIA (VARIÁVEL DE ESTADO ADICIONAL)
    # ==========================================================================
    # Define a taxa de variação da energia acumulada W.
    # Equação: dW/dt = n(t) (Potência/Densidade de Nêutrons)
    def deriv_W(t, state):
        return state['n']

    # ==========================================================================
    # PARTE A: VALIDAÇÃO DA DENSIDADE TEMPORAL (TABELA 4a)
    # ==========================================================================
    print(f"\n---> [A] Validação Numérica: Tabela 4a (Evolução Temporal)")
    print(f"{'Rho($)':<6} | {'Tempo(s)':<8} | {'N (Calculado)':<15} | {'N (Ref)':<15} | {'Erro Rel.'}")
    print("-" * 75)
    
    # Casos de teste: Rho ($) -> Lista de tuplas (Tempo, N_referencia)
    cases_4a = {
        1.0: [(10.0, 132.038), (50.0, 12.779), (100.0, 3.550)],
        1.5: [(10.0, 107.911), (50.0, 10.890), (100.0, 2.966)],
        2.0: [(10.0, 103.380), (50.0, 10.318), (100.0, 2.755)]
    }
    
    for rho_doll, points in cases_4a.items():
        # Conversão de Dólares para Reatividade Absoluta
        rho_abs = rho_doll * beta_total
        
        # Definição da função de reatividade acoplada: rho(t) = rho_step - B * W(t)
        # Utiliza closure para capturar rho_abs e B_coeff do escopo local
        def rho_func_feedback(t, state):
            return rho_abs - B_coeff * state['W']
        
        # Inicialização do Solver Multi-Física
        solver = MultiPhysicsSolver(config, rho_func_feedback)
        
        # Adição da variável de estado 'W' (Energia) ao sistema
        # Inicializa W(0) = 0.0
        solver.add_variable('W', initial=0.0, derivative_func=deriv_W)
        
        # Definição dos pontos de verificação
        eval_times = [p[0] for p in points]
        
        # Execução da simulação
        # Tolerâncias ajustadas para garantir precisão compatível com o benchmark
        sol = solver.solve([0, 100], t_eval=eval_times, method='Radau', rtol=1e-10, atol=1e-12)
        
        # Comparação dos resultados
        for i, (t_ref, n_ref) in enumerate(points):
            n_calc = sol.y[0][i]
            err = abs((n_calc - n_ref)/n_ref)
            print(f"{rho_doll:<6.1f} | {t_ref:<8.1f} | {n_calc:<15.5e} | {n_ref:<15.5e} | {err:.2e}")

    # ==========================================================================
    # PARTE B: ANÁLISE DE PICOS DE POTÊNCIA (TABELA 4c)
    # ==========================================================================
    print(f"\n---> [B] Validação de Extremos: Tabela 4c (Busca de Picos)")
    print(f"{'Rho($)':<6} | {'Tp Calc':<12} | {'Tp Ref':<12} | {'Np Calc':<12} | {'Np Ref':<12} | {'Erro Np'}")
    print("-" * 80)
    
    # Casos: (Rho [$], Tempo Pico Ref, N Pico Ref)
    cases_4c = [
        (0.1, 108.9694, 2.11049),
        (0.5, 28.29469, 45.75243),
        (1.0, 0.953477, 807.8681),
        (1.5, 0.168289, 43024.61),
        (2.0, 0.098390, 167845.7)
    ]
    
    for rho_doll, tp_ref, np_ref in cases_4c:
        rho_abs = rho_doll * beta_total
        
        # Definição local da função de reatividade para o caso específico
        def rho_func_peak(t, state, r=rho_abs): 
            return r - B_coeff * state['W']
            
        sys_peak = MultiPhysicsSolver(config, rho_func_peak)
        sys_peak.add_variable('W', initial=0.0, derivative_func=deriv_W)
        
        # Definição de evento para localização de máximos locais (Root Finding)
        # Monitora quando a derivada dN/dt cruza zero no sentido negativo
        def peak_event(t, y):
            derivs = sys_peak._rhs(t, y)
            return derivs[0] # Retorna dN/dt
        
        peak_event.direction = -1  # Apenas cruzamento descendente (máximo)
        peak_event.terminal = True # Interrompe a simulação ao encontrar o pico
        
        # Ajuste dinâmico do tempo máximo de simulação baseado na magnitude da reatividade
        t_max = 200.0 if rho_doll <= 0.5 else 2.0
        
        sol_peak = sys_peak.solve([0, t_max], events=peak_event, method='Radau', rtol=1e-12, atol=1e-12)
        
        # Extração e comparação dos dados do evento
        if sol_peak.t_events and len(sol_peak.t_events[0]) > 0:
            tp = sol_peak.t_events[0][0]
            np_val = sol_peak.y_events[0][0][0]
        else:
            tp, np_val = 0.0, 0.0
            
        err = abs((np_val - np_ref)/np_ref)
        print(f"{rho_doll:<6.1f} | {tp:<12.5f} | {tp_ref:<12.5f} | {np_val:<12.5e} | {np_ref:<12.5e} | {err:.2e}")

    # ==========================================================================
    # PARTE C: VISUALIZAÇÃO GRÁFICA (FIGURA 3)
    # ==========================================================================
    print(f"\n---> [C] Geração da Figura 3 (Escala Log-Log)")
    
    rhos_plot = [0.1, 0.2, 0.5, 1.0, 1.2, 1.5, 2.0]
    
    plt.figure(figsize=(10, 6))
    
    for rho_doll in rhos_plot:
        rho_abs = rho_doll * beta_total
        
        def rho_func_plot(t, state, r=rho_abs):
            return r - B_coeff * state['W']
            
        sys_plot = MultiPhysicsSolver(config, rho_func_plot)
        sys_plot.add_variable('W', 0.0, deriv_W)
        
        t_end = 2000.0 if rho_doll < 0.5 else 100.0
        
        # Simulação com passo adaptativo automático do SciPy para plotagem suave
        sol = sys_plot.solve([0, t_end], method='Radau', rtol=1e-6)
        
        plt.loglog(sol.t, sol.y[0], label=f'$\\rho_0={rho_doll}$')

    plt.title('Fig 3. Densidade de Nêutrons: Inserção Degrau com Feedback Adiabático')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Densidade de Nêutrons (Normalizada)')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.xlim(1e-7, 1e4)
    
    plt.tight_layout()
    plt.show()
    print("     Gráfico gerado com sucesso.")
    print("-" * 80)

if __name__ == "__main__":
    problema_4()