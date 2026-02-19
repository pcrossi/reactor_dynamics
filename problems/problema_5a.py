# -*- coding: utf-8 -*-
"""
Modulo: problema_5a.py
Descrição: Script de validação (Benchmark) conforme Ganapol (2013), Seção 4.2.2.
Cenário: Inserção de Reatividade em Rampa com Realimentação de Temperatura Adiabática.
Objetivo: Validar a capacidade do solver de capturar picos de potência estreitos e de alta 
          magnitude em transientes iniciados por rampas, considerando diferentes taxas de 
          inserção (a) e coeficientes de realimentação (B).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from reactor_dynamics import ReactorPhysics, MultiPhysicsSolver

def problema_5a():
    """
    Executa o Benchmark de Ganapol (2013) para o caso de Rampa com Feedback.
    
    Este teste requer um controle rigoroso do passo de tempo (time-stepping) 
    para garantir a captura precisa dos picos de potência, que podem atingir 
    ordens de magnitude de 10^13 em intervalos de microssegundos.
    """
    print("\n" + "="*80)
    print(f"{'BENCHMARK GANAPOL (2013) - RAMP W/ ADIABATIC FEEDBACK':^80}")
    print("="*80)

    # ==========================================================================
    # 1. DEFINIÇÃO DOS PARÂMETROS FÍSICOS
    # ==========================================================================
    lambdas = [0.0124, 0.0305, 0.111, 0.301, 1.13, 3.00]
    betas   = [0.00021, 0.00141, 0.00127, 0.00255, 0.00074, 0.00027]
    gen_time = 5.0e-5 # Tempo de geração [s]
    
    config = ReactorPhysics(lambdas, betas, gen_time)

    # Modelo de Energia Adiabático: dW/dt = n(t)
    def deriv_W(t, state):
        return state['n']

    # ==========================================================================
    # PARTE A: VALIDAÇÃO DE PICOS (TABELA 5)
    # ==========================================================================
    print(f"\n---> [A] Validação de Extremos: Tabela 5 (Picos de Potência)")
    print(f"{'a(s^-1)':<8} | {'B':<10} | {'Tp Calc':<12} | {'Tp Ref':<12} | {'Np Calc':<13} | {'Np Ref':<13} | {'Erro Np'}")
    print("-" * 90)

    # Casos: (Taxa de Rampa a, Coef. Feedback B, Tempo Pico Ref, N Pico Ref)
    cases_table_5 = [
        (0.1,   1.0e-11, 0.22466, 2.42038e+11),
        (0.1,   1.0e-13, 0.23890, 2.89867e+13),
        (0.01,  1.0e-11, 1.10608, 2.01235e+10),
        (0.01,  1.0e-13, 1.15515, 2.49118e+12),
        (0.003, 1.0e-11, 2.91048, 5.11416e+09),
        (0.003, 1.0e-13, 3.00760, 6.53447e+11),
        (0.001, 1.0e-11, 7.48877, 1.27408e+09),
        (0.001, 1.0e-13, 7.68359, 1.72101e+11),
    ]

    for a_abs, B_val, tp_ref, np_ref in cases_table_5:
        
        # Função de reatividade local: rho(t) = a*t - B*W(t)
        def rho_func_ramp_fb(t, state, a=a_abs, B=B_val):
            return (a * t) - (B * state['W'])

        sys_peak = MultiPhysicsSolver(config, rho_func_ramp_fb)
        sys_peak.add_variable('W', initial=0.0, derivative_func=deriv_W)

        # === ESTRATÉGIA DE CONTROLE DE PASSO ===
        # Ajuste dinâmico dos parâmetros de integração baseado na severidade do transiente.
        # Transientes mais rápidos (maior 'a') exigem passos máximos (max_step) menores.
        if a_abs >= 0.09:      # Taxa severa (0.1 s^-1)
            t_max = 0.6
            max_s = 2e-6       # Passo máximo: 2 microssegundos
            first_s = 1e-14    # Inicialização conservadora
        elif a_abs >= 0.009:   # Taxa alta (0.01 s^-1)
            t_max = 2.0
            max_s = 2e-5       # Passo máximo: 20 microssegundos
            first_s = 1e-12
        elif a_abs >= 0.002:   # Taxa moderada (0.003 s^-1)
            t_max = 5.0
            max_s = 1e-3       
            first_s = 1e-10
        else:                  # Taxa lenta (0.001 s^-1)
            t_max = 15.0
            max_s = np.inf     # Delega a decisão ao solver
            first_s = 1e-8

        # Execução do Solver
        # 'dense_output=True' permite interpolar a solução entre os passos de integração
        sol = sys_peak.solve(
            [0, t_max], 
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13, 
            dense_output=True,
            max_step=max_s,
            first_step=first_s 
        )

        # === LOCALIZAÇÃO PRECISA DO PICO (OTIMIZAÇÃO) ===
        # 1. Varredura densa inicial para encontrar a região aproximada do pico.
        t_scan = np.linspace(0, t_max, 50000) 
        y_scan = sol.sol(t_scan)[0]
        
        idx_scan_max = np.argmax(y_scan)
        t_approx = t_scan[idx_scan_max]
        
        # 2. Refinamento via Otimização Escalar
        # Utiliza o interpolador do solver para maximizar N(t) com alta precisão.
        def neg_N(t): return -sol.sol(t)[0]
        
        # Define janela de busca estreita ao redor da estimativa inicial
        b_min = max(0, t_approx - 0.1) 
        b_max = min(t_max, t_approx + 0.1)
        
        try:
            res = minimize_scalar(neg_N, bounds=(b_min, b_max), method='bounded')
            tp = res.x
            np_val = -res.fun
        except Exception:
            # Fallback seguro caso a otimização falhe
            tp, np_val = t_approx, y_scan[idx_scan_max]

        err_np = abs((np_val - np_ref)/np_ref)
        
        print(f"{a_abs:<8.3f} | {B_val:<10.0e} | {tp:<12.5f} | {tp_ref:<12.5f} | {np_val:<13.5e} | {np_ref:<13.5e} | {err_np:.2e}")

    # ==========================================================================
    # PARTE B: VISUALIZAÇÃO GRÁFICA (FIGURA 4)
    # ==========================================================================
    print(f"\n---> [B] Geração da Figura 4 (Desligamento Compensado)")
    plt.figure(figsize=(10, 7))
    
    ramp_rates = [0.1, 0.01, 0.003, 0.001]
    b_values = [1.0e-11, 1.0e-13]
    styles = {1.0e-11: '-', 1.0e-13: '--'}
    
    for a_abs in ramp_rates:
        for B_val in b_values:
            def rho_func_plot(t, state, a=a_abs, B=B_val): return (a * t) - (B * state['W'])
            
            sys_plot = MultiPhysicsSolver(config, rho_func_plot)
            sys_plot.add_variable('W', initial=0.0, derivative_func=deriv_W)
            
            # Configuração da discretização para plotagem suave
            if a_abs >= 0.09: ms_plot = 1e-4
            elif a_abs >= 0.009: ms_plot = 5e-4
            else: ms_plot = 1e-3
            
            sol = sys_plot.solve([0, 12.0], method='Radau', rtol=1e-6, max_step=ms_plot)
            
            plt.semilogy(sol.t, sol.y[0], linestyle=styles[B_val], linewidth=1.5, color='k', alpha=0.8)
            
            # Adiciona rótulos textuais para os picos da curva sólida
            if B_val == 1.0e-11:
                idx_peak = np.argmax(sol.y[0])
                if sol.y[0][idx_peak] > 10:
                    plt.text(sol.t[idx_peak], sol.y[0][idx_peak]*2, f"{a_abs}", fontsize=10)

    plt.title('Fig 4. Compensated reactor shutdown initiated by ramp (a in $s^{-1}$)')
    plt.xlabel('Time (s)')
    plt.ylabel('Neutron Density')
    plt.grid(True, which="both", alpha=0.3)
    plt.ylim(1e-1, 1e16)
    plt.xlim(0, 12)
    
    # Configuração da legenda manual
    from matplotlib.lines import Line2D
    lines = [Line2D([0], [0], color='k', ls='-'), Line2D([0], [0], color='k', ls='--')]
    plt.legend(lines, ['$B=10^{-11}$', '$B=10^{-13}$'], loc='lower right')
    
    plt.tight_layout()
    plt.show()
    print("     Gráfico gerado com sucesso.")
    print("-" * 80)

if __name__ == "__main__":
    problema_5a()