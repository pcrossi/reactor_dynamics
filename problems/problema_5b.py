# -*- coding: utf-8 -*-
"""
Modulo: problema_5b.py
Descrição: Script de validação (Benchmark) conforme Ganapol (2013), Seção 4.2.2.
Cenário: Inserção de Reatividade em Rampa com Realimentação de Temperatura Adiabática.
Método: Utiliza o algoritmo de passo adaptativo baseado na física (Physics-Based Adaptive Stepping).

Objetivo: Demonstrar a capacidade do solver de ajustar automaticamente o passo temporal 
          durante excursões de potência severas (Prompt Critical), utilizando a funcionalidade
          nativa de interpolação densa (dense_output) da classe atualizada.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from reactor_dynamics import ReactorPhysics, MultiPhysicsSolver

def problema_5b():
    """
    Executa o Benchmark de Ganapol (2013) utilizando controlo de passo adaptativo.
    
    A detecção precisa dos picos utiliza a funcionalidade 'dense_output' integrada
    na classe MultiPhysicsSolver, permitindo otimização escalar contínua
    sem necessidade de interpolação externa manual.
    """
    print("\n" + "="*80)
    print(f"{'BENCHMARK GANAPOL (2013) - RAMP W/ ADIABATIC FEEDBACK (ADAPTIVE)':^80}")
    print("="*80)

    # ==========================================================================
    # 1. CONFIGURAÇÃO DOS PARÂMETROS FÍSICOS
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
        
        # Função de reatividade local
        def rho_func_ramp_fb(t, state, a=a_abs, B=B_val):
            return (a * t) - (B * state['W'])

        sys_peak = MultiPhysicsSolver(config, rho_func_ramp_fb)
        sys_peak.add_variable('W', initial=0.0, derivative_func=deriv_W)

        # Definição do horizonte de simulação
        t_max = 15.0 if a_abs < 0.005 else 2.5
        
        # ======================================================================
        # EXECUÇÃO DO SOLVER E OTIMIZAÇÃO DE PRECISÃO NUMÉRICA
        # ======================================================================
        # Para capturar o pico de uma excursão pronta-crítica com erro na ordem 
        # de 10^-6, duas restrições rigorosas são aplicadas à heurística adaptativa:
        #
        # 1. interp_kind='cubic': Restaura a reconstrução geométrica do pico. 
        #    A interpolação linear cortaria o topo da curva parabolica natural.
        # 2. safety_factor=0.01: Proximo ao pico, a derivada zera (dn/dt = 0) e o 
        #    tempo característico (Tau) tende ao infinito. Este fator de segurança
        #    apertado força o solver a dar passos minúsculos, amostrando a crista
        #    da onda densamente para que o spline cúbico tenha pontos suficientes.
        # ======================================================================
        # Teste e veja a diferença:
        # sol = sys_peak.solve(
        #     [0, t_max], 
        #     adaptive=True,      
        #     method='Radau', 
        #     rtol=1e-11, 
        #     atol=1e-13, 
        #     dense_output=True   # Essencial para a otimização abaixo
        # )
        # Compare com: 

        sol = sys_peak.solve(
            [0, t_max], 
            adaptive=True,      
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13, 
            dense_output=True,
            interp_kind='cubic',  
            safety_factor=0.01    
        )
        
        # Refinamento da Localização do Pico via Otimização Escalar
        t_scan = np.linspace(0, sol.t[-1], 50000) 
        y_scan = sol.sol(t_scan)[0] 
        
        idx_scan_max = np.argmax(y_scan)
        t_approx = t_scan[idx_scan_max]
        
        # Função objetivo para maximização (negativo da densidade)
        def neg_N(t): 
            return -sol.sol(t)[0]
        
        # Janela de busca limitada ao redor da estimativa inicial
        bounds = (max(0, t_approx - 0.1), min(t_max, t_approx + 0.1))
        
        try:
            res = minimize_scalar(neg_N, bounds=bounds, method='bounded')
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
            def rho_func_plot(t, state, a=a_abs, B=B_val): 
                return (a * t) - (B * state['W'])
            
            sys_plot = MultiPhysicsSolver(config, rho_func_plot)
            sys_plot.add_variable('W', initial=0.0, derivative_func=deriv_W)
            
            # Utiliza interp_kind='cubic' para garantir curvas visualmente perfeitas
            sol = sys_plot.solve(
                [0, 12.0], 
                adaptive=True, 
                rtol=1e-7, 
                interp_kind='cubic'
            )
            
            plt.semilogy(sol.t, sol.y[0], linestyle=styles[B_val], linewidth=1.5, color='k', alpha=0.8)
            
            # Adiciona anotações textuais para a série B=10^-11
            if B_val == 1.0e-11:
                idx_peak = np.argmax(sol.y[0])
                if sol.y[0][idx_peak] > 10:
                    plt.text(sol.t[idx_peak], sol.y[0][idx_peak]*2, f"{a_abs}", fontsize=10)

    plt.title('Fig 4. Compensated reactor shutdown (Adaptive Step Control)')
    plt.xlabel('Time (s)')
    plt.ylabel('Neutron Density')
    plt.grid(True, which="both", alpha=0.3)
    plt.ylim(1e-1, 1e16)
    plt.xlim(0, 12)
    
    # Configuração da legenda
    from matplotlib.lines import Line2D
    lines = [Line2D([0], [0], color='k', ls='-'), Line2D([0], [0], color='k', ls='--')]
    plt.legend(lines, ['$B=10^{-11}$', '$B=10^{-13}$'], loc='lower right')
    
    plt.tight_layout()
    plt.show()
    print("     Gráfico gerado com sucesso.")
    print("-" * 80)

if __name__ == "__main__":
    problema_5b()