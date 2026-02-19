# -*- coding: utf-8 -*-
"""
Módulo: problema_6.py
Descrição: Script de validação (Benchmark) conforme Ganapol (2013) - Secção 4.2.3.
Cenário: Cinética Simples e Termohidráulica (SKINATH - Simple Kinetics and Thermal Hydraulics).
Adaptação: Utiliza a classe reactor_dynamics v1.7 com interpolação nativa (dense_output).

Este script realiza uma simulação de longo prazo (4000 minutos) de um reator 
arrefecido a ar, demonstrando o acoplamento neutrônico-térmico e a eficácia 
do controlo rigoroso de passo para transientes oscilatórios fortemente amortecidos.
"""

import numpy as np
import matplotlib.pyplot as plt
from reactor_dynamics import ReactorPhysics, MultiPhysicsSolver

def problema_6():
    """
    Executa o Benchmark SKINATH (Ganapol, 2013).
    
    A simulação abrange 4000 minutos de operação. O integrador numérico deve ser 
    capaz de capturar as oscilações iniciais rápidas (rigidez elevada) e, 
    simultaneamente, acelerar o passo de tempo durante a fase de estabilização 
    assintótica de longo prazo para garantir a viabilidade computacional.
    
    Saídas:
    - Comparação tabular de alta precisão utilizando o interpolador contínuo (Tabela 6b).
    - Gráfico triplo reproduzindo a Figura 5a (Evolução da Potência, Temperatura e Reatividade).
    """
    print("\n" + "="*80)
    print(f"{'BENCHMARK GANAPOL (2013) - SKINATH (PROBLEMA 6)':^80}")
    print("="*80)

    # ==========================================================================
    # 1. CONFIGURAÇÃO DOS PARÂMETROS FÍSICOS
    # ==========================================================================
    # Parâmetros Cinéticos (Thermal Reactor IV - Tabela 6a)
    lambdas = [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01]
    betas   = [0.00022, 0.00142, 0.00127, 0.00257, 0.00075, 0.00027]
    gen_time = 5.0e-5  # Tempo de geração de nêutrons prontos [s]
    beta_total = sum(betas)
    
    config = ReactorPhysics(lambdas, betas, gen_time)

    # Parâmetros Termohidráulicos (Modelo SKINATH)
    # Ref: Ganapol (2013), Equações 13-15
    Cp = 1.301e4      # Capacidade calorífica total do núcleo (J/°C)
    Tc = 20.0         # Temperatura do refrigerante/ambiente (°C)
    
    # Constante de dissipação de calor K = a * h * d^0.75
    # Valores assumidos: a=17.52, h=0.23m, d=0.20m
    K_loss = 17.52 * 0.23 * (0.20**0.75) 
    
    # Coeficientes de Reatividade do Núcleo
    rho0_doll = 0.043        # Reatividade externa inserida inicialmente ($)
    alpha_T_doll = -0.00306  # Coeficiente de realimentação térmica de reatividade ($/°C)

    # ==========================================================================
    # 2. DEFINIÇÃO DO MODELO ACOPLADO (MULTI-FÍSICA)
    # ==========================================================================
    
    # Função de Reatividade (Acoplamento Térmico -> Neutrônico)
    def rho_func(t, state):
        # rho($) = rho0 + alpha * (T(t) - Tc)
        rho_doll = rho0_doll + alpha_T_doll * (state['T'] - Tc)
        # O solver exige que a reatividade seja fornecida em unidades absolutas (delta k/k)
        return rho_doll * beta_total

    # Equação Diferencial da Temperatura (Balanço de Energia do Núcleo)
    def deriv_T(t, state):
        N = state['n'] # Potência térmica gerada (Watts)
        T = state['T'] # Temperatura média do combustível (°C)
        W = T - Tc     # Gradiente de temperatura em relação ao ambiente
        
        # Lei de arrefecimento não-linear (combinação de convecção natural e forçada)
        if T > Tc:
            Ploss = K_loss * (W / T)**0.25 * W
        else:
            Ploss = 0.0
            
        # Dinâmica térmica: dT/dt = (Potência Gerada - Potência Dissipada) / Capacidade Calorífica
        return (N - Ploss) / Cp

    # Instanciação do Solver Multi-Física
    solver = MultiPhysicsSolver(config, rho_func)
    
    # Registo da variável de estado 'T' no sistema diferencial
    # O valor inicial estabelece o reator em equilíbrio térmico com o ambiente
    solver.add_variable('T', initial=Tc, derivative_func=deriv_T)

    # ==========================================================================
    # 3. EXECUÇÃO DA SIMULAÇÃO NUMÉRICA
    # ==========================================================================
    t_final_min = 4000
    t_final_sec = t_final_min * 60.0

    print(f"Iniciando simulação de longo prazo: {t_final_min} minutos...")
    
    # OTIMIZAÇÃO: Para um transiente de 4000 minutos, o uso do integrador 'Radau' 
    # nativo do SciPy (adaptive=False) garante a resolução do modelo rígido em segundos, 
    # tirando partido das rotinas compiladas em Fortran.
    print("Modo: Integrador Implícito Robusto (SciPy Radau)")

    sol = solver.solve(
        (0, t_final_sec), 
        n0=0.01,            # Condição inicial: Potência de 0.01 W
        driver='scipy',
        adaptive=False,     # Desativa a heurística manual para maximizar a performance a longo prazo
        method='Radau',     # L-stable, ideal para a rigidez extrema do problema SKINATH
        rtol=1e-10,         # Tolerância restrita para manter a integridade assintótica
        atol=1e-14,
        dense_output=True   # Gera o interpolador (.sol) essencial para a extração dos pontos da tabela
    )

    # --- EXEMPLO: MODO ADAPTATIVO MANUAL (DIDÁTICO) ---
    # Para fins de análise profunda dos primeiros segundos do transiente (Prompt Jump),
    # o controlo heurístico pode ser ativado com as restrições abaixo:
    #
    # sol = solver.solve(
    #     (0, t_final_sec), 
    #     n0=0.01,
    #     driver='scipy',
    #     adaptive=True,       # Ativa o ciclo de integração manual
    #     safety_factor=0.2,   # Passo conservador (20% da constante de tempo física)
    #     max_step=60.0,       # OBRIGATÓRIO: Impede que o passo exceda 1 minuto na fase de equilíbrio
    #     dense_output=True
    # )    
        
    print("Simulação concluída com sucesso.\n")

    # ==========================================================================
    # 4. COMPARAÇÃO TABULAR DE ALTA PRECISÃO (TABELA 6b)
    # ==========================================================================
    # Pontos de calibração extraídos de Ganapol (2013)
    # Estrutura: Tempo (min) -> (Potência [W], Reatividade [$], Temperatura [°C])
    ref_data = {
        0:    (1.0000e-2,  0.043,       20.00),
        1:    (1.3907e-2,  0.0429998,   20.00),
        10:   (1.0331e-1,  0.0429941,   20.00),
        100:  (7.9461e-1, -0.029495,    43.69),
        250:  (2.1187e-3,  0.0063885,   31.96),
        500:  (9.8189e-1, -0.0048251,   35.63),
        750:  (4.0335e0,  -0.0018573,   34.66),
        1000: (7.6810e0,   0.0013791,   33.60),
        2000: (13.5885e0, -0.0002105,   34.12),
        4000: (13.5721e0,  0.0000006,   34.05)
    }

    print(f"---> [A] Comparação de Resultados: Tabela 6b (Benchmark SKINATH)")
    print(f"{'t(min)':<7} | {'N(Ref) W':<11} | {'N(Calc) W':<11} | {'q$(Ref)':<11} | {'q$(Calc)':<11} | {'T(Ref) C':<9} | {'T(Calc) C':<9}")
    print("-" * 95)

    # Conversão do vetor de tempos alvo para segundos (unidade do integrador)
    target_mins = sorted(ref_data.keys())
    target_secs = np.array(target_mins) * 60.0
    
    # Interpolação contínua da matriz de resultados (.sol)
    # y_interp conterá as variáveis de estado avaliadas nos instantes exatos da referência
    y_interp = sol.sol(target_secs)
    
    n_calc_vec = y_interp[0]  # Índice 0: Densidade de Nêutrons / Potência
    t_calc_vec = y_interp[-1] # Último Índice: Temperatura do Núcleo

    for i, t_min in enumerate(target_mins):
        n_val = n_calc_vec[i]
        t_val = t_calc_vec[i]
        
        # Reconstrução da reatividade instantânea em Dólares para verificação
        q_val_doll = rho0_doll + alpha_T_doll * (t_val - Tc)
        
        n_ref, q_ref, t_ref = ref_data[t_min]
        
        print(f"{t_min:<7.0f} | {n_ref:<11.4e} | {n_val:<11.4e} | {q_ref:<11.4e} | {q_val_doll:<11.4e} | {t_ref:<9.2f} | {t_val:<9.2f}")
    print("-" * 95)

    # ==========================================================================
    # 5. REPRESENTAÇÃO GRÁFICA MULTI-VARIÁVEL (FIGURA 5a)
    # ==========================================================================
    print(f"\n---> [B] Geração da Figura 5a (Dinâmica Oscilatória Acoplada)")
    
    # Geração de malha hiper-densa (5000 pontos) para garantir a integridade visual das oscilações
    t_plot_sec = np.linspace(0, t_final_sec, 5000)
    t_plot_min = t_plot_sec / 60.0
    
    # Extração dos dados interpolados
    y_plot = sol.sol(t_plot_sec)
    n_plot = y_plot[0]
    t_vals_plot = y_plot[-1]
    q_vals_plot_doll = rho0_doll + alpha_T_doll * (t_vals_plot - Tc)
    
    # Configuração do painel triplo
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Subplot 1: Dinâmica da Potência (Escala Logarítmica)
    ax1.semilogy(t_plot_min, n_plot, 'k-', linewidth=1.5)
    ax1.set_ylabel('Power (W)', fontsize=11)
    ax1.set_title('Fig 5a. SKINATH Simulation (0 - 4000 min)', fontsize=12, fontweight='bold')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Subplot 2: Dinâmica da Temperatura (Escala Linear)
    ax2.plot(t_plot_min, t_vals_plot, 'k-', linewidth=1.5)
    ax2.set_ylabel('Temperature (°C)', fontsize=11)
    ax2.grid(True, linestyle='--', linewidth=0.5)
    
    # Subplot 3: Dinâmica da Reatividade (Escala Linear)
    ax3.plot(t_plot_min, q_vals_plot_doll, 'k-', linewidth=1.5)
    ax3.set_ylabel('Reactivity ($)', fontsize=11)
    ax3.set_xlabel('Time (min)', fontsize=11)
    ax3.grid(True, linestyle='--', linewidth=0.5)
    
    plt.xlim(0, 4000)
    plt.tight_layout()
    plt.show()
    print("     Figura gerada com sucesso.")
    print("="*80)

if __name__ == "__main__":
    problema_6()