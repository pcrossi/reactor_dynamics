"""
Módulo: problema_1.py
Descrição: Script de validação (Benchmark) baseado em Ganapol (2013).
Aplicação: Cinética Pontual com inserção de reatividade senoidal em Reator Rápido.
"""

import numpy as np
import matplotlib.pyplot as plt
from reactor_dynamics import ReactorPhysics, MultiPhysicsSolver

def problema_1():
    """
    Executa o Benchmark de Ganapol (2013) para um reator rápido sujeito a 
    uma inserção de reatividade senoidal.
    
    Etapas:
    1. Definição de parâmetros cinéticos rígidos (stiff).
    2. Validação numérica pontual (Tabela 3a do paper original).
    3. Detecção de eventos de pico de potência (Tabela 3b).
    4. Geração de gráficos de resposta temporal (Figuras 2a e 2b).
    """
    print("\n" + "="*80)
    print(f"{'BENCHMARK GANAPOL (2013) - SENOIDAL / FAST REACTOR':^80}")
    print("="*80)

    # ==========================================================================
    # 1. CONFIGURAÇÃO DOS PARÂMETROS FÍSICOS
    # ==========================================================================
    # Parâmetros cinéticos representativos de um reator rápido.
    # O tempo de geração (1.0e-8 s) caracteriza um sistema de equações diferenciais 
    # ordinárias (EDO) rigidamente acoplado (stiff system).
    config = ReactorPhysics(
        lambdas=[0.077],   # Constante de decaimento (1 grupo) [s^-1]
        betas=[0.0079],    # Fração total de nêutrons atrasados [-]
        gen_time=1.0e-8    # Tempo de geração de nêutrons prontos [s]
    )

    # Parâmetros da função de Reatividade
    rho0_abs = 0.0053333   # Amplitude da oscilação [delta k/k]
    T_case    = 50.0       # Período da oscilação [s]
    
    def rho_func_50(t, state):
        """
        Define a reatividade dependente do tempo: rho(t) = rho0 * sin(pi * t / T).
        """
        return rho0_abs * np.sin(np.pi * t / T_case)

    # Inicialização do solver para o caso base (T=50s)
    solver = MultiPhysicsSolver(config, rho_func_50)

    # ==========================================================================
    # PARTE A: VALIDAÇÃO DA DENSIDADE TEMPORAL (TABELA 3a)
    # ==========================================================================
    print(f"\n---> [A] Validação Numérica: Tabela 3a (T=50s)")
    
    # Pontos de discretização temporal para comparação com a referência
    check_points = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    
    # Dados de referência (Ganapol, 2013)
    ref_table = [
        2.065383519e+00, 8.854133921e+00, 4.064354222e+01,
        6.135607517e+01, 4.610628770e+01, 2.912634840e+01,
        1.895177042e+01, 1.393829211e+01, 1.253353406e+01,
        1.544816514e+01
    ]

    print(f"{'Tempo (s)':<10} | {'N (Calculado)':<25} | {'N (Referência)':<25} | {'Erro Rel.'}")
    print("-" * 80)

    # Execução da simulação.
    # O método 'Radau' (implícito) é selecionado devido à rigidez do sistema.
    # Tolerâncias rtol/atol ajustadas para garantir precisão compatível com o benchmark.
    sol = solver.solve([0, 100], t_eval=check_points, method='Radau', rtol=1e-12, atol=1e-14)

    # Comparação ponto a ponto
    for i, n_ref in enumerate(ref_table):
        n_calc = sol.y[0][i] 
        t_val = sol.t[i]
        err = abs((n_calc - n_ref)/n_ref)
        print(f"{t_val:<10.1f} | {n_calc:<25.18e} | {n_ref:<25.18e} | {err:.2e}")

    # ==========================================================================
    # PARTE B: ANÁLISE DE PICOS (TABELA 3b)
    # ==========================================================================
    print(f"\n---> [B] Validação de Extremos: Tabela 3b (Busca de Picos)")
    print(f"{'T (s)':<6} | {'Tp Calc':<10} | {'Tp Ref':<10} | {'Np Calc':<10} | {'Np Ref':<10} | {'Erro Np(%)'}")
    print("-" * 80)

    # Tuplas de teste: (Período, Rho0, Tempo de Pico Ref, N de Pico Ref)
    cases_3b = [
        (50.0,  0.0053333, 39.10712,  61.53015),
        (150.0, 0.0032327, 137.3198,  95.81150),
        (250.0, 0.0023193, 237.1265, 113.4679)
    ]

    for T_val, r_abs, tp_ref, np_ref in cases_3b:
        
        # Definição local da função de reatividade para o caso específico
        def rho_peak_func(t, state, T=T_val, r=r_abs):
            return r * np.sin(np.pi * t / T)

        sys_peak = MultiPhysicsSolver(config, rho_peak_func)

        # Definição do evento para localização de raízes (Root Finding)
        # O objetivo é identificar o instante onde dN/dt = 0.
        def peak_event(t, y):
            # Acessa o método interno _rhs para calcular a derivada no instante t
            vals = sys_peak._rhs(t, y)
            return vals[0] # Retorna dN/dt
        
        peak_event.direction = -1  # Apenas cruzamento descendente (máximo local)
        peak_event.terminal = True # Interrompe a integração ao encontrar o evento
        
        # Simulação até T_val (cobre o primeiro semi-ciclo onde ocorre o pico)
        sol_peak = sys_peak.solve([0, T_val], events=peak_event, method='Radau', rtol=1e-12, atol=1e-14)

        # Extração dos resultados do evento detectado
        if sol_peak.t_events and len(sol_peak.t_events[0]) > 0:
            tp = sol_peak.t_events[0][0]       # Tempo do pico
            np_calc = sol_peak.y_events[0][0][0] # Amplitude do pico
        else:
            tp, np_calc = 0.0, 0.0

        e_np = 100 * abs(np_calc - np_ref)/np_ref
        print(f"{T_val:<6.0f} | {tp:<10.4f} | {tp_ref:<10.4f} | {np_calc:<10.4f} | {np_ref:<10.4f} | {e_np:.2f}%")

    # ==========================================================================
    # PARTE C: PLOTAGEM DOS RESULTADOS (FIGURAS 2a e 2b)
    # ==========================================================================
    print(f"\n---> [C] Geração de Gráficos")
    
    # Simulação estendida para 1000s (10 ciclos) para visualização global.
    # O solver ajusta automaticamente o passo de tempo (sem t_eval fixo).
    t_max_fig = 1000.0
    sol_plot = solver.solve([0, t_max_fig], method='Radau', rtol=1e-8)
    
    times = sol_plot.t
    densities = sol_plot.y[0]

    try:
        plt.figure(figsize=(12, 5))

        # Subplot 1: Detalhe do primeiro ciclo (0-100s)
        mask_2a = times <= 100.0
        plt.subplot(1, 2, 1)
        plt.plot(times[mask_2a], densities[mask_2a], 'b-', lw=1.5)
        plt.title('Fig 2a: Primeiro Ciclo (0-100s)')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Densidade de Nêutrons (Normalizada)')
        plt.grid(True, alpha=0.3)

        # Subplot 2: Visão de múltiplos ciclos (escala logarítmica)
        plt.subplot(1, 2, 2)
        plt.semilogy(times, densities, 'k-', lw=1)
        plt.title('Fig 2b: Evolução em 10 Ciclos (0-1000s)')
        plt.xlabel('Tempo (s)')
        plt.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        plt.show()
        print("     Gráficos gerados.")
        
    except Exception as e:
        print(f"     Erro na rotina de plotagem: {e}")
    
    print("-" * 80)

if __name__ == "__main__":
    problema_1()
