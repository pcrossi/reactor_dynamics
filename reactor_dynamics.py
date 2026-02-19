# -*- coding: utf-8 -*-
"""
Módulo: reactor_dynamics.py
Versão: 1.0 (Advanced Multi-Physics, Source Aware & Adaptive Control)
Codificação: UTF-8
Descrição: Framework profissional para Cinética Pontual de Neutrons com 
acoplamento multi-física, fonte externa e estratégias de passo adaptativo.

Este módulo está otimizado para simular transientes onde a "stiffness" (rigidez) 
das equações de nêutrons prontos exige métodos numéricos robustos (Radau/BDF) 
ou um controlo heurístico de passo baseado nas constantes de tempo físicas (Tau).

==============================================================================
GUIA RÁPIDO DE UTILIZAÇÃO
==============================================================================

CASO 1: Inserção de Reatividade em Degrau (Step)
------------------------------------------------
    physics = ReactorPhysics(lambdas=[0.08], betas=[0.0065], gen_time=1e-5)
    
    def rho_step(t, state):
        return 0.001 if t > 1.0 else 0.0

    solver = MultiPhysicsSolver(physics, rho_step)
    sol = solver.solve(t_span=(0, 5), n0=1.0)

CASO 2: Realimentação Térmica (Feedback Doppler)
------------------------------------------------
    def d_temp_fuel(t, state):
        return 50.0 * state['n'] - 0.5 * (state['temp_fuel'] - 300.0) 

    def rho_com_feedback(t, state):
        alpha_doppler = -1e-5
        delta_T = state['temp_fuel'] - 300.0
        rho_ext = 0.003 if t > 0.5 else 0.0
        return rho_ext + (alpha_doppler * delta_T)

    solver = MultiPhysicsSolver(physics, rho_com_feedback)
    solver.add_variable('temp_fuel', 300.0, d_temp_fuel)
    sol = solver.solve((0, 10), adaptive=True)
==============================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from typing import Callable, List, Dict, Any, Union, Optional

class SimulationResult:
    """
    Padroniza o formato de saída das simulações para garantir uma interface única.

    Atributos:
        t (np.ndarray): Vetor de tempo de dimensão (M,).
        y (np.ndarray): Matriz de estados de dimensão (N, M).
        sol (Callable): Função interpoladora (Cubic Spline, etc.) se dense_output=True.
        success (bool): Indicador de sucesso da integração.
        message (str): Mensagem de estado.
    """
    def __init__(self, t: np.ndarray, y: np.ndarray):
        self.t = t
        self.y = y
        self.success = True
        self.message = "Simulação concluída com sucesso."
        self.sol = None

class StateVariable:
    """
    Representa uma variável de estado acoplada à neutrónica (ex: Temperatura, Venenos).
    """
    def __init__(self, name: str, initial_value: float, 
                 derivative_func: Callable[[float, Dict[str, float]], float]):
        """
        Inicializa uma variável de estado extra.
        
        Args:
            name: Identificador da variável (ex: 'temp_fuel').
            initial_value: Valor da condição inicial em t=0.
            derivative_func: Função que define a derivada dy/dt = f(t, state_dict).
        """
        self.name = name
        self.initial_value = initial_value
        self.derivative_func = derivative_func

class ReactorPhysics:
    """
    Gestor dos parâmetros cinéticos fundamentais e dos seus modificadores dinâmicos.
    """
    def __init__(self, lambdas: List[float], betas: List[float], gen_time: float):
        """
        Args:
            lambdas: Constantes de decaimento dos precursores [s^-1].
            betas: Frações efetivas de nêutrons atrasados.
            gen_time: Tempo de geração de nêutrons prontos (Lambda) [s].
        """
        self.lambdas_base = np.array(lambdas)
        self.betas_base = np.array(betas)
        self.gen_time_base = gen_time
        self.num_groups = len(lambdas)
        self._modifiers = []

    def add_modifier(self, target_param: str, func: Callable[[float, Dict[str, float]], float]):
        """Regista uma função multiplicadora para alterar dinamicamente um parâmetro nuclear."""
        self._modifiers.append((target_param, func))

    def get_parameters(self, t: float, state_dict: Dict[str, float]):
        """Calcula os parâmetros instantâneos aplicando todos os modificadores ativos."""
        L, B, Lam = self.gen_time_base, self.betas_base.copy(), self.lambdas_base.copy()
        for target, func in self._modifiers:
            factor = func(t, state_dict)
            if target == 'gen_time': L *= factor
            elif target == 'betas':  B *= factor
            elif target == 'lambdas': Lam *= factor
        return Lam, B, L

class MultiPhysicsSolver:
    """
    Integrador numérico para a Cinética Pontual com Acoplamentos Multi-Física e Fonte Externa.
    """
    def __init__(self, physics: ReactorPhysics, 
                 rho_func: Callable[[float, Dict], float],
                 source_func: Optional[Callable[[float, Dict], float]] = None):
        """
        Inicializa o solver definindo a física do núcleo, a reatividade e a fonte externa.

        Args:
            physics: Instância de ReactorPhysics contendo lambdas, betas e Lambda.
            rho_func: Função rho(t, state) que retorna a reatividade absoluta [delta_k/k].
            source_func: (Opcional) Função S(t, state) que retorna a taxa da fonte [nêutrons/s].
                         Se não for fornecida (None), assume-se S(t) = 0 (reator sem fonte).
        """
        self.physics = physics
        self.rho_func = rho_func
        # Função anónima nula para assegurar retrocompatibilidade se não houver fonte
        self.source_func = source_func if source_func else lambda t, s: 0.0
        self.extra_vars: List[StateVariable] = []

    def add_variable(self, name: str, initial: float, derivative_func: Callable):
        """Adiciona uma Equação Diferencial Ordinária (EDO) extra ao sistema global."""
        self.extra_vars.append(StateVariable(name, initial, derivative_func))

    def _unpack_state(self, y: np.ndarray):
        """Traduz o vetor numérico contínuo 'y' para um dicionário legível de componentes físicas."""
        N = y[0]
        C = y[1 : 1 + self.physics.num_groups]
        extras_vals = y[1 + self.physics.num_groups :]
        state_dict = {'n': N}
        for i, var in enumerate(self.extra_vars):
            state_dict[var.name] = extras_vals[i]
        return N, C, extras_vals, state_dict

    def _get_initial_state_vector(self, n0: float) -> np.ndarray:
        """Calcula o vetor de estado inicial assumindo o equilíbrio analítico dos precursores."""
        lam, betas, L = self.physics.get_parameters(0, {'n': n0}) 
        y0_neutronics = np.zeros(self.physics.num_groups + 1)
        y0_neutronics[0] = n0
        # Equilíbrio dos precursores: dC/dt = 0 => lambda*C = (beta/L)*n
        y0_neutronics[1:] = (betas * n0) / (L * lam)
        y0_extras = [v.initial_value for v in self.extra_vars]
        return np.concatenate((y0_neutronics, y0_extras))

    def _rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Define o sistema de EDOs (Right-Hand Side), incluindo o termo de Fonte Externa."""
        N, C, _, state_dict = self._unpack_state(y)
        lam, betas, L = self.physics.get_parameters(t, state_dict)
        beta_t = np.sum(betas)
        rho = self.rho_func(t, state_dict)
        
        # Calcula o valor instantâneo da Fonte Externa S(t)
        source = self.source_func(t, state_dict)
        
        # Equação da Cinética Pontual com Fonte:
        # dn/dt = ((rho - beta)/Lambda) * n + Soma(lambda * C) + S
        dn = ((rho - beta_t) / L) * N + np.sum(lam * C) + source
        
        # Equação dos Precursores de nêutrons Atrasados
        dc = (betas / L) * N - lam * C
        
        # Avaliação das EDOs acopladas (Temperatura, Venenos, etc.)
        d_extras = [var.derivative_func(t, state_dict) for var in self.extra_vars]
        return np.concatenate(([dn], dc, d_extras))

    def solve(self, t_span: tuple, n0: Optional[float] = 1.0, driver: Union[str, Callable] = 'scipy', 
              adaptive: bool = False, **kwargs):
        """
        Executa a integração numérica do sistema acoplado.

        Args:
            t_span: Tuplo contendo os instantes (t_inicial, t_final).
            n0: Potência inicial (N). Se None, o solver calcula automaticamente o equilíbrio 
                subcrítico com base na fonte S(0) e na reatividade rho(0).
            driver: Motor de integração: 'scipy' (recomendado) ou 'rk4' (didático).
            adaptive: Se True, ativa a heurística manual baseada no parâmetro físico Tau.
                      Se False, delega o controlo adaptativo aos métodos nativos do SciPy (Radau/BDF).
            **kwargs: Opções avançadas de controlo numérico:
                - method (str): Método SciPy ('Radau', 'BDF', 'RK45'). Padrão: 'Radau'.
                - rtol, atol (float): Tolerâncias de erro relativo e absoluto.
                - dense_output (bool): Retorna uma função interpoladora contínua em res.sol.
                - interp_kind (str): Método de interpolação se dense_output=True 
                                     ('linear', 'cubic', 'nearest'). Padrão: 'linear' 
                                     (evita overshoot artificial em descontinuidades bruscas).
                - max_step (float): Limite superior do passo no modo adaptativo manual.
                - safety_factor (float): Fracção de Tau utilizada para definir o passo (padrão: 0.1).
                - first_step (float): Dimensão do passo inicial para arrancar o modo adaptativo.
        """
        # --- Tratamento Automático do Estado Inicial (Cálculo de Equilíbrio Subcrítico) ---
        if n0 is None:
            # Cria um estado provisório para extrair as avaliações paramétricas em t=0
            t0 = t_span[0]
            dummy_state = {'n': 0.0} 
            for var in self.extra_vars: 
                dummy_state[var.name] = var.initial_value

            rho_0 = self.rho_func(t0, dummy_state)
            src_0 = self.source_func(t0, dummy_state)
            _, _, L = self.physics.get_parameters(t0, dummy_state)

            # Validações de viabilidade física para sistemas estacionários
            if abs(rho_0) < 1e-9 and src_0 > 1e-9:
                 raise ValueError("Configuração Inválida: Reator Crítico (rho=0) com Fonte não possui equilíbrio estacionário (dn/dt > 0).")
            if rho_0 > 0:
                 raise ValueError("Configuração Inválida: Reator Supercrítico (rho>0) não possui equilíbrio estacionário.")
            
            # Aplicação da Fórmula da Multiplicação Subcrítica: n = - (Lambda * S) / rho
            if abs(rho_0) > 1e-9:
                n0 = - (L * src_0) / rho_0
                print(f"--> [AUTO] Inicialização: n0 calculado como {n0:.4e} (Equilíbrio com Fonte).")
            else:
                n0 = 0.0 # Sem fonte e sem reatividade, o reator está vazio.
        
        # Gera o vetor de estado inicial completo
        y0 = self._get_initial_state_vector(n0)
        want_dense = kwargs.get('dense_output', False)

        # --- OPÇÃO 1: INTEGRADOR NATIVO SCIPY (Recomendado para simulações longas e rígidas) ---
        if driver == 'scipy' and not adaptive:
            if 'method' not in kwargs: kwargs['method'] = 'Radau'
            if 'rtol' not in kwargs: kwargs['rtol'] = 1e-8
            if 'atol' not in kwargs: kwargs['atol'] = 1e-10
            
            return solve_ivp(self._rhs, t_span, y0, **kwargs)

        # --- OPÇÃO 2: ALGORITMO ADAPTATIVO MANUAL (Controlo fino e análise didática) ---
        if driver == 'scipy' and adaptive:
            t_start, t_final = t_span
            method = kwargs.get('method', 'Radau')
            safety_factor = kwargs.get('safety_factor', 0.1) 
            
            # Limite superior de passo (max_step) para prevenir "saltos" cegos no regime permanente
            default_max = max(1.0, (t_final - t_start) / 100.0)
            max_step_user = kwargs.get('max_step', default_max)
            
            t_all, y_all = [t_start], [y0]
            current_t, current_y = t_start, y0
            current_dt = kwargs.get('first_step', 1e-6)

            while current_t < t_final:
                # Ajuste de fronteira para impedir que o solver ultrapasse t_final
                if current_t + current_dt > t_final:
                    current_dt = t_final - current_t
                
                sol_step = solve_ivp(self._rhs, (current_t, current_t + current_dt), 
                                     current_y, method=method, 
                                     rtol=kwargs.get('rtol', 1e-8), 
                                     atol=kwargs.get('atol', 1e-10))
                
                # Redução drástica de passo caso o integrador de baixo nível divirja
                if not sol_step.success:
                    current_dt *= 0.5
                    if current_dt < 1e-15: 
                        break
                    continue
                
                y_next = sol_step.y[:, -1]
                t_next = sol_step.t[-1]
                
                # Cálculo da Heurística Física Baseada no Tempo Característico (Tau)
                derivs = self._rhs(t_next, y_next)
                scale_times = []
                for i, val in enumerate(y_next):
                    rate = abs(derivs[i])
                    val_abs = abs(val)
                    if val_abs > 1e-12 and rate > 1e-20:
                        scale_times.append(val_abs / rate)
                
                # Aceleração em Regime Permanente (Steady State) vs Transiente Rápido
                if not scale_times:
                    suggested_dt = current_dt * 2.0
                else:
                    min_tau = min(scale_times)
                    suggested_dt = safety_factor * min_tau
                
                # Suavização do passo (evita instabilidade no controlo do dt)
                next_dt = np.clip(suggested_dt, current_dt * 0.1, current_dt * 2.0)
                current_dt = min(next_dt, max_step_user)
                
                current_t = t_next
                current_y = y_next
                t_all.append(current_t)
                y_all.append(current_y)

            # Empacotamento do resultado com interpolação customizável
            res = SimulationResult(np.array(t_all), np.array(y_all).T)
          
            if want_dense:
                # O padrão 'linear' garante estabilidade morfológica sem overshoot em degraus
                interp_method = kwargs.get('interp_kind', 'linear')
                
                res.sol = interp1d(
                    res.t, res.y, 
                    kind=interp_method, 
                    axis=1, 
                    fill_value="extrapolate"
                )
            return res

        # --- DRIVERS LEGADOS E CUSTOMIZADOS ---
        elif driver == 'rk4':
            return self._solve_rk4(t_span, y0, kwargs.get('dt', 0.001))
        elif callable(driver):
            return driver(self._rhs, t_span, y0, **kwargs)
        else:
            raise ValueError(f"Driver '{driver}' não reconhecido.")

    def _solve_rk4(self, t_span, y0, dt):
        """
        Método de Runge-Kutta de 4ª ordem clássico (Passo Fixo).
        Incluído exclusivamente para fins comparativos ou didáticos, uma vez que 
        este método explícito é vulnerável à rigidez da cinética de nêutrons.
        """
        t_start, t_end = t_span
        times = np.arange(t_start, t_end + dt, dt)
        y_all = [y0]
        y_curr = y0
        
        for t in times[:-1]:
            k1 = self._rhs(t, y_curr)
            k2 = self._rhs(t + dt/2, y_curr + dt/2 * k1)
            k3 = self._rhs(t + dt/2, y_curr + dt/2 * k2)
            k4 = self._rhs(t + dt, y_curr + dt * k3)
            y_curr = y_curr + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            y_all.append(y_curr)
            
        return SimulationResult(times, np.array(y_all).T)
