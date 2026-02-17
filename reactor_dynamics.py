"""
Módulo: reactor_dynamics.py
Versão: 1.1 (Revisão Didática)
Autor: Pedro C. R. Rossi / Adaptado para Curso de Eng. Nuclear
Descrição: Framework para resolução de equações de Cinética Pontual com Acoplamento Multi-Física.

Este módulo permite simular transientes nucleares onde a reatividade e os parâmetros 
do reator podem variar dinamicamente em função de variáveis de estado (Temperatura, Vazio, Xenônio).

==============================================================================
GUIA RÁPIDO DE USO (EXEMPLOS)
==============================================================================

CASO 1: Inserção de Reatividade em Degrau (Step)
------------------------------------------------
Simula um salto de reatividade (ex: retirada de barra) sem realimentação térmica.

    # 1. Defina os parâmetros nucleares (1 grupo de precursores para simplificar)
    #    Lambdas [1/s], Betas [-], Tempo de Geração [s]
    physics = ReactorPhysics(lambdas=[0.08], betas=[0.0065], gen_time=1e-5)

    # 2. Defina a função de Reatividade (rho em delta_k/k)
    def rho_step(t, state):
        # Insere 100 pcm (0.001) após 1 segundo
        return 0.001 if t > 1.0 else 0.0

    # 3. Instancie o Solver e execute
    solver = MultiPhysicsSolver(physics, rho_step)
    sol = solver.solve(t_span=(0, 5), n0=1.0)
    
    # Resultado: sol.t (tempo) e sol.y[0] (nêutrons)


CASO 2: Transiente com Realimentação Térmica (Feedback)
-------------------------------------------------------
Simula um pulso de potência limitado pelo Efeito Doppler (aquecimento do combustível).
Precisamos adicionar uma variável extra ('temp_fuel') ao sistema.

    # 1. Defina a equação diferencial da Temperatura (dT/dt)
    #    Modelo simples: Calor gerado pela potência - Calor removido
    def d_temp_fuel(t, state):
        potencia = state['n']          # Densidade de nêutrons
        temp = state['temp_fuel']      # Temperatura atual
        return 50.0 * potencia - 0.5 * (temp - 300.0) 

    # 2. Defina a Reatividade com Coeficiente de Feedback (Doppler)
    def rho_com_feedback(t, state):
        alpha_doppler = -1e-5          # -1 pcm/°C
        delta_T = state['temp_fuel'] - 300.0
        
        rho_externo = 0.003 if t > 0.5 else 0.0 # Degrau grande de 300 pcm
        return rho_externo + (alpha_doppler * delta_T)

    # 3. Configure o Solver com a variável extra
    solver = MultiPhysicsSolver(physics, rho_com_feedback)
    
    # Adiciona a variável: nome='temp_fuel', valor_inicial=300.0, derivada=d_temp_fuel
    solver.add_variable('temp_fuel', 300.0, d_temp_fuel)
    
    sol = solver.solve((0, 10))

==============================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, List, Dict, Any, Union, Optional

# ==============================================================================
# 0. CLASSE AUXILIAR DE RESULTADO
# ==============================================================================

class SimulationResult:
    """
    Padroniza a saída da simulação.
    Garante que o aluno receba o mesmo formato de objeto (sol.t, sol.y),
    seja usando o solver profissional (Scipy) ou o solver didático (RK4).
    """
    def __init__(self, t, y):
        self.t = t
        self.y = y
        self.success = True
        self.message = "Simulação concluída com sucesso."

# ==============================================================================
# 1. CLASSE DE SUPORTE: VARIÁVEL DE ESTADO
# ==============================================================================

class StateVariable:
    """
    Define uma variável física acoplada ao reator (ex: Temperatura, Concentração de Xe-135).
    Conecta a física nuclear à termohidráulica ou avenenamento.
    """
    
    def __init__(self, name: str, initial_value: float, 
                 derivative_func: Callable[[float, Dict[str, float]], float]):
        """
        Registra uma nova equação diferencial dy/dt = f(t, state).

        Args:
            name (str): Identificador único (chave usada no dicionário de estado).
            initial_value (float): Valor inicial em t=0.
            derivative_func (Callable): Função que calcula a taxa de variação.
        """
        self.name = name
        self.initial_value = initial_value
        self.derivative_func = derivative_func


# ==============================================================================
# 2. GERENCIADOR DE FÍSICA NUCLEAR (ReactorPhysics)
# ==============================================================================

class ReactorPhysics:
    """
    Guardião dos parâmetros nucleares (Cinética Pontual).
    Permite alterar Beta, Lambda e Tempo de Geração durante a simulação
    (útil para simular mudanças espectrais ou efeitos de temperatura nos parâmetros).
    """

    def __init__(self, lambdas, betas, gen_time):
        """
        Args:
            lambdas (list): Constantes de decaimento dos precursores [s^-1].
            betas (list): Frações de nêutrons atrasados (efetivos) [-].
            gen_time (float): Tempo de geração de nêutrons prontos (Lambda) [s].
        """
        self.lambdas_base = np.array(lambdas)
        self.betas_base = np.array(betas)
        self.gen_time_base = gen_time
        self.num_groups = len(lambdas)
        self._modifiers = [] # Lista de funções que alteram os parâmetros base

    def add_modifier(self, target_param: str, func: Callable[[float, Dict[str, float]], float]):
        """
        Adiciona uma regra para variação dinâmica de parâmetros.
        Ex: Alterar o tempo de geração se houver mudança de fase (vazio).
        """
        self._modifiers.append((target_param, func))

    def get_parameters(self, t: float, state_dict: Dict[str, float]):
        """
        Retorna os parâmetros (Lambda, Beta, GenTime) válidos para o instante atual,
        aplicando quaisquer modificadores registrados.
        """
        L = self.gen_time_base
        B = self.betas_base.copy()
        Lam = self.lambdas_base.copy()
        
        for target, func in self._modifiers:
            factor = func(t, state_dict)
            if target == 'gen_time': L *= factor
            elif target == 'betas':  B *= factor
            elif target == 'lambdas': Lam *= factor
            
        return Lam, B, L


# ==============================================================================
# 3. MOTOR DE SOLUÇÃO (MultiPhysicsSolver)
# ==============================================================================

class MultiPhysicsSolver:
    """
    Núcleo da simulação. Integra o sistema acoplado:
    [Neutrônica (Rígida)] <---> [Termohidráulica/Outros (Lenta/Rápida)]
    """

    def __init__(self, physics: ReactorPhysics, rho_func: Callable[[float, Dict], float]):
        """
        Args:
            physics: Objeto ReactorPhysics configurado.
            rho_func: Função que retorna a Reatividade Total (externa + feedback) [delta_k/k].
                      Assinatura: def rho(t, state_dict) -> float
        """
        self.physics = physics
        self.rho_func = rho_func 
        self.extra_vars: List[StateVariable] = []
        
    def add_variable(self, name, initial, derivative_func):
        """
        Adiciona uma equação de balanço extra (energia, massa, veneno).
        """
        self.extra_vars.append(StateVariable(name, initial, derivative_func))

    # --- Helpers de Vetorização (Internos) ---
    # O solver numérico "enxerga" apenas um vetor longo de números (y).
    # Estes métodos convertem y <-> Dicionário legível (state_dict).

    def _pack_state(self, N, C, extras):
        """Junta Nêutrons, Precursores e Extras em um vetor único."""
        return np.concatenate(([N], C, extras))

    def _unpack_state(self, y):
        """Separa o vetor único em componentes físicas e cria o dicionário de estado."""
        N = y[0] # Nêutrons (sempre o índice 0)
        C = y[1 : 1 + self.physics.num_groups] # Precursores
        extras_vals = y[1 + self.physics.num_groups :] # Variáveis extras
        
        # Cria dicionário para facilitar o acesso nas funções do usuário (ex: state['temp'])
        state_dict = {'n': N}
        for i, var in enumerate(self.extra_vars):
            state_dict[var.name] = extras_vals[i]
            
        return N, C, extras_vals, state_dict

    def _get_initial_state_vector(self, n0):
        """Calcula o vetor inicial (t=0) assumindo equilíbrio dos precursores."""
        lam, betas, L = self.physics.get_parameters(0, {'n': n0}) 
        
        y0_neutronics = np.zeros(self.physics.num_groups + 1)
        y0_neutronics[0] = n0
        # Condição de equilíbrio: dC/dt = 0 => C_i = (beta_i * n) / (Lambda * lambda_i)
        y0_neutronics[1:] = (betas * n0) / (L * lam)
        
        y0_extras = [v.initial_value for v in self.extra_vars]
        return np.concatenate((y0_neutronics, y0_extras))

    def _rhs(self, t, y):
        """
        Right-Hand Side (O coração do sistema).
        Calcula as derivadas dy/dt para o instante t.
        """
        # 1. Desempacota o estado atual
        N, C, _, state_dict = self._unpack_state(y)
        
        # 2. Atualiza parâmetros físicos (caso variem com o tempo/estado)
        lam, betas, L = self.physics.get_parameters(t, state_dict)
        beta_t = np.sum(betas)
        
        # 3. Calcula a Reatividade Total (incluindo feedback se houver)
        rho = self.rho_func(t, state_dict)
        
        # 4. Equações de Cinética Pontual (Neutrônica)
        # dN/dt = ((rho - beta)/L)*N + sum(lambda * C)
        dn = ((rho - beta_t) / L) * N + np.sum(lam * C)
        # dC/dt = (beta/L)*N - lambda * C
        dc = (betas / L) * N - lam * C
        
        # 5. Equações das Variáveis Extras (Termohidráulica, etc.)
        d_extras = []
        for var in self.extra_vars:
            d_extras.append(var.derivative_func(t, state_dict))
            
        return np.concatenate(([dn], dc, d_extras))

    def _solve_rk4(self, t_span, y0, dt):
        """
        Método Didático: Runge-Kutta de 4ª Ordem com passo fixo.
        Ideal para ensino, pois permite ver o algoritmo 'abrindo a caixa preta'.
        """
        t_start, t_end = t_span
        times = np.arange(t_start, t_end + dt, dt)
        states = [y0]
        
        y_curr = y0
        for t in times[:-1]:
            # Passo padrão RK4
            k1 = self._rhs(t, y_curr)
            k2 = self._rhs(t + dt/2, y_curr + dt/2 * k1)
            k3 = self._rhs(t + dt/2, y_curr + dt/2 * k2)
            k4 = self._rhs(t + dt, y_curr + dt * k3)
            
            y_next = y_curr + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            states.append(y_next)
            y_curr = y_next
            
        min_len = min(len(times), len(states))
        return SimulationResult(np.array(times[:min_len]), np.array(states[:min_len]).T)
    
    def solve(self, t_span, n0=1.0, driver: Union[str, Callable] = 'scipy', adaptive: bool = False, **kwargs):
            """
            Executa a simulação.
    
            Args:
                t_span (tuple): (inicio, fim), ex: (0.0, 10.0).
                n0 (float): Potência inicial (normalizada).
                driver (str): 'scipy' (recomendado) ou 'rk4' (didático).
                adaptive (bool): Se True, usa lógica manual de controle de passo físico
                                 (útil para transientes extremamente rápidos/rígidos).
                **kwargs: Opções extras (method, rtol, dt, etc).
            """
            y0 = self._get_initial_state_vector(n0)
            adaptive_flag = kwargs.pop('adaptive', False) or adaptive
            
            # --- DRIVER 1: SCIPY (Profissional) ---
            if driver == 'scipy':
                # Padrões robustos para equações rígidas (Stiff)
                if 'method' not in kwargs: kwargs['method'] = 'Radau'
                if 'rtol' not in kwargs: kwargs['rtol'] = 1e-8
                if 'atol' not in kwargs: kwargs['atol'] = 1e-10
                
                # Modo Adaptativo Físico (Manual)
                # Tenta controlar o passo observando a constante de tempo mais rápida do sistema
                if adaptive_flag:
                    t_start, t_final = t_span
                    method = kwargs.get('method', 'Radau')
                    rtol = kwargs.get('rtol', 1e-8)
                    atol = kwargs.get('atol', 1e-10)
                    
                    safety_factor = 0.05  # Limite: passo <= 5% da menor constante de tempo
                    min_step = 1e-14
                    max_step_user = kwargs.get('max_step', np.inf)
                    
                    t_all = [t_start]
                    y_all = [y0]
                    current_t = t_start
                    current_y = y0
                    current_dt = kwargs.get('first_step', 1e-6)
    
                    # Loop manual de integração
                    while current_t < t_final:
                        if current_t + current_dt > t_final:
                            current_dt = t_final - current_t
                        
                        sol_step = solve_ivp(
                            self._rhs, 
                            (current_t, current_t + current_dt), 
                            current_y,
                            method=method, rtol=rtol, atol=atol
                        )
                        
                        if not sol_step.success:
                            current_dt *= 0.5 
                            if current_dt < min_step: break
                            continue
                        
                        y_next = sol_step.y[:, -1]
                        t_next = sol_step.t[-1]
                        
                        # Análise Física para o próximo passo:
                        # Estima as derivadas e encontra a variável que está mudando mais rápido
                        derivs = self._rhs(t_next, y_next)
                        scale_times = []
                        for i, val in enumerate(y_next):
                            rate = abs(derivs[i])
                            val_abs = abs(val)
                            if val_abs > 1e-12 and rate > 1e-20:
                                scale_times.append(val_abs / rate) # Tau = valor / taxa
                        
                        if scale_times:
                            min_tau = min(scale_times)
                            suggested_dt = safety_factor * min_tau
                            # Evita oscilação brusca de dt
                            suggested_dt = np.clip(suggested_dt, current_dt * 0.1, current_dt * 2.0)
                        else:
                            suggested_dt = current_dt * 2.0
    
                        current_dt = np.clip(suggested_dt, min_step, max_step_user)
                        
                        t_all.append(t_next)
                        y_all.append(y_next)
                        current_t = t_next
                        current_y = y_next
                    
                    return SimulationResult(np.array(t_all), np.array(y_all).T)
    
                # Modo Padrão Scipy (Automático)
                else:
                    return solve_ivp(self._rhs, t_span, y0, **kwargs)
            
            # --- DRIVER 2: RK4 (Didático) ---
            elif driver == 'rk4':
                dt = kwargs.get('dt', 0.001)
                return self._solve_rk4(t_span, y0, dt)
                
            elif callable(driver):
                return driver(self._rhs, t_span, y0, **kwargs)
                
            else:
                raise ValueError(f"Driver '{driver}' não reconhecido.")
