import gymnasium as gym 
from gymnasium import spaces 
import numpy as np
import os
import time 
# Ya no importa directamente calcular_utilidad_con_precios, lo recibirá como callable
from utils import cargar_costos_unitarios_desde_general_params

class PrecioOptEnv(gym.Env): 
    metadata = {'render_modes': ['human'], 'render_fps': 4} 

    # MODIFICADO: Acepta optimizer_callable
    def __init__(self, optimizer_callable, ruta_datos_gym='parametros', n_productos=10, n_tiendas=2):
        super(PrecioOptEnv, self).__init__() 

        self.n_productos = n_productos
        self.n_tiendas = n_tiendas
        # self.ruta_datos_gym = ruta_datos_gym # ruta_datos para el optimizador ya está en el callable
        self.optimizer_callable = optimizer_callable # Esta función ya tendrá el inv. y semana seteados por main.py

        if not hasattr(PrecioOptEnv, '_init_print_done_gym'): # Usar un nombre diferente para el flag
            print(f"Entorno PrecioOptEnv (Gymnasium API) siendo inicializado...")
            PrecioOptEnv._init_print_done_gym = True

        # Cargar costos solo para el action_space del gym
        ruta_general_params_gym = os.path.join(ruta_datos_gym, "General_Parameters.csv")
        try:
            self.costos_unitarios = cargar_costos_unitarios_desde_general_params(ruta_general_params_gym, self.n_productos)
        except Exception as e:
            print(f"ERROR CRITICO al cargar costos para Gym action_space: {e}.")
            self.costos_unitarios = np.random.uniform(10, 50, size=self.n_productos).astype(np.float32)

        self.action_space = spaces.Box(
            low=1.05 * np.tile(self.costos_unitarios[:, np.newaxis], (1, self.n_tiendas)),
            high=5.0 * np.tile(self.costos_unitarios[:, np.newaxis], (1, self.n_tiendas)),
            shape=(self.n_productos, self.n_tiendas),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        if not hasattr(PrecioOptEnv, '_action_space_print_done_gym'):
            print(f"  Gym Action Space Low (P0T0): {self.action_space.low[0,0]:.2f}, High (P0T0): {self.action_space.high[0,0]:.2f}")
            PrecioOptEnv._action_space_print_done_gym = True

    def reset(self, seed=None, options=None): 
        super().reset(seed=seed) 
        observation = np.array([0.0], dtype=np.float32)
        info = {} 
        return observation, info 

    def step(self, action): 
        precios_propuestos_np = np.array(action).reshape(self.n_productos, self.n_tiendas)
        precios_propuestos_np = np.clip(precios_propuestos_np, self.action_space.low, self.action_space.high)
        
        # Llamar al callable proporcionado, que ya sabe sobre el inventario y semana
        # Este callable debe devolver la utilidad total del horizonte.
        # Los otros datos (pedidos, inv_final) los manejará main.py directamente si es necesario.
        utilidad_horizonte = self.optimizer_callable(precios_propuestos_np)
        
        reward = float(utilidad_horizonte) 
        terminated = True 
        truncated = False 
        next_observation = np.array([0.0], dtype=np.float32)
        info = { 'utilidad_obtenida_horizonte': utilidad_horizonte }
        return next_observation, reward, terminated, truncated, info 

    def render(self): 
        pass
    
    def close(self):
        pass