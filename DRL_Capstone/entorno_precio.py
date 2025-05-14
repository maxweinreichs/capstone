import gym
import numpy as np
from gym import spaces
from demanda import simular_demanda, crear_parametros_demanda, cargar_costos_unitarios
from optimizador import resolver_subproblema, crear_parametros_optimizacion, resolver_multiples_escenarios

class PrecioPorPeriodoEnv(gym.Env):
    """
    Entorno de Gym para optimización de precios en retail con horizonte de múltiples semanas.
    """
    def __init__(self, n_productos=9, n_tiendas=2, n_semanas=52, n_escenarios=10, ruta_datos=None):
        super(PrecioPorPeriodoEnv, self).__init__()
        
        self.n_productos = n_productos
        self.n_tiendas = n_tiendas
        self.n_semanas = n_semanas
        self.n_escenarios = n_escenarios
        self.semana_actual = 0
        
        # Cargar costos unitarios
        self.costos_unitarios = cargar_costos_unitarios(ruta_datos)
        
        # Espacio de acción: precios para cada combinación producto-tienda
        # Sin límite superior de precio (la demanda se encargará de penalizar precios muy altos)
        self.action_space = spaces.Box(
            low=1.05 * np.tile(self.costos_unitarios[:, np.newaxis], (1, self.n_tiendas)),
            high=np.ones((self.n_productos, self.n_tiendas)) * float('inf'),
            shape=(self.n_productos, self.n_tiendas),
            dtype=np.float32
        )
        
        # Espacio de observación: incluye la semana actual
        self.observation_space = spaces.Box(
            low=0,
            high=n_semanas,
            shape=(1,),
            dtype=np.float32
        )
        
        # Parámetros
        self.params_demanda = crear_parametros_demanda(n_productos, n_tiendas, ruta_datos)
        self.params_optimizacion = crear_parametros_optimizacion(n_productos, n_tiendas, ruta_datos)
        
        # Historial de resultados por semana
        self.historial_precios = []
        self.historial_demandas = []
        self.historial_pedidos = []
        self.historial_inventarios = []
        self.historial_utilidades = []
        
        # Estado interno
        self.reset()
    
    def reset(self):
        """
        Reinicia el entorno.
        """
        self.semana_actual = 0
        
        # Reiniciar inventario inicial
        self.params_optimizacion['I0'] = np.full((self.n_productos, self.n_tiendas), 30.0)
        
        # Reiniciar historiales
        self.historial_precios = []
        self.historial_demandas = []
        self.historial_pedidos = []
        self.historial_inventarios = []
        self.historial_utilidades = []
        
        return np.array([self.semana_actual], dtype=np.float32)
    
    def step(self, action):
        """
        Ejecuta un paso del entorno utilizando el enfoque SAA con múltiples escenarios.
        
        Args:
            action (np.array): Array de precios con shape (n_productos, n_tiendas)
        
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Validar precios mínimos
        precios_minimos = 1.05 * np.tile(self.costos_unitarios[:, np.newaxis], (1, self.n_tiendas))
        action = np.maximum(action, precios_minimos)
        
        try:
            print(f"Semana {self.semana_actual+1}/{self.n_semanas} - Generando {self.n_escenarios} escenarios de demanda...")
            
            # Generar múltiples escenarios de demanda para la semana actual
            escenarios_demanda = []
            escenarios_media = []
            
            for i in range(self.n_escenarios):
                media, demanda = simular_demanda(action, self.params_demanda, semana=self.semana_actual)
                escenarios_demanda.append(demanda)
                escenarios_media.append(media)
            
            # Resolver múltiples escenarios y obtener el promedio (SAA)
            print(f"Resolviendo {self.n_escenarios} escenarios para la semana {self.semana_actual+1}...")
            utilidad, pedidos, inventario = resolver_multiples_escenarios(
                action, escenarios_demanda, self.params_optimizacion, self.n_escenarios
            )
            
            # Utilizar el promedio de la demanda para los historiales
            demanda_promedio = np.mean(escenarios_demanda, axis=0)
            
            # Actualizar inventario inicial para el siguiente paso
            self.params_optimizacion['I0'] = inventario
            
            # Guardar resultados en el historial
            self.historial_precios.append(action.copy())
            self.historial_demandas.append(demanda_promedio)
            self.historial_pedidos.append(pedidos)
            self.historial_inventarios.append(inventario)
            self.historial_utilidades.append(utilidad)
            
            # Imprimir estado actual
            print(f"Semana {self.semana_actual+1}: Utilidad = {utilidad:.2f}")
            print(f"  Precios promedio: {np.mean(action):.2f}")
            print(f"  Demanda promedio: {np.mean(demanda_promedio):.2f}")
            print(f"  Pedidos promedio: {np.mean(pedidos):.2f}")
            
            # Actualizar semana
            self.semana_actual += 1
            done = self.semana_actual >= self.n_semanas
            
            # Información adicional
            info = {
                'semana': self.semana_actual - 1,
                'demanda': demanda_promedio,
                'pedidos': pedidos,
                'inventario': inventario,
                'precios': action,
                'utilidad_semana': utilidad
            }
            
            # Si hemos terminado, agregar resumen en info
            if done:
                info['historial_precios'] = np.array(self.historial_precios)
                info['historial_demandas'] = np.array(self.historial_demandas)
                info['historial_pedidos'] = np.array(self.historial_pedidos)
                info['historial_inventarios'] = np.array(self.historial_inventarios)
                info['historial_utilidades'] = np.array(self.historial_utilidades)
                info['utilidad_total'] = sum(self.historial_utilidades)
                
                # Imprimir resumen final
                print("\nResumen del episodio:")
                print(f"  Utilidad total: {info['utilidad_total']:.2f}")
                print(f"  Utilidad promedio por semana: {np.mean(self.historial_utilidades):.2f}")
                print(f"  Precio promedio: {np.mean(self.historial_precios):.2f}")
                print(f"  Demanda promedio: {np.mean(self.historial_demandas):.2f}")
                print(f"  Pedidos promedio: {np.mean(self.historial_pedidos):.2f}")
            
            return np.array([self.semana_actual], dtype=np.float32), utilidad, done, info
            
        except Exception as e:
            print(f"Error en step: {e}")
            # Si hay error en la optimización, penalizar fuertemente
            return np.array([self.semana_actual], dtype=np.float32), -1e6, True, {'error': str(e)}
    
    def render(self, mode='human'):
        """
        Renderiza el estado actual del entorno (no implementado).
        """
        pass
    
    def close(self):
        """
        Cierra el entorno (no implementado).
        """
        pass 