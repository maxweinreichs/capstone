import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from demanda import cargar_costos_unitarios

def resolver_subproblema(p, s, params):
    """
    Resuelve el subproblema de optimización para cada combinación producto-tienda.
    
    Args:
        p (np.array): Array de precios con shape (n_productos, n_tiendas)
        s (np.array): Array de demanda simulada con shape (n_productos, n_tiendas)
        params (dict): Diccionario con parámetros:
            - c: costos unitarios (n_productos)
            - h: costos de inventario (n_productos)
            - d: penalización por faltantes (n_productos, n_tiendas)
            - K: costos fijos de pedido (n_productos)
            - IF: capacidad de inventario por tienda (n_tiendas)
            - I0: inventario inicial (n_productos, n_tiendas)
    
    Returns:
        tuple: (utilidad_total, pedidos, inventario_final)
            - utilidad_total: utilidad total obtenida
            - pedidos: cantidad a pedir por producto-tienda
            - inventario_final: inventario final por producto-tienda
    """
    n_productos, n_tiendas = p.shape
    M = 1e6  # Big M para restricción de pedido
    
    # Crear modelo
    model = gp.Model("Subproblema_Precios")
    model.setParam('OutputFlag', 0)  # Suprimir salida de Gurobi
    
    # Variables
    o = {}  # Cantidad a pedir
    y = {}  # Variable binaria de pedido
    I = {}  # Inventario final
    
    for q in range(n_productos):
        for l in range(n_tiendas):
            o[q,l] = model.addVar(vtype=GRB.CONTINUOUS, name=f"o_{q}_{l}")
            y[q,l] = model.addVar(vtype=GRB.BINARY, name=f"y_{q}_{l}")
            I[q,l] = model.addVar(vtype=GRB.CONTINUOUS, name=f"I_{q}_{l}")
    
    # Función objetivo
    obj = gp.quicksum(
        min(s[q,l], params['I0'][q,l] + o[q,l]) * p[q,l]  # Ingresos
        - params['c'][q] * o[q,l]  # Costos de pedido
        - params['K'][q] * y[q,l]  # Costos fijos
        - params['h'][q] * I[q,l]  # Costos de inventario
        - params['d'][q,l] * max(0, s[q,l] - params['I0'][q,l] - o[q,l])  # Penalización faltantes
        for q in range(n_productos)
        for l in range(n_tiendas)
    )
    
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Restricciones
    for q in range(n_productos):
        for l in range(n_tiendas):
            # Restricción de capacidad de inventario
            model.addConstr(
                params['I0'][q,l] + o[q,l] <= params['IF'][l],
                name=f"cap_inv_{q}_{l}"
            )
            
            # Restricción de variable binaria
            model.addConstr(
                o[q,l] <= M * y[q,l],
                name=f"bin_{q}_{l}"
            )
            
            # Balance de inventario
            model.addConstr(
                I[q,l] == params['I0'][q,l] + o[q,l] - min(s[q,l], params['I0'][q,l] + o[q,l]),
                name=f"balance_{q}_{l}"
            )
    
    # Optimizar
    model.optimize()
    
    # Recopilar resultados
    if model.status == GRB.OPTIMAL:
        utilidad_total = model.objVal
        pedidos = np.zeros((n_productos, n_tiendas))
        inventario_final = np.zeros((n_productos, n_tiendas))
        
        for q in range(n_productos):
            for l in range(n_tiendas):
                pedidos[q,l] = o[q,l].x
                inventario_final[q,l] = I[q,l].x
        
        return utilidad_total, pedidos, inventario_final
    else:
        raise Exception("No se pudo encontrar una solución óptima")

def resolver_multiples_escenarios(p, escenarios_demanda, params, n_escenarios=10):
    """
    Resuelve múltiples escenarios de demanda y devuelve el promedio de utilidad (enfoque SAA).
    
    Args:
        p (np.array): Array de precios con shape (n_productos, n_tiendas)
        escenarios_demanda (list): Lista de arrays de demanda simulada
        params (dict): Diccionario con parámetros de optimización
        n_escenarios (int): Número de escenarios a resolver
    
    Returns:
        tuple: (utilidad_promedio, pedidos_promedio, inventario_promedio)
    """
    n_productos, n_tiendas = p.shape
    
    # Almacenar resultados de cada escenario
    utilidades = []
    pedidos_totales = np.zeros((n_productos, n_tiendas))
    inventarios_totales = np.zeros((n_productos, n_tiendas))
    
    # Resolver cada escenario
    for i, demanda in enumerate(escenarios_demanda):
        try:
            utilidad, pedidos, inventario = resolver_subproblema(p, demanda, params)
            utilidades.append(utilidad)
            pedidos_totales += pedidos
            inventarios_totales += inventario
            
            print(f"  Escenario {i+1}/{len(escenarios_demanda)}: Utilidad = {utilidad:.2f}")
        except Exception as e:
            print(f"  Error en escenario {i+1}: {e}")
            # Si hay error en un escenario, usar valores por defecto
            utilidades.append(0)
    
    # Calcular promedios
    if utilidades:
        utilidad_promedio = sum(utilidades) / len(utilidades)
        pedidos_promedio = pedidos_totales / len(escenarios_demanda)
        inventario_promedio = inventarios_totales / len(escenarios_demanda)
        
        return utilidad_promedio, pedidos_promedio, inventario_promedio
    else:
        raise Exception("No se pudo resolver ningún escenario")

def crear_parametros_optimizacion(n_productos=9, n_tiendas=2, ruta_datos=None):
    """
    Crea un diccionario con parámetros de optimización cargados desde archivos CSV.
    
    Args:
        n_productos (int): Número de productos
        n_tiendas (int): Número de tiendas
        ruta_datos (str, optional): Ruta al directorio de datos
    
    Returns:
        dict: Parámetros de optimización
    """
    # Cargar costos unitarios desde los datos
    costos_unitarios = cargar_costos_unitarios(ruta_datos)
    
    try:
        # Cargar datos del archivo CSV de parámetros base
        params_base_df = pd.read_csv('parametros/Parametros_Base.csv', delimiter=';')
        
        # Extraer costos fijos de pedido (tercera fila)
        costos_fijos_str = params_base_df.iloc[1, 1:10].values
        costos_fijos = np.array([float(costo.replace(',', '.')) for costo in costos_fijos_str])
        
        # Extraer cantidades mínimas de orden (cuarta fila)
        min_orden_str = params_base_df.iloc[2, 1:10].values
        min_orden = np.array([float(orden) for orden in min_orden_str])
        
        # Extraer capacidades de almacenamiento (últimas filas)
        cap_tienda1 = float(params_base_df.iloc[10, 1].replace('.', '').replace(',', '.'))
        cap_tienda2 = float(params_base_df.iloc[11, 1].replace('.', '').replace(',', '.'))
        
        # Crear diccionario de parámetros
        params = {
            'c': costos_unitarios,                                # Costos unitarios
            'h': np.full(n_productos, 0.5),                       # Costos de inventario (por defecto)
            'd': np.full((n_productos, n_tiendas), 2.0),          # Penalización por faltantes (por defecto)
            'K': costos_fijos,                                    # Costos fijos de pedido
            'IF': np.array([cap_tienda1, cap_tienda2]),           # Capacidad de inventario por tienda
            'I0': np.full((n_productos, n_tiendas), 30.0)         # Inventario inicial (por defecto)
        }
        
        print("Parámetros de optimización cargados correctamente desde Parametros_Base.csv")
        return params
        
    except Exception as e:
        print(f"Error al cargar parámetros de optimización desde CSV: {e}")
        print("Usando parámetros por defecto.")
        
        # Si hay error, usar valores por defecto
        return {
            'c': costos_unitarios,                               # Costos unitarios
            'h': np.full(n_productos, 0.5),                     # Costos de inventario
            'd': np.full((n_productos, n_tiendas), 2.0),        # Penalización por faltantes
            'K': np.full(n_productos, 5.0),                     # Costos fijos de pedido
            'IF': np.full(n_tiendas, 100.0),                    # Capacidad de inventario por tienda
            'I0': np.full((n_productos, n_tiendas), 30.0)       # Inventario inicial
        } 