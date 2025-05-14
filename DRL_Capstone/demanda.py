import numpy as np
import pandas as pd
import os

def simular_demanda(p, params, semana=0, tipo_distribucion=None):
    """
    Simula la demanda para cada combinación producto-tienda.
    
    Args:
        p (np.array): Array de precios con shape (n_productos, n_tiendas)
        params (dict): Diccionario con parámetros de demanda por tipo de distribución
        semana (int): Semana del año (0-51)
        tipo_distribucion (dict, optional): Mapeo explícito de distribución.
                                            Si es None, se usa el mapeo interno.
    
    Returns:
        tuple: (demanda_media, demanda_real)
            - demanda_media: media de la demanda (n_productos, n_tiendas)
            - demanda_real: realización de la demanda (n_productos, n_tiendas)
    """
    n_productos, n_tiendas = p.shape
    
    # Determinar tipo de distribución por semana si no se proporciona explícitamente
    if tipo_distribucion is None:
        tipo_distribucion = determinar_tipo_distribucion(semana)
    
    mu_valores = np.zeros((n_productos, n_tiendas))
    sigma_valores = np.zeros((n_productos, n_tiendas))
    
    # Calcular media y desviación estándar para cada producto y tienda
    for q in range(n_productos):
        for l in range(n_tiendas):
            # Obtener parámetros según tipo de distribución para este producto y tienda
            producto_key = f'producto_{q+1}'
            tienda_key = l+1
            
            # Obtener el tipo de distribución para este producto, tienda y semana
            dist_tipo = tipo_distribucion.get((producto_key, tienda_key, semana), 'medio')
            
            # Obtener parámetros para este producto, tienda y tipo de distribución
            dist_params = params.get((producto_key, tienda_key, dist_tipo), None)
            
            if dist_params is None:
                # Si no hay parámetros específicos, usar valores por defecto
                dist_params = {
                    'rho': 100.0,
                    'alpha': 0.07,
                    'theta': 20.0
                }
                print(f"Advertencia: No se encontraron parámetros para {producto_key}, tienda {tienda_key}, tipo {dist_tipo}")
            
            # Calcular media de demanda usando los parámetros de la distribución seleccionada
            mu = dist_params['rho'] * np.exp(-dist_params['alpha'] * p[q, l])
            
            # Calcular varianza
            sigma2 = mu * (1 + mu / dist_params['theta'])
            
            mu_valores[q, l] = mu
            sigma_valores[q, l] = np.sqrt(sigma2)
    
    # Generar realización de demanda
    s = np.random.normal(mu_valores, sigma_valores)
    s = np.maximum(0, s)  # Asegurar demanda no negativa
    
    return mu_valores, s

def determinar_tipo_distribucion(semana):
    """
    Determina el tipo de distribución (alta, media, baja) para cada producto-tienda-semana
    basado en el archivo de asociación semana-temporada.
    
    Args:
        semana (int): Semana del año (0-51)
    
    Returns:
        dict: Diccionario con claves (producto, tienda, semana) y valores 'alto', 'medio' o 'bajo'
    """
    # Cargar el mapeo de semana a temporada desde el archivo CSV
    try:
        asociacion_df = pd.read_csv('parametros/asociacion_semana_temporada.csv', delimiter=';')
        
        # Crear diccionario para mapear semana a tipo de distribución
        tipo_dist = {}
        
        # Filtrar filas para la semana actual
        for _, row in asociacion_df.iterrows():
            if row['nro_semana'] == semana:
                producto = row['producto']
                tienda = int(row['Tienda'])
                grupo = row['grupo']
                tipo_dist[(producto, tienda, semana)] = grupo
        
        # Si no hay datos para alguna combinación, usar 'medio' como valor por defecto
        if not tipo_dist:
            print(f"Advertencia: No se encontró mapeo para la semana {semana}, usando distribución media.")
            for q in range(9):
                for l in range(2):
                    producto_key = f'producto_{q+1}'
                    tienda_key = l+1
                    tipo_dist[(producto_key, tienda_key, semana)] = 'medio'
        
        return tipo_dist
    
    except Exception as e:
        print(f"Error al cargar el mapeo de semana a temporada: {e}")
        print("Usando mapeo por defecto.")
        
        # Mapeo por defecto si no se puede cargar el archivo
        tipo_dist = {}
        for q in range(9):
            for l in range(2):
                producto_key = f'producto_{q+1}'
                tienda_key = l+1
                tipo_dist[(producto_key, tienda_key, semana)] = 'medio'
        
        return tipo_dist

def crear_parametros_demanda(n_productos=9, n_tiendas=2, ruta_datos=None):
    """
    Crea un diccionario con parámetros de demanda para diferentes distribuciones
    cargados desde archivos CSV.
    
    Args:
        n_productos (int): Número de productos
        n_tiendas (int): Número de tiendas
        ruta_datos (str, optional): Ruta al directorio de datos
    
    Returns:
        dict: Parámetros de demanda por tipo de distribución
    """
    # Estructura para almacenar parámetros por producto, tienda y tipo de distribución
    params = {}
    
    try:
        # Cargar parámetros para la tienda 1
        tienda1_df = pd.read_csv('parametros/parametros_distribucion_tienda_1.csv', delimiter=';')
        
        # Cargar parámetros para la tienda 2
        tienda2_df = pd.read_csv('parametros/parametros_distribuciones_tienda_2.csv', delimiter=';')
        
        # Procesar datos para la tienda 1
        for _, row in tienda1_df.iterrows():
            producto = row['producto']
            grupo = row['grupo']
            params[(producto, 1, grupo)] = {
                'rho': float(row['rho']),
                'alpha': float(row['alfa']),
                'theta': float(row['theta (simbolo raro)'])
            }
        
        # Procesar datos para la tienda 2
        for _, row in tienda2_df.iterrows():
            producto = row['producto']
            grupo = row['grupo']
            params[(producto, 2, grupo)] = {
                'rho': float(row['rho']),
                'alpha': float(row['alfa']),
                'theta': float(row['theta (simbolo raro)'])
            }
        
        print("Parámetros de demanda cargados correctamente desde archivos CSV.")
        return params
    
    except Exception as e:
        print(f"Error al cargar parámetros de demanda desde CSV: {e}")
        print("Usando parámetros por defecto.")
        
        # Si hay error, generar parámetros por defecto
        for q in range(n_productos):
            producto_key = f'producto_{q+1}'
            for l in range(n_tiendas):
                tienda_key = l+1
                
                # Parámetros para distribución alta (demanda alta)
                params[(producto_key, tienda_key, 'alto')] = {
                    'rho': 150.0 + np.random.uniform(-20, 20),     # Demanda base más alta
                    'alpha': 0.05 + np.random.uniform(-0.01, 0.01),  # Menos sensible al precio
                    'theta': 25.0 + np.random.uniform(-5, 5)         # Más dispersión
                }
                
                # Parámetros para distribución media (demanda media)
                params[(producto_key, tienda_key, 'medio')] = {
                    'rho': 100.0 + np.random.uniform(-15, 15),     # Demanda base media
                    'alpha': 0.07 + np.random.uniform(-0.01, 0.01),  # Sensibilidad media
                    'theta': 20.0 + np.random.uniform(-4, 4)         # Dispersión media
                }
                
                # Parámetros para distribución baja (demanda baja)
                params[(producto_key, tienda_key, 'bajo')] = {
                    'rho': 50.0 + np.random.uniform(-10, 10),      # Demanda base baja
                    'alpha': 0.09 + np.random.uniform(-0.01, 0.01),  # Más sensible al precio
                    'theta': 15.0 + np.random.uniform(-3, 3)         # Menos dispersión
                }
        
        return params

def cargar_costos_unitarios(ruta_datos=None):
    """
    Carga los costos unitarios desde el archivo Parametros_Base.csv.
    
    Args:
        ruta_datos (str, optional): Ruta al archivo de datos
    
    Returns:
        np.array: Array de costos unitarios con shape (n_productos)
    """
    try:
        # Cargar datos del archivo CSV
        params_base_df = pd.read_csv('parametros/Parametros_Base.csv', delimiter=';')
        
        # Extraer costos unitarios (segunda fila)
        costos_str = params_base_df.iloc[0, 1:10].values  # Tomar las columnas 1 a 9 (productos 1-9)
        
        # Convertir a valores numéricos (reemplazando comas por puntos)
        costos = np.array([float(costo.replace(',', '.')) for costo in costos_str])
        
        print("Costos unitarios cargados correctamente desde Parametros_Base.csv")
        return costos
    
    except Exception as e:
        print(f"Error al cargar costos unitarios desde CSV: {e}")
        print("Usando costos unitarios por defecto.")
        
        # Si hay error, usar valores por defecto
        costos = np.array([28.792, 20.792, 31.992, 52.792, 25.592, 44.8, 62.993, 46.4925, 32.3919])
        return costos 