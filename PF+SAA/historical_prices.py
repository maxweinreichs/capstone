import pandas as pd
import numpy as np
import os

def load_historical_prices(ruta_datos_historicos="datos_semanales_agrupados.csv"):
    """
    Carga los precios históricos desde el archivo CSV y los organiza en una estructura
    fácil de acceder por semana, producto y tienda.
    
    Args:
        ruta_datos_historicos (str): Ruta al archivo CSV con datos históricos
        
    Returns:
        dict: Diccionario con estructura {semana: {producto: {tienda: precio_promedio}}}
    """
    try:
        # Cargar el archivo CSV
        df = pd.read_csv(ruta_datos_historicos, delimiter=';', decimal=',')
        
        # Verificar que las columnas necesarias existen
        required_columns = ['numero_semana', 'producto', 'tienda', 'precio_promedio']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Columna requerida '{col}' no encontrada en el archivo")
        
        # Crear estructura de datos organizada
        precios_historicos = {}
        
        for _, row in df.iterrows():
            semana = int(row['numero_semana'])
            producto = int(row['producto'])
            tienda = int(row['tienda'])
            precio = float(str(row['precio_promedio']).replace(',', '.'))
            
            # Inicializar estructura si no existe
            if semana not in precios_historicos:
                precios_historicos[semana] = {}
            if producto not in precios_historicos[semana]:
                precios_historicos[semana][producto] = {}
            
            # Guardar el precio
            precios_historicos[semana][producto][tienda] = precio
        
        print(f"✅ Datos históricos cargados exitosamente:")
        print(f"   - Semanas: {min(precios_historicos.keys())} a {max(precios_historicos.keys())}")
        print(f"   - Productos: {len(precios_historicos[1])} productos")
        print(f"   - Tiendas: {len(precios_historicos[1][1])} tiendas")
        
        return precios_historicos
        
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {ruta_datos_historicos}")
        return None
    except Exception as e:
        print(f"❌ Error al cargar datos históricos: {e}")
        return None

def get_historical_prices_matrix(precios_historicos, semana_objetivo, n_productos, n_tiendas):
    """
    Obtiene una matriz de precios históricos para una semana específica.
    
    Args:
        precios_historicos (dict): Diccionario con datos históricos
        semana_objetivo (int): Número de semana (1-52)
        n_productos (int): Número de productos
        n_tiendas (int): Número de tiendas
        
    Returns:
        np.array: Matriz de precios históricos (n_productos, n_tiendas)
    """
    if precios_historicos is None:
        print("❌ No hay datos históricos disponibles")
        return None
    
    # Ajustar semana_objetivo al rango 1-52 (ciclo anual)
    semana_ciclica = ((semana_objetivo - 1) % 52) + 1
    
    if semana_ciclica not in precios_historicos:
        print(f"❌ No se encontraron datos para la semana {semana_ciclica}")
        return None
    
    # Crear matriz de precios
    precios_matrix = np.zeros((n_productos, n_tiendas))
    
    for producto_idx in range(n_productos):
        for tienda_idx in range(n_tiendas):
            # Los índices en el CSV van de 1 a N, pero en numpy van de 0 a N-1
            producto_csv = producto_idx + 1
            tienda_csv = tienda_idx + 1
            
            if (producto_csv in precios_historicos[semana_ciclica] and 
                tienda_csv in precios_historicos[semana_ciclica][producto_csv]):
                precios_matrix[producto_idx, tienda_idx] = precios_historicos[semana_ciclica][producto_csv][tienda_csv]
            else:
                print(f"⚠️  No se encontró precio histórico para P{producto_csv}T{tienda_csv} semana {semana_ciclica}")
                # Usar un precio por defecto si no se encuentra
                precios_matrix[producto_idx, tienda_idx] = 30.0  # Precio por defecto
    
    print(f"📊 Precios históricos para semana {semana_objetivo} (ciclo: {semana_ciclica}):")
    print(f"   - P1T1: {precios_matrix[0,0]:.2f}")
    print(f"   - P2T1: {precios_matrix[1,0]:.2f}")
    print(f"   - P1T2: {precios_matrix[0,1]:.2f}")
    
    return precios_matrix

def test_historical_prices():
    """
    Función de prueba para verificar que la carga de datos funciona correctamente.
    """
    print("🧪 Probando carga de datos históricos...")
    
    precios_historicos = load_historical_prices()
    
    if precios_historicos:
        # Probar obtener precios para diferentes semanas
        for semana in [1, 2, 26, 52]:
            print(f"\n--- Semana {semana} ---")
            matriz = get_historical_prices_matrix(precios_historicos, semana, 10, 2)
            if matriz is not None:
                print(f"Matriz shape: {matriz.shape}")
                print(f"Precio promedio: {np.mean(matriz):.2f}")

if __name__ == "__main__":
    test_historical_prices() 