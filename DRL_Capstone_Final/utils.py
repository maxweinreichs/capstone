import pandas as pd
import numpy as np
import os

def read_csv_with_comma_decimal(filepath, delimiter=';'):
    """
    Lee un CSV donde los números pueden usar coma como decimal.
    """
    df = pd.read_csv(filepath, delimiter=delimiter)
    for col in df.columns:
        if df[col].dtype == 'object':
            # Intentar reemplazar coma por punto y convertir a float
            try:
                # Primero asegurar que todos los valores son strings
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False).astype(float)
            except ValueError:
                # Si falla, la columna podría no ser numérica o tener otros formatos
                pass
    return df

def cargar_costos_unitarios_desde_general_params(ruta_general_params_csv, n_productos=10):
    """Carga los costos unitarios c_q desde General_Parameters.csv."""
    # Leer usando punto como decimal, ya que es el formato más común para valores numéricos
    # si están mezclados, read_csv_with_comma_decimal es más robusto para la columna 'valor'
    
    df_params_gen = pd.read_csv(ruta_general_params_csv, delimiter=';')
    params_gen_dict = {}
    # Convertir la columna 'valor' si tiene comas como decimales
    if df_params_gen['valor'].dtype == 'object':
        try:
            df_params_gen['valor'] = df_params_gen['valor'].astype(str).str.replace(',', '.', regex=False).astype(float)
        except ValueError:
            print("Advertencia: No se pudo convertir toda la columna 'valor' a float. Algunos parámetros pueden no cargarse correctamente.")

    for _, row in df_params_gen.iterrows():
        params_gen_dict[row['parametro']] = row['valor']
        
    costos = np.zeros(n_productos, dtype=np.float32)
    for q in range(n_productos):
        costo_val = params_gen_dict.get(f'c_{q}')
        if costo_val is None:
            raise ValueError(f"Costo c_{q} no encontrado en {ruta_general_params_csv}")
        costos[q] = float(costo_val) # Ya debería ser float por la conversión anterior
    return costos

def cargar_precios_iniciales_csv(ruta_precios_csv, n_productos=10, n_tiendas=2):
    """
    Carga los precios desde Precios_Iniciales.csv.
    Formato esperado: L filas, la primera columna es etiqueta, las siguientes Q son precios.
    Los precios en el CSV usan ',' como decimal.
    """
    # Leer el archivo, todas las columnas como string inicialmente para manejar decimales con coma
    df_precios_raw = pd.read_csv(ruta_precios_csv, header=None, delimiter=';', dtype=str)
    
    precios_iniciales_np = np.zeros((n_productos, n_tiendas), dtype=np.float32)
    
    if len(df_precios_raw) < n_tiendas:
        raise ValueError(f"Datos insuficientes para {n_tiendas} tiendas en {ruta_precios_csv}. Encontradas: {len(df_precios_raw)} filas.")

    for l_idx in range(n_tiendas):
        # Precios están desde la columna 1 en adelante
        precios_tienda_str = df_precios_raw.iloc[l_idx, 1:].values 
        
        if len(precios_tienda_str) < n_productos:
            raise ValueError(f"Datos insuficientes de precios para producto en tienda {l_idx+1} en {ruta_precios_csv}. Esperados: {n_productos}, Encontrados: {len(precios_tienda_str)}")
            
        for q_idx in range(n_productos):
            try:
                precios_iniciales_np[q_idx, l_idx] = float(precios_tienda_str[q_idx].replace(',', '.'))
            except ValueError:
                raise ValueError(f"Error convirtiendo precio '{precios_tienda_str[q_idx]}' a float para producto {q_idx+1}, tienda {l_idx+1}.")
                
    return precios_iniciales_np # Shape: (n_productos, n_tiendas)

def guardar_precios_optimos_csv(precios_optimos_np, ruta_archivo_salida, n_productos=10, n_tiendas=2):
    """
    Guarda los precios óptimos en un CSV.
    precios_optimos_np tiene shape (n_productos, n_tiendas).
    El formato de salida usará ',' como decimal y ';' como separador.
    """
    if precios_optimos_np.shape != (n_productos, n_tiendas):
        raise ValueError(f"Dimensiones de precios_optimos_np ({precios_optimos_np.shape}) no coinciden con ({n_productos}, {n_tiendas})")

    data_to_save = []
    for l_idx in range(n_tiendas):
        etiqueta_tienda = f"Precios Optimos Tienda {l_idx + 1}"
        # Convertir precios a string con coma decimal
        precios_str = [f"{p:.2f}".replace('.', ',') for p in precios_optimos_np[:, l_idx]]
        data_to_save.append([etiqueta_tienda] + precios_str)
        
    df_to_save = pd.DataFrame(data_to_save)
    os.makedirs(os.path.dirname(ruta_archivo_salida), exist_ok=True)
    df_to_save.to_csv(ruta_archivo_salida, header=False, index=False, sep=';')
    print(f"Precios óptimos guardados en: {ruta_archivo_salida}")