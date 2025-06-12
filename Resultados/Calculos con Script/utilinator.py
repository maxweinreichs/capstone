import pandas as pd
import sys

# --- CONSTANTES Y CONFIGURACIÓN ---
TASA_COSTO_INVENTARIO = 0.10
TASA_PENALIDAD_SHORTAGE = 0.10

# Nombres de los archivos de entrada
DATOS_MODELO_FILE = 'Planificacion_Semanal_Optima_PF.csv'
PARAMETROS_FILE = 'General_Parameters.csv'
DEMANDA_REAL_FILE = 'demanda_real.csv'
OUTPUT_FILE = 'resultados_detallados_final.csv'

# --- FUNCIONES ---

def cargar_costos_desde_parametros(filepath):
    """
    Carga el archivo de parámetros y extrae los costos variables y fijos
    en un DataFrame limpio.
    """
    try:
        df_params = pd.read_csv(filepath, sep=';', encoding='utf-8-sig')
        df_c = df_params[df_params['parametro'].str.startswith('c_')].copy()
        df_c['producto'] = df_c['parametro'].str.replace('c_', '').astype(int)
        df_c.rename(columns={'valor': 'costo_var'}, inplace=True)
        
        df_k = df_params[df_params['parametro'].str.startswith('K_')].copy()
        df_k['producto'] = df_k['parametro'].str.replace('K_', '').astype(int)
        df_k.rename(columns={'valor': 'costo_fijo'}, inplace=True)

        df_costos = pd.merge(df_c[['producto', 'costo_var']], df_k[['producto', 'costo_fijo']], on='producto')
        
        df_costos['costo_var'] = df_costos['costo_var'].str.replace(',', '.').astype(float)
        df_costos['costo_fijo'] = df_costos['costo_fijo'].str.replace(',', '.').astype(float)

        return df_costos
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de parámetros '{filepath}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Ocurrió un error al procesar el archivo de parámetros: {e}")
        sys.exit(1)


def calcular_utilidad():
    """
    Función principal que carga todos los datos, realiza los cálculos
    y muestra los resultados.
    """
    try:
        print("Cargando archivos de datos...")
        df_modelo = pd.read_csv(DATOS_MODELO_FILE, sep=';', decimal=',')
        df_demanda = pd.read_csv(DEMANDA_REAL_FILE, sep=',')
        df_costos = cargar_costos_desde_parametros(PARAMETROS_FILE)
        print("Archivos cargados correctamente.")
    
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo '{e.filename}'.")
        return
    except Exception as e:
        print(f"Ocurrió un error al leer los archivos: {e}")
        return

    # --- 2. Preparar y Unificar los Datos ---
    df_modelo.rename(columns={'semana_año': 'semana', 'producto_idx': 'producto', 'tienda_idx': 'tienda'}, inplace=True)
    df = pd.merge(df_modelo, df_demanda, on=['semana', 'tienda', 'producto'], how='left')
    df = pd.merge(df, df_costos, on='producto', how='left')
    df['demanda_real'] = df['demanda_real'].fillna(0)
    
    # Ordenar correctamente para aplicar la lógica de inventario
    df.sort_values(by=['tienda', 'producto', 'semana'], inplace=True)

    # --- 3. SIMULACIÓN (LÓGICA CORRECTA Y VECTORIZADA) ---
    print("Ejecutando simulación con la lógica correcta...")
    
    # El inventario inicial de la semana t es el final proyectado de la semana t-1
    # Usamos .shift(1) dentro de cada grupo (tienda, producto)
    inv_inicial_proyectado = df.groupby(['tienda', 'producto'])['inventario_final_sem1_horizonte'].shift(1)
    
    # Para la semana 1, el valor será NaN. Lo rellenamos con el inventario inicial original.
    df['inventario_inicial_real'] = inv_inicial_proyectado.fillna(df['inventario_inicial_sem1_horizonte'])
    
    # Ahora todos los demás cálculos se pueden hacer de forma vectorizada
    df['inventario_disponible'] = df['inventario_inicial_real'] + df['pedido_optimo_sem1_horizonte']
    df['ventas_reales'] = df[['inventario_disponible', 'demanda_real']].min(axis=1)
    df['inventario_final_real'] = df['inventario_disponible'] - df['ventas_reales']
    df['shortage_real'] = df['demanda_real'] - df['ventas_reales']
    
    print("Simulación completada.")

    # --- 4. Cálculos Financieros ---
    df['binaria_ordenar'] = (df['pedido_optimo_sem1_horizonte'] > 0).astype(int)
    df['venta'] = df['ventas_reales'] * df['precio_optimo']
    df['costo_orden'] = df['pedido_optimo_sem1_horizonte'] * df['costo_var']
    df['costo_orden_fijo'] = df['binaria_ordenar'] * df['costo_fijo']
    df['costo_inv'] = df['inventario_final_real'] * df['costo_var'] * TASA_COSTO_INVENTARIO
    df['costo_dem_ins'] = df['shortage_real'] * df['precio_optimo'] * TASA_PENALIDAD_SHORTAGE
    
    # --- 5. Generar Reporte Detallado ---
    df.sort_values(by=['producto', 'tienda', 'semana'], inplace=True)
    
    # Seleccionar y renombrar columnas para que coincidan EXACTAMENTE con tu Excel
    df_final = df[[
        'semana', 'tienda', 'producto', 'demanda_real', 
        'inventario_disponible', 'precio_optimo', 'pedido_optimo_sem1_horizonte',
        'shortage_real', 'costo_var', 'costo_fijo', 'binaria_ordenar',
        'venta', 'costo_orden', 'costo_orden_fijo', 'costo_dem_ins', 'costo_inv'
    ]].copy()
    
    df_final.rename(columns={
        'inventario_disponible': 'inventario_disp',
        'pedido_optimo_sem1_horizonte': 'orden_inv_opt',
        'shortage_real': 'demanda_insatisfecha'
    }, inplace=True)
    
    # Reordenar para que sea idéntico al screenshot
    columnas_ordenadas = [
        'semana', 'tienda', 'producto', 'demanda_real', 'inventario_disp', 'precio_optimo',
        'orden_inv_opt', 'demanda_insatisfecha', 'costo_var', 'costo_fijo', 'binaria_ordenar',
        'venta', 'costo_orden', 'costo_orden_fijo', 'costo_dem_ins', 'costo_inv'
    ]
    df_final = df_final[columnas_ordenadas]
    
    # Añadir la fila de totales y utilidad
    totales_df = df_final[['venta', 'costo_orden', 'costo_orden_fijo', 'costo_dem_ins', 'costo_inv']].sum().to_frame().T
    utilidad_total = totales_df['venta'].iloc[0] - totales_df.iloc[0, 1:].sum()
    
    # Guardar el archivo detallado
    df_final.to_csv(OUTPUT_FILE, sep=';', decimal=',', index=False, encoding='utf-8-sig')

    print(f"\nSe ha guardado un archivo detallado para depuración: '{OUTPUT_FILE}'")
    
    # --- 6. Mostrar Resumen Final en Consola ---
    print("\n" + "="*50)
    print("Resumen Financiero Total")
    print("="*50)
    print(f"{'Ingresos por Ventas':<40} {totales_df['venta'].iloc[0]:15,.2f}")
    print(f"{'Costo de Orden (Variable)':<40} {-totales_df['costo_orden'].iloc[0]:15,.2f}")
    print(f"{'Costo de Orden (Fijo)':<40} {-totales_df['costo_orden_fijo'].iloc[0]:15,.2f}")
    print(f"{'Costo de Inventario':<40} {-totales_df['costo_inv'].iloc[0]:15,.2f}")
    print(f"{'Costo de Demanda Insatisfecha (Shortage)':<40} {-totales_df['costo_dem_ins'].iloc[0]:15,.2f}")
    print(f"{'UTILIDAD TOTAL':<40} {utilidad_total:15,.2f}")
    

# --- Punto de Entrada del Script ---
if __name__ == "__main__":
    calcular_utilidad()