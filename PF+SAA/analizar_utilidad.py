import pandas as pd
import os
import numpy as np
#Constantes
TASA_COSTO_INVENTARIO = 0.10
TASA_PENALIDAD_SHORTAGE = 0.10

#carga de archivos
DATOS_MODELO_FILE = "resultados/Planificacion_Semanal_Optima_PF.csv"
PARAMETROS_FILE = "parametros/General_Parameters.csv"
OUTPUT_FILE = "resultados/resultados_detallados_final.csv"
DEMANDA_REAL_PATH = "resultados/demanda_real.csv"

def cargar_costos_desde_parametros(filepath):
    df_params = pd.read_csv(filepath, sep=';', decimal=',')
    
    df_c = df_params[df_params['parametro'].str.startswith('c_')].copy()
    df_c['producto'] = df_c['parametro'].str.replace('c_', '', regex=False).astype(int)
    df_c.rename(columns={'valor': 'costo_var'}, inplace=True)
    
    df_k = df_params[df_params['parametro'].str.startswith('K_')].copy()
    df_k['producto'] = df_k['parametro'].str.replace('K_', '', regex=False).astype(int)
    df_k.rename(columns={'valor': 'costo_fijo'}, inplace=True)

    df_costos = pd.merge(df_c[['producto', 'costo_var']], df_k[['producto', 'costo_fijo']], on='producto')
    
    df_costos['costo_var'] = df_costos['costo_var'].astype(float)
    df_costos['costo_fijo'] = df_costos['costo_fijo'].astype(float)
    
    return df_costos

def guardar_demanda_real_simulada(semana, mu_dict, sigma_dict, n_muestras=3, guardar_mu=True, ruta_csv=None):
    """
    Simula demanda real a partir de los parámetros mu y sigma, y la guarda como CSV.
    Si el archivo ya existe, agrega los nuevos datos al final.
    Puedes cambiar la ruta del archivo usando 'ruta_csv'.
    """
    if ruta_csv is None:
        ruta_csv = "resultados/demanda_real.csv"

    registros = []

    for (q, l, t_h), mu in mu_dict.items():
        if t_h != 1:
            continue

        sigma = sigma_dict.get((q, l, t_h), 0.0)
        muestras = [max(0, int(x)) for x in np.random.normal(mu, sigma, n_muestras)]
        demanda_real = round(np.mean(muestras), 2)

        fila = {
            "semana_año": semana,
            "producto_idx": q,
            "tienda_idx": l,
            "demanda_real": demanda_real
        }

        if guardar_mu:
            fila["demanda_estimacion_modelo"] = round(mu, 2)

        registros.append(fila)

    df_nueva = pd.DataFrame(registros)

    os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)

    if os.path.exists(ruta_csv):
        df_antiguo = pd.read_csv(ruta_csv)
        df_total = pd.concat([df_antiguo, df_nueva], ignore_index=True)
    else:
        df_total = df_nueva

    df_total.to_csv(ruta_csv, index=False)
    print(f"Demanda real simulada guardada en {ruta_csv}")
    

def calcular_utilidad_total(path_csv=DATOS_MODELO_FILE, path_parametros=PARAMETROS_FILE, exportar_csv=True):
    print("Cargando archivos...")
    df_modelo = pd.read_csv(path_csv, sep=';', decimal='.')
    df_costos = cargar_costos_desde_parametros(path_parametros)
    print("Archivos cargados correctamente.\n")

    df_modelo.rename(columns={'semana_año': 'semana', 'producto_idx': 'producto', 'tienda_idx': 'tienda'}, inplace=True)
    df = pd.merge(df_modelo, df_costos, on='producto', how='left')

    if os.path.exists(DEMANDA_REAL_PATH):
        print("Usando demanda real desde CSV...")
        df_demanda_real = pd.read_csv(DEMANDA_REAL_PATH)
        df_demanda_real.rename(columns={'semana_año': 'semana', 'producto_idx': 'producto', 'tienda_idx': 'tienda'}, inplace=True)
        df = pd.merge(df, df_demanda_real[['semana', 'producto', 'tienda', 'demanda_real']], 
                      on=['semana', 'producto', 'tienda'], how='left')
        df['demanda_real'] = df['demanda_real'].fillna(df['demanda_promedio_sem1_horizonte'])
    else:
        print("No se encontró demanda_real.csv. Usando demanda promedio.")
        df['demanda_real'] = df['demanda_promedio_sem1_horizonte']

    df.sort_values(by=['tienda', 'producto', 'semana'], inplace=True)

    df['inventario_inicial_real'] = np.nan
    df['inventario_disp'] = np.nan
    df['ventas_reales'] = np.nan
    df['inventario_final_real'] = np.nan
    df['shortage_real'] = np.nan

    for (tienda, producto), df_sub in df.groupby(['tienda', 'producto']):
        df_sub = df_sub.sort_values('semana').copy()
        inventario_inicial = df_sub.iloc[0]['inventario_inicial_sem1_horizonte']

        inventarios_iniciales = []
        inventarios_disponibles = []
        inventarios_finales = []
        ventas = []
        shortages = []

        for i, row in df_sub.iterrows():
            pedido = row['pedido_optimo_sem1_horizonte']
            demanda = row['demanda_real']

            inventario_disp = inventario_inicial + pedido
            venta = min(inventario_disp, demanda)
            inventario_final = inventario_disp - venta
            shortage = max(demanda - venta, 0)

            inventarios_iniciales.append(inventario_inicial)
            inventarios_disponibles.append(inventario_disp)
            ventas.append(venta)
            inventarios_finales.append(inventario_final)
            shortages.append(shortage)

            inventario_inicial = inventario_final

        idxs = df_sub.index
        df.loc[idxs, 'inventario_inicial_real'] = inventarios_iniciales
        df.loc[idxs, 'inventario_disp'] = inventarios_disponibles
        df.loc[idxs, 'ventas_reales'] = ventas
        df.loc[idxs, 'inventario_final_real'] = inventarios_finales
        df.loc[idxs, 'shortage_real'] = shortages

    df['binaria_ordenar'] = (df['pedido_optimo_sem1_horizonte'] > 1e-6).astype(int)
    df['venta'] = df['ventas_reales'] * df['precio_optimo']
    df['costo_orden'] = df['pedido_optimo_sem1_horizonte'] * df['costo_var']
    df['costo_orden_fijo'] = df['binaria_ordenar'] * df['costo_fijo']
    df['costo_inv'] = df['inventario_final_real'] * df['costo_var'] * TASA_COSTO_INVENTARIO
    df['costo_dem_ins'] = df['shortage_real'] * df['precio_optimo'] * TASA_PENALIDAD_SHORTAGE

    df_final = df[[
        'semana', 'tienda', 'producto', 'demanda_real', 
        'inventario_disp', 'precio_optimo', 'pedido_optimo_sem1_horizonte',
        'shortage_real', 'costo_var', 'costo_fijo', 'binaria_ordenar',
        'venta', 'costo_orden', 'costo_orden_fijo', 'costo_dem_ins', 'costo_inv'
    ]].copy()

    df_final.rename(columns={
        'pedido_optimo_sem1_horizonte': 'orden_inv_opt',
        'shortage_real': 'demanda_insatisfecha'
    }, inplace=True)

    #limpieza y redondeo de cols
    columnas_redondear = [
        'orden_inv_opt', 'venta', 'costo_orden', 
        'costo_orden_fijo', 'costo_inv', 'costo_dem_ins'
    ]

    for col in columnas_redondear:
        df_final[col] = df_final[col].apply(lambda x: 0 if abs(x) < 1e-6 else round(x, 2))

    os.makedirs("resultados", exist_ok=True)
    if exportar_csv:
        df_final.to_csv(OUTPUT_FILE, sep=';', decimal=',', index=False, encoding='utf-8-sig')
        print(f"✅ Resultado detallado guardado en: {OUTPUT_FILE}\n")

    totales = df_final[['venta', 'costo_orden', 'costo_orden_fijo', 'costo_inv', 'costo_dem_ins']].sum()
    utilidad_total = totales['venta'] - totales[['costo_orden', 'costo_orden_fijo', 'costo_inv', 'costo_dem_ins']].sum()

    print("="*50)
    print("Resumen Financiero Total")
    print("="*50)
    print(f"{'Ingresos por Ventas':<40} {totales['venta']:15,.2f}")
    print(f"{'Costo de Orden (Variable)':<40} {-totales['costo_orden']:15,.2f}")
    print(f"{'Costo de Orden (Fijo)':<40} {-totales['costo_orden_fijo']:15,.2f}")
    print(f"{'Costo de Inventario':<40} {-totales['costo_inv']:15,.2f}")
    print(f"{'Costo de Demanda Insatisfecha (Shortage)':<40} {-totales['costo_dem_ins']:15,.2f}")
    print(f"{'UTILIDAD TOTAL':<40} {utilidad_total:15,.2f}")

    return utilidad_total, df_final.groupby('semana')['venta'].sum()

if __name__ == "__main__":
    calcular_utilidad_total()
