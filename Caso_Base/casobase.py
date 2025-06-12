import pandas as pd
import numpy as np
import os # For path handling

# --- Cargar datos ---
# Obtiene la ruta absoluta del directorio donde se encuentra este script
script_dir = os.path.dirname(os.path.abspath(__file__)) 
# Construye la ruta completa al archivo Excel
file_path = os.path.join(script_dir, "Datos_v1.xlsx")

# Crear el directorio de salida si no existe
output_dir = "Caso_Base_Resultados"
os.makedirs(output_dir, exist_ok=True)

try:
    xls = pd.ExcelFile(file_path)
    df_t1 = pd.read_excel(xls, sheet_name="Datos Tienda 1", skiprows=5)
    df_t2 = pd.read_excel(xls, sheet_name="Datos tienda 2", skiprows=5)
except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo '{file_path}'. Asegúrate de que esté en la misma carpeta que el script.")
    exit()


df_t1['Fecha'] = pd.to_datetime(df_t1.iloc[:, 1], errors='coerce')
df_t2['Fecha'] = pd.to_datetime(df_t2.iloc[:, 1], errors='coerce')

product_names = [f"Producto {i+1}" for i in range(10)]

# --- Entrenamiento con TODO el historial disponible ---
stock_base_t1, price_avg_t1 = {}, {}
stock_base_t2, price_avg_t2 = {}, {}

def calculate_base_policy(df, stock_base, price_avg):
    for i in range(10):
        demand_col = df.columns[2*i + 2]
        price_col = df.columns[2*i + 3]
        demand_data = pd.to_numeric(df[demand_col], errors='coerce').dropna()
        if demand_data.empty:
            stock_base[product_names[i]] = 0
        else:
            stock_base[product_names[i]] = np.percentile(demand_data, 90)
        
        price_data = pd.to_numeric(df[price_col], errors='coerce').dropna()
        if price_data.empty:
            price_avg[product_names[i]] = 0
        else:
            price_avg[product_names[i]] = price_data.mean()


calculate_base_policy(df_t1, stock_base_t1, price_avg_t1)
calculate_base_policy(df_t2, stock_base_t2, price_avg_t2)

# --- Obtener solo las primeras 4 semanas de 2025 ---
df_t1_test = df_t1[(df_t1['Fecha'] >= '2025-01-01') & (df_t1['Fecha'] < '2025-01-29')].reset_index(drop=True)
df_t2_test = df_t2[(df_t2['Fecha'] >= '2025-01-01') & (df_t2['Fecha'] < '2025-01-29')].reset_index(drop=True)

demanda_t1_4sem, demanda_t2_4sem = pd.DataFrame(), pd.DataFrame()
for i in range(10):
    demanda_t1_4sem[product_names[i]] = pd.to_numeric(df_t1_test.iloc[:, 2*i + 2], errors='coerce')
    demanda_t2_4sem[product_names[i]] = pd.to_numeric(df_t2_test.iloc[:, 2*i + 2], errors='coerce')

# --- Supuestos de costos ---
costo_unitario = {
    f"Producto {i+1}": c for i, c in enumerate([
        28.792, 20.792, 31.992, 52.792, 25.592, 44.8, 62.993, 46.4925, 32.3919, 17.592
    ])
}
costo_fijo_orden = {
    f"Producto {i+1}": c for i, c in enumerate([
        530, 530, 530, 320, 530, 530, 780, 530, 530, 530
    ])
}

# --- Simulación con precio dinámico y reposición parcial ---
def simular_base_stock_con_precio(demanda_df, stock_base_dict, precios_prom):
    semanas = len(demanda_df)
    productos = demanda_df.columns
    
    inventario_hist = {p: [stock_base_dict[p]] for p in productos} 
    
    quiebres, ordenes, precios, ingresos = {}, {}, {}, {}
    inventario_inicial_semana = {p: [] for p in productos}
    inventario_final_antes_repo_semana = {p: [] for p in productos}

    for p in productos:
        quiebres[p], ordenes[p], precios[p], ingresos[p] = [], [], [], []

    for t in range(semanas):
        for p in productos:
            demanda = demanda_df.loc[t, p] if pd.notna(demanda_df.loc[t, p]) else 0
            inv_ant = inventario_hist[p][-1] 

            inventario_inicial_semana[p].append(inv_ant)

            if inv_ant > 1.2 * stock_base_dict[p]:
                precio_actual = precios_prom[p] * 0.8
            elif inv_ant < 0.8 * stock_base_dict[p] and stock_base_dict[p] > 0 :
                precio_actual = precios_prom[p] * 1.2
            else:
                precio_actual = precios_prom[p]

            vendido = min(inv_ant, demanda)
            quiebre = max(demanda - inv_ant, 0)
            stock_post = inv_ant - vendido

            inventario_final_antes_repo_semana[p].append(stock_post)

            if stock_base_dict[p] > 0:
                if stock_post < 0.6 * stock_base_dict[p]:
                    nueva_orden = stock_base_dict[p]
                elif stock_post < 0.9 * stock_base_dict[p]:
                    nueva_orden = 0.5 * (stock_base_dict[p] - stock_post)
                else:
                    nueva_orden = 0.2 * (stock_base_dict[p] - stock_post)
                nueva_orden = max(0, nueva_orden)
            else:
                nueva_orden = 0

            inventario_hist[p].append(stock_post + nueva_orden)
            quiebres[p].append(quiebre)
            ordenes[p].append(nueva_orden)
            precios[p].append(precio_actual)
            ingresos[p].append(vendido * precio_actual)
            
    return (
        pd.DataFrame(quiebres),
        pd.DataFrame(ordenes),
        pd.DataFrame(precios),
        pd.DataFrame(ingresos),
        pd.DataFrame(inventario_inicial_semana),
        pd.DataFrame(inventario_final_antes_repo_semana)
    )

# --- Simular ---
q_t1, o_t1, precios_t1, ingresos_t1, inv_ini_t1, inv_fin_t1 = simular_base_stock_con_precio(demanda_t1_4sem, stock_base_t1, price_avg_t1)
q_t2, o_t2, precios_t2, ingresos_t2, inv_ini_t2, inv_fin_t2 = simular_base_stock_con_precio(demanda_t2_4sem, stock_base_t2, price_avg_t2)


# --- (La sección de KPIs se mantiene igual, se omite por brevedad pero está en tu código) ---
# ...
# --- Código de cálculo de KPIs va aquí ---
# ...
print("\n--- (Se omitió la impresión de KPIs por brevedad) ---")


# ##############################################################################
# # SECCIÓN MODIFICADA: PREPARAR DATOS PARA EL FORMATO CSV REQUERIDO            #
# ##############################################################################

print("\n--- Generando archivo de salida en formato 'datos.csv' ---")

datos_planificacion_semanal = []
num_semanas = len(demanda_t1_4sem)

# Tienda 1 (tienda_idx = 0)
for semana_idx in range(num_semanas):
    for prod_idx, prod_name in enumerate(product_names):
        demanda_real_sem = demanda_t1_4sem.loc[semana_idx, prod_name] if pd.notna(demanda_t1_4sem.loc[semana_idx, prod_name]) else 0
        datos_planificacion_semanal.append({
            "semana_año": semana_idx + 1, "producto_idx": prod_idx, "tienda_idx": 0,
            "precio_optimo": precios_t1.loc[semana_idx, prod_name],
            "pedido_optimo_sem1_horizonte": o_t1.loc[semana_idx, prod_name],
            "demanda_promedio_sem1_horizonte": demanda_real_sem,
            "shortage_promedio_sem1_horizonte": q_t1.loc[semana_idx, prod_name],
            "inventario_inicial_sem1_horizonte": inv_ini_t1.loc[semana_idx, prod_name],
            "inventario_final_sem1_horizonte": inv_fin_t1.loc[semana_idx, prod_name]
        })

# Tienda 2 (tienda_idx = 1)
for semana_idx in range(num_semanas):
    for prod_idx, prod_name in enumerate(product_names):
        demanda_real_sem = demanda_t2_4sem.loc[semana_idx, prod_name] if pd.notna(demanda_t2_4sem.loc[semana_idx, prod_name]) else 0
        datos_planificacion_semanal.append({
            "semana_año": semana_idx + 1, "producto_idx": prod_idx, "tienda_idx": 1,
            "precio_optimo": precios_t2.loc[semana_idx, prod_name],
            "pedido_optimo_sem1_horizonte": o_t2.loc[semana_idx, prod_name],
            "demanda_promedio_sem1_horizonte": demanda_real_sem,
            "shortage_promedio_sem1_horizonte": q_t2.loc[semana_idx, prod_name],
            "inventario_inicial_sem1_horizonte": inv_ini_t2.loc[semana_idx, prod_name],
            "inventario_final_sem1_horizonte": inv_fin_t2.loc[semana_idx, prod_name]
        })

df_planificacion = pd.DataFrame(datos_planificacion_semanal)

# ***** LÍNEA CLAVE AÑADIDA *****
# Ordenar el DataFrame para que coincida con el formato solicitado
df_planificacion.sort_values(by=['semana_año', 'producto_idx', 'tienda_idx'], inplace=True)


# Definir el orden de las columnas para el CSV para que coincida con el input
column_order = [
    "semana_año", "producto_idx", "tienda_idx", "precio_optimo",
    "pedido_optimo_sem1_horizonte", "demanda_promedio_sem1_horizonte",
    "shortage_promedio_sem1_horizonte", "inventario_inicial_sem1_horizonte",
    "inventario_final_sem1_horizonte"
]
df_planificacion = df_planificacion[column_order]

# Exportar a CSV con formato específico
csv_output_path = os.path.join(output_dir, "Planificacion_Semanal_Simulacion_Base.csv")
df_planificacion.to_csv(csv_output_path, sep=';', decimal=',', index=False, float_format='%.6f')

print(f"\n✅ Archivo de planificación semanal para análisis de utilidad generado: {csv_output_path}")
print("\nPrimeras filas del archivo de planificación (ordenado):")
print(df_planificacion.head(15)) # Imprimir más filas para ver el patrón