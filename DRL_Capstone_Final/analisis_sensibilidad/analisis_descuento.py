import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath("."))

from optimizador import calcular_resultados_optimizacion, load_static_params_once
from analizar_utilidad import guardar_demanda_real_simulada, cargar_costos_desde_parametros

# --- CONSTANTES ---
TASA_COSTO_INVENTARIO = 0.10
TASA_PENALIDAD_SHORTAGE = 0.10

# Parámetros
SEMANA_DESCUENTO = 1
PORCENTAJE_DESCUENTO = 0.8
CANTIDAD_PRODUCTOS_DESC = 6

# Archivos
ARCHIVO_BASE = "resultados/resultados_detallados_final.csv"
ARCHIVO_PARAMETROS = "parametros/General_Parameters.csv"
ARCHIVO_CSV_FINAL = "analisis_sensibilidad/resultados_detallados_descuento.csv"
ARCHIVO_DEMANDA_ORIGINAL = "resultados/demanda_real.csv"
ARCHIVO_DEMANDA_DESC = "analisis_sensibilidad/demanda_real_descuento.csv"
ARCHIVO_DEMANDA_FINAL = "analisis_sensibilidad/demanda_real_completa.csv"

# === FLUJO PRINCIPAL ===

df = pd.read_csv(ARCHIVO_BASE, sep=';', decimal=',')

static_params = load_static_params_once("parametros", Q_val=10, L_val=2)
precios_base = static_params["precios_base_np"]
inventario_actual = static_params["I_initial_global"]

productos_disponibles = df[df['semana'] == SEMANA_DESCUENTO]['producto'].unique()
productos_elegidos = np.random.choice(productos_disponibles, size=CANTIDAD_PRODUCTOS_DESC, replace=False)

df_modificado = df.copy()
for q in productos_elegidos:
    for l in range(2):
        idx = (df_modificado['semana'] == SEMANA_DESCUENTO) & (df_modificado['producto'] == q) & (df_modificado['tienda'] == l)
        df_modificado.loc[idx, 'precio_optimo'] *= (1 - PORCENTAJE_DESCUENTO)

precios_modificados = np.zeros_like(precios_base)
for q in range(10):
    for l in range(2):
        filtro = (df_modificado['semana'] == SEMANA_DESCUENTO) & (df_modificado['producto'] == q) & (df_modificado['tienda'] == l)
        precio = df_modificado.loc[filtro, 'precio_optimo']
        precios_modificados[q, l] = precio.values[0] if not precio.empty else precios_base[q, l]

res = calcular_resultados_optimizacion(
    precios_modificados,
    inventario_actual,
    semana_año_actual_optimizando=SEMANA_DESCUENTO,
    ruta_datos="parametros",
    Q_val=10,
    L_val=2,
    use_eval_seed=True
)

mu_dict = res["mu_calculado_horizonte"]
sigma_dict = res["sigma_calculado_horizonte"]

guardar_demanda_real_simulada(
    semana=SEMANA_DESCUENTO,
    mu_dict=mu_dict,
    sigma_dict=sigma_dict,
    n_muestras=3,
    guardar_mu=True,
    ruta_csv=ARCHIVO_DEMANDA_DESC
)

# Cargar y combinar demandas
df_demanda_original = pd.read_csv(ARCHIVO_DEMANDA_ORIGINAL)
df_demanda_nueva = pd.read_csv(ARCHIVO_DEMANDA_DESC)

df_demanda_final = pd.concat([
    df_demanda_original[df_demanda_original['semana_año'] != SEMANA_DESCUENTO],
    df_demanda_nueva
], ignore_index=True)
df_demanda_final.to_csv(ARCHIVO_DEMANDA_FINAL, index=False)

# Asegurar tipos para merge
for col in ['semana_año', 'producto_idx', 'tienda_idx']:
    df_demanda_final[col] = df_demanda_final[col].astype(int)
for col in ['semana', 'producto', 'tienda']:
    df_modificado[col] = df_modificado[col].astype(int)

# Merge demanda antes y después
df_demanda_original = df_demanda_original.rename(columns={
    'semana_año': 'semana', 'producto_idx': 'producto', 'tienda_idx': 'tienda', 'demanda_real': 'demanda_real_antes'
})
df_demanda_final = df_demanda_final.rename(columns={
    'semana_año': 'semana', 'producto_idx': 'producto', 'tienda_idx': 'tienda', 'demanda_real': 'demanda_real_despues'
})

df_modificado = df_modificado.merge(df_demanda_original[['semana', 'producto', 'tienda', 'demanda_real_antes']],
                                    on=['semana', 'producto', 'tienda'], how='left')
df_modificado = df_modificado.merge(df_demanda_final[['semana', 'producto', 'tienda', 'demanda_real_despues']],
                                    on=['semana', 'producto', 'tienda'], how='left')

# Calcular inventario_disp_despues semana a semana
df_modificado['inventario_inicial_despues'] = np.nan
df_modificado['inventario_disp_despues'] = np.nan
df_modificado['ventas_reales_despues'] = np.nan
df_modificado['inventario_final_despues'] = np.nan

for (tienda, producto), sub_df in df_modificado.groupby(['tienda', 'producto']):
    sub_df = sub_df.sort_values('semana').copy()
    inventario_inicial = sub_df.iloc[0]['inventario_disp'] - sub_df.iloc[0]['orden_inv_opt']


    inv_inic_list, inv_disp_list, ventas_list, inv_final_list = [], [], [], []

    for i, row in sub_df.iterrows():
        pedido = row['orden_inv_opt']
        demanda = row['demanda_real_despues']
        inv_disp = inventario_inicial + pedido
        venta = min(inv_disp, demanda)
        inv_final = inv_disp - venta

        inv_inic_list.append(inventario_inicial)
        inv_disp_list.append(inv_disp)
        ventas_list.append(venta)
        inv_final_list.append(inv_final)

        inventario_inicial = inv_final  # para la próxima semana

    idxs = sub_df.index
    df_modificado.loc[idxs, 'inventario_inicial_despues'] = inv_inic_list
    df_modificado.loc[idxs, 'inventario_disp_despues'] = inv_disp_list
    df_modificado.loc[idxs, 'ventas_reales_despues'] = ventas_list
    df_modificado.loc[idxs, 'inventario_final_despues'] = inv_final_list

# Calcular shortage_despues
df_modificado['shortage_despues'] = (df_modificado['demanda_real_despues'] - df_modificado['inventario_disp_despues']).clip(lower=0)

# Guardar CSV actualizado con nueva columna
df_modificado.to_csv(ARCHIVO_CSV_FINAL, sep=';', decimal=',', index=False)

# Calcular utilidad usando las variables _despues
df_util = df_modificado.copy()
df_util['binaria_ordenar'] = (df_util['orden_inv_opt'] > 1e-6).astype(int)
df_util['venta'] = df_util['ventas_reales_despues'] * df_util['precio_optimo']
df_util['costo_orden'] = df_util['orden_inv_opt'] * df_util['costo_var']
df_util['costo_orden_fijo'] = df_util['binaria_ordenar'] * df_util['costo_fijo']
df_util['costo_inv'] = df_util['inventario_final_despues'] * df_util['costo_var'] * TASA_COSTO_INVENTARIO
df_util['costo_dem_ins'] = df_util['shortage_despues'] * df_util['precio_optimo'] * TASA_PENALIDAD_SHORTAGE

totales = df_util[['venta', 'costo_orden', 'costo_orden_fijo', 'costo_inv', 'costo_dem_ins']].sum()
utilidad_total = totales['venta'] - totales[['costo_orden', 'costo_orden_fijo', 'costo_inv', 'costo_dem_ins']].sum()

df_base = pd.read_csv(ARCHIVO_BASE, sep=';', decimal=',')
totales_base = df_base[['venta', 'costo_orden', 'costo_orden_fijo', 'costo_inv', 'costo_dem_ins']].sum()
utilidad_base_total = totales_base['venta'] - totales_base[['costo_orden', 'costo_orden_fijo', 'costo_inv', 'costo_dem_ins']].sum()

# === PRINT FINAL ===
print("\n=== RESULTADOS ANÁLISIS DE SENSIBILIDAD ===")
print(f"📅 Semana de descuento aplicada: {SEMANA_DESCUENTO}")
print(f"📉 Porcentaje de descuento aplicado: {PORCENTAJE_DESCUENTO * 100:.1f}%")
print(f"🎯 Productos afectados: {sorted(productos_elegidos.tolist())}\n")

print("📊 Comparación de precios, demanda y shortage antes y después (semana afectada):")
print(f"{'Producto':<10} {'Tienda':<8} {'Precio antes':<15} {'Precio después':<17} {'Demanda antes':<17} {'Demanda después':<17} {'Shortage después':<17}")
print("-" * 120)

for q in sorted(productos_elegidos.tolist()):
    for l in range(2):
        row = df_modificado[(df_modificado['semana'] == SEMANA_DESCUENTO) &
                            (df_modificado['producto'] == q) &
                            (df_modificado['tienda'] == l)]
        if not row.empty:
            precio_antes = df[(df['semana'] == SEMANA_DESCUENTO) & (df['producto'] == q) & (df['tienda'] == l)]['precio_optimo'].values[0]
            precio_despues = row['precio_optimo'].values[0]
            demanda_antes = row['demanda_real_antes'].values[0]
            demanda_despues = row['demanda_real_despues'].values[0]
            shortage_despues = row['shortage_despues'].values[0]
            print(f"{q:<10} {l:<8} {precio_antes:<15.2f} {precio_despues:<17.2f} {demanda_antes:<17.2f} {demanda_despues:<17.2f} {shortage_despues:<17.2f}")
        else:
            print(f"{q:<10} {l:<8} {'N/A':<15} {'N/A':<17} {'N/A':<17} {'N/A':<17} {'N/A':<17}")

print("\n💰 Comparación de utilidad:")
print(f"  Utilidad total original (sin descuento): ${utilidad_base_total:,.2f}")
print(f"  Utilidad total con descuento aplicado:   ${utilidad_total:,.2f}")
print(f"  Diferencia absoluta de utilidad:         ${utilidad_total - utilidad_base_total:,.2f}")
print("============================================")
