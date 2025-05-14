import pandas as pd
import numpy as np

# --- Cargar archivos ---
kpis_base = pd.read_excel("comparación/KPIs_2025_Sim_4_Semanas.xlsx", sheet_name="Resumen KPIs")
df_politica = pd.read_excel("comparación/resultados_optimizacion_saa_Enero2025_s10_t4_TL600_SL95_UnmetCost.xlsx", sheet_name="Politica_Optima_Enero_SAA")

# --- Cargar demanda real ---
file_path = "comparación/Datos_v1.xlsx"
df_t1 = pd.read_excel(file_path, sheet_name="Datos Tienda 1", skiprows=5)
df_t2 = pd.read_excel(file_path, sheet_name="Datos tienda 2", skiprows=5)

df_t1["Fecha"] = pd.to_datetime(df_t1.iloc[:, 1], errors="coerce")
df_t2["Fecha"] = pd.to_datetime(df_t2.iloc[:, 1], errors="coerce")

df_t1_4sem = df_t1[(df_t1["Fecha"] >= "2025-01-01") & (df_t1["Fecha"] < "2025-01-29")].reset_index(drop=True)
df_t2_4sem = df_t2[(df_t2["Fecha"] >= "2025-01-01") & (df_t2["Fecha"] < "2025-01-29")].reset_index(drop=True)

product_names = [f"Producto {i+1}" for i in range(10)]
demanda_real = {"Tienda 1": pd.DataFrame(), "Tienda 2": pd.DataFrame()}
for i in range(10):
    demanda_real["Tienda 1"][product_names[i]] = pd.to_numeric(df_t1_4sem.iloc[:, 2*i + 2], errors='coerce')
    demanda_real["Tienda 2"][product_names[i]] = pd.to_numeric(df_t2_4sem.iloc[:, 2*i + 2], errors='coerce')

# --- Parámetros de costo ---
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

# --- KPI desde política SAA usando demanda real ---
def calcular_kpis_desde_politica(df_politica, demanda_real_dict):
    productos = sorted(df_politica['producto'].unique())
    semanas = df_politica['semana_plan'].max()

    utilidad_total = demanda_total = demanda_satisfecha = 0
    total_quiebres = total_ordenes = costo_total_inv = costo_total_q = costo_total_o = 0
    dias_prom_inventario = []

    for prod in productos:
        prod_nombre = f"Producto {prod}"
        df_p = df_politica[df_politica["producto"] == prod]
        precio_prom = df_p['precio_optimo_saa'].mean()
        ordenes_realizadas = df_p[df_p["orden_optima_saa"] > 0].shape[0]

        # Separar por tienda
        ventas_t1 = df_p[df_p["tienda"] == 1]['prom_ventas_saa'].values
        ventas_t2 = df_p[df_p["tienda"] == 2]['prom_ventas_saa'].values

        demanda_real_t1 = demanda_real_dict["Tienda 1"][prod_nombre].fillna(0).values
        demanda_real_t2 = demanda_real_dict["Tienda 2"][prod_nombre].fillna(0).values

        ventas_finales_t1 = np.minimum(demanda_real_t1, ventas_t1)
        ventas_finales_t2 = np.minimum(demanda_real_t2, ventas_t2)

        quiebres_t1 = demanda_real_t1 - ventas_finales_t1
        quiebres_t2 = demanda_real_t2 - ventas_finales_t2

        ingreso = np.sum(ventas_finales_t1 + ventas_finales_t2) * precio_prom
        costo_q = np.sum(quiebres_t1 + quiebres_t2) * 0.1 * precio_prom
        inv_cost = 0.1 * costo_unitario[prod_nombre]
        demanda_prom = np.mean(np.concatenate([demanda_real_t1, demanda_real_t2]))
        costo_inv = inv_cost * semanas * demanda_prom
        costo_o = ordenes_realizadas * costo_fijo_orden[prod_nombre]

        utilidad = ingreso - costo_q - costo_inv - costo_o

        utilidad_total += utilidad
        demanda_total += np.sum(demanda_real_t1 + demanda_real_t2)
        demanda_satisfecha += np.sum(ventas_finales_t1 + ventas_finales_t2)
        total_quiebres += np.sum(quiebres_t1 + quiebres_t2)
        total_ordenes += ordenes_realizadas
        costo_total_inv += costo_inv
        costo_total_q += costo_q
        costo_total_o += costo_o
        dias_prom_inventario.append(demanda_prom / max(1, demanda_prom))

    nivel_servicio = (demanda_satisfecha / demanda_total) * 100 if demanda_total > 0 else 0
    dias_prom = np.mean(dias_prom_inventario)

    return {
        "Utilidad Total (4 sem)": utilidad_total,
        "Demanda Total": demanda_total,
        "Demanda Satisfecha": demanda_satisfecha,
        "Demanda Insatisfecha": total_quiebres,
        "Nivel de Servicio (%)": nivel_servicio,
        "Días Prom. Inventario": dias_prom,
        "Órdenes Emitidas": total_ordenes,
        "Costo Inventario": costo_total_inv,
        "Costo Ordenamiento": costo_total_o,
        "Costo Demanda Insatisfecha": costo_total_q
    }

# --- KPIs agregados (por total) ---
kpis_base_total = kpis_base.drop(columns=["Tienda"]).sum(numeric_only=True)
kpis_base_total["Nivel de Servicio (%)"] = (kpis_base["Demanda Satisfecha"].sum() / kpis_base["Demanda Total"].sum()) * 100
kpis_base_total["Días Prom. Inventario"] = kpis_base["Días Prom. Inventario"].mean()
kpis_base_total_df = pd.DataFrame(kpis_base_total).T

kpis_saa_total = calcular_kpis_desde_politica(df_politica, demanda_real)
kpis_saa_total_df = pd.DataFrame(kpis_saa_total, index=[0])

# --- Comparación global ---
def comparar_kpis(df_base, df_saa):
    row_base = df_base.iloc[0]
    row_saa = df_saa.iloc[0]
    resultados = []

    for col in df_base.columns:
        base_val = row_base[col]
        saa_val = row_saa[col]
        diff = saa_val - base_val
        perc = (diff / base_val * 100) if base_val != 0 else np.nan
        resultados.append({
            "Métrica": col,
            "Caso Base": base_val,
            "Modelo SAA": saa_val,
            "Diferencia Absoluta": diff,
            "Diferencia (%)": perc
        })
    return pd.DataFrame(resultados)

df_comp = comparar_kpis(kpis_base_total_df, kpis_saa_total_df)

# --- Guardar archivos ---
with pd.ExcelWriter("comparación/Comparacion_KPIs_Totales.xlsx") as writer:
    df_comp.to_excel(writer, sheet_name="Comparación Total", index=False)
    kpis_base_total_df.to_excel(writer, sheet_name="Total Caso Base", index=False)
    kpis_saa_total_df.to_excel(writer, sheet_name="Total Modelo SAA", index=False)

print("✅ Comparación total guardada en: comparación/Comparacion_KPIs_Totales.xlsx")
