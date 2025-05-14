import pandas as pd
import numpy as np

# Cargar archivo Excel
file_path = "Datos v1.xlsx"
xls = pd.ExcelFile(file_path)

# Leer hojas de datos
df_t1 = pd.read_excel(xls, sheet_name="Datos Tienda 1", skiprows=5)
df_t2 = pd.read_excel(xls, sheet_name="Datos tienda 2", skiprows=5)

# Parsear fechas desde la columna de la izquierda
df_t1['Fecha'] = pd.to_datetime(df_t1.iloc[:, 1], errors='coerce')
df_t2['Fecha'] = pd.to_datetime(df_t2.iloc[:, 1], errors='coerce')

# Nombres de productos
product_names = [f"Producto {i+1}" for i in range(10)]

# Separar datos de entrenamiento (hasta 2024) y test (desde 2025)
df_t1_train = df_t1[df_t1['Fecha'] < '2025-01-01'].reset_index(drop=True)
df_t2_train = df_t2[df_t2['Fecha'] < '2025-01-01'].reset_index(drop=True)
df_t1_test = df_t1[df_t1['Fecha'] >= '2025-01-01'].reset_index(drop=True)
df_t2_test = df_t2[df_t2['Fecha'] >= '2025-01-01'].reset_index(drop=True)

# Calcular stock base (percentil 90) y precio promedio
stock_base_t1, price_avg_t1 = {}, {}
stock_base_t2, price_avg_t2 = {}, {}

def calculate_base_policy(df, stock_base, price_avg):
    for i in range(10):
        demand_col = df.columns[2*i + 2]
        price_col = df.columns[2*i + 3]
        demand_data = pd.to_numeric(df[demand_col], errors='coerce').dropna()
        price_data = pd.to_numeric(df[price_col], errors='coerce').dropna()
        stock_base[product_names[i]] = np.percentile(demand_data, 90)
        price_avg[product_names[i]] = price_data.mean()

calculate_base_policy(df_t1_train, stock_base_t1, price_avg_t1)
calculate_base_policy(df_t2_train, stock_base_t2, price_avg_t2)

# Extraer demandas de 2025
demanda_t1_2025, demanda_t2_2025 = pd.DataFrame(), pd.DataFrame()
for i in range(10):
    demanda_t1_2025[f"Producto {i+1}"] = pd.to_numeric(df_t1_test.iloc[:, 2*i + 2], errors='coerce')
    demanda_t2_2025[f"Producto {i+1}"] = pd.to_numeric(df_t2_test.iloc[:, 2*i + 2], errors='coerce')

# Simulación base stock
def simular_base_stock(demanda_df, stock_base_dict):
    semanas = len(demanda_df)
    productos = demanda_df.columns
    inventario = {p: [stock_base_dict[p]] for p in productos}
    quiebres = {p: [] for p in productos}
    ordenes = {p: [] for p in productos}
    for t in range(semanas):
        for p in productos:
            demanda = demanda_df.loc[t, p] if pd.notna(demanda_df.loc[t, p]) else 0
            inv_ant = inventario[p][-1]
            vendido = min(inv_ant, demanda)
            quiebre = max(demanda - inv_ant, 0)
            stock_post = inv_ant - vendido
            nueva_orden = stock_base_dict[p] - stock_post
            inventario[p].append(stock_post + nueva_orden)
            quiebres[p].append(quiebre)
            ordenes[p].append(nueva_orden)
    return pd.DataFrame(quiebres), pd.DataFrame(ordenes)

quiebres_t1, ordenes_t1 = simular_base_stock(demanda_t1_2025, stock_base_t1)
quiebres_t2, ordenes_t2 = simular_base_stock(demanda_t2_2025, stock_base_t2)

# Supuestos de costos
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

# Cálculo de KPIs
def calcular_kpis(demanda_df, quiebres_df, ordenes_df, stock_base, precios_prom, tienda):
    semanas = len(demanda_df)
    productos = demanda_df.columns
    utilidad_total = demanda_total = demanda_satisfecha = 0
    total_quiebres = total_ordenes = costo_total_inventario = costo_total_quiebre = costo_total_orden = 0
    dias_prom_inventario = []
    for p in productos:
        precio = precios_prom[p]
        costo = costo_unitario[p]
        inv_cost = 0.1 * costo
        quiebre_cost = 0.1 * precio
        orden_fijo = costo_fijo_orden[p]
        demanda = demanda_df[p].fillna(0)
        quiebre = quiebres_df[p]
        ordenes = ordenes_df[p]
        venta = demanda - quiebre
        ingreso = venta * precio
        costo_q = quiebre * quiebre_cost
        costo_inv = stock_base[p] * inv_cost * semanas
        ordenes_realizadas = (ordenes > 0).sum()
        costo_o = ordenes_realizadas * orden_fijo
        utilidad = ingreso.sum() - costo_q.sum() - costo_inv - costo_o
        utilidad_total += utilidad
        demanda_total += demanda.sum()
        demanda_satisfecha += venta.sum()
        total_quiebres += quiebre.sum()
        total_ordenes += ordenes_realizadas
        costo_total_inventario += costo_inv
        costo_total_quiebre += costo_q.sum()
        costo_total_orden += costo_o
        dias_prom_inventario.append(stock_base[p] / max(1, demanda.mean()))
    nivel_servicio = (demanda_satisfecha / demanda_total) * 100 if demanda_total > 0 else 0
    dias_prom = np.mean(dias_prom_inventario)
    return {
        "Tienda": tienda,
        "Utilidad Total (2025)": utilidad_total,
        "Demanda Total (2025)": demanda_total,
        "Demanda Satisfecha (2025)": demanda_satisfecha,
        "Demanda Insatisfecha (2025)": total_quiebres,
        "Nivel de Servicio (%)": nivel_servicio,
        "Días Prom. Inventario": dias_prom,
        "Órdenes Emitidas": total_ordenes,
        "Costo Inventario": costo_total_inventario,
        "Costo Ordenamiento": costo_total_orden,
        "Costo Demanda Insatisfecha": costo_total_quiebre
    }

# Calcular KPIs solo para 2025
kpi_t1 = calcular_kpis(demanda_t1_2025, quiebres_t1, ordenes_t1, stock_base_t1, price_avg_t1, "Tienda 1")
kpi_t2 = calcular_kpis(demanda_t2_2025, quiebres_t2, ordenes_t2, stock_base_t2, price_avg_t2, "Tienda 2")

df_kpis_2025 = pd.DataFrame([kpi_t1, kpi_t2])

# Guardar en Excel con hojas adicionales
with pd.ExcelWriter("KPIs_2025_BasePolicy_Completo.xlsx", engine='xlsxwriter') as writer:
    df_kpis_2025.to_excel(writer, sheet_name="Resumen KPIs", index=False)

    # Stock y precios promedio
    pd.DataFrame({
        "Producto": product_names,
        "Stock Base T1": [stock_base_t1[p] for p in product_names],
        "Precio Prom. T1": [price_avg_t1[p] for p in product_names],
        "Stock Base T2": [stock_base_t2[p] for p in product_names],
        "Precio Prom. T2": [price_avg_t2[p] for p in product_names],
    }).to_excel(writer, sheet_name="Stock y Precios", index=False)

    # Datos semana a semana para 2025
    demanda_t1_2025.to_excel(writer, sheet_name="Demanda 2025 T1", index=False)
    demanda_t2_2025.to_excel(writer, sheet_name="Demanda 2025 T2", index=False)
    quiebres_t1.to_excel(writer, sheet_name="Quiebres 2025 T1", index=False)
    ordenes_t1.to_excel(writer, sheet_name="Órdenes 2025 T1", index=False)
    quiebres_t2.to_excel(writer, sheet_name="Quiebres 2025 T2", index=False)
    ordenes_t2.to_excel(writer, sheet_name="Órdenes 2025 T2", index=False)

# Mostrar por consola
print("\n✅ Archivo generado: KPIs_2025_BasePolicy_Completo.xlsx")
print("\nResumen de KPI’s por tienda:\n")
print(df_kpis_2025.round(2))


