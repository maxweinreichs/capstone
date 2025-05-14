import pandas as pd
import numpy as np

# --- Cargar datos ---
file_path = "Caso_Base/Datos_v1.xlsx"
xls = pd.ExcelFile(file_path)

df_t1 = pd.read_excel(xls, sheet_name="Datos Tienda 1", skiprows=5)
df_t2 = pd.read_excel(xls, sheet_name="Datos tienda 2", skiprows=5)

df_t1['Fecha'] = pd.to_datetime(df_t1.iloc[:, 1], errors='coerce')
df_t2['Fecha'] = pd.to_datetime(df_t2.iloc[:, 1], errors='coerce')

product_names = [f"Producto {i+1}" for i in range(10)]

# --- Cálculo de stock base y precio promedio con TODO el dataset ---
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

calculate_base_policy(df_t1, stock_base_t1, price_avg_t1)
calculate_base_policy(df_t2, stock_base_t2, price_avg_t2)

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

# --- Simulación de 4 semanas futuras ---
def simular_4_semanas_futuras(stock_base_dict, precios_prom):
    productos = list(stock_base_dict.keys())
    semanas = 4
    inventario = {p: [stock_base_dict[p]] for p in productos}
    quiebres, ordenes, precios, ingresos = {}, {}, {}, {}

    for p in productos:
        quiebres[p], ordenes[p], precios[p], ingresos[p] = [], [], [], []

    for t in range(semanas):
        for p in productos:
            demanda = np.random.poisson(lam=stock_base_dict[p] * 0.7)  # demanda simulada conservadora
            inv_ant = inventario[p][-1]

            # --- Precio dinámico basado en inventario ---
            if inv_ant > 1.2 * stock_base_dict[p]:
                precio = precios_prom[p] * 0.8
            elif inv_ant < 0.8 * stock_base_dict[p]:
                precio = precios_prom[p] * 1.2
            else:
                precio = precios_prom[p]

            vendido = min(inv_ant, demanda)
            quiebre = max(demanda - inv_ant, 0)
            stock_post = inv_ant - vendido

            # --- Política de reabastecimiento parcial escalonada ---
            if stock_post < 0.6 * stock_base_dict[p]:
                nueva_orden = stock_base_dict[p]
            elif stock_post < 0.9 * stock_base_dict[p]:
                nueva_orden = 0.5 * (stock_base_dict[p] - stock_post)
            else:
                nueva_orden = 0.2 * (stock_base_dict[p] - stock_post)

            inventario[p].append(stock_post + nueva_orden)
            quiebres[p].append(quiebre)
            ordenes[p].append(nueva_orden)
            precios[p].append(precio)
            ingresos[p].append(vendido * precio)

    return (
        pd.DataFrame(quiebres),
        pd.DataFrame(ordenes),
        pd.DataFrame(precios),
        pd.DataFrame(ingresos)
    )

# --- Simulación ---
q_t1, o_t1, precios_t1, ingresos_t1 = simular_4_semanas_futuras(stock_base_t1, price_avg_t1)
q_t2, o_t2, precios_t2, ingresos_t2 = simular_4_semanas_futuras(stock_base_t2, price_avg_t2)

# --- Calcular KPIs ---
def calcular_kpis_dynamic(quiebres_df, ordenes_df, precios_df, ingresos_df, stock_base, tienda):
    productos = quiebres_df.columns
    semanas = len(quiebres_df)

    utilidad_total = demanda_total = demanda_satisfecha = 0
    total_quiebres = total_ordenes = costo_total_inv = costo_total_q = costo_total_o = 0
    dias_prom_inventario = []

    for p in productos:
        costo = costo_unitario[p]
        inv_cost = 0.1 * costo
        orden_fijo = costo_fijo_orden[p]

        quiebre = quiebres_df[p]
        ordenes = ordenes_df[p]
        ingresos = ingresos_df[p]
        precios = precios_df[p]
        demanda = quiebre + (ingresos / precios)

        venta = ingresos / precios
        ingreso_total = ingresos.sum()
        quiebre_cost = 0.1 * precios.mean()
        costo_q = quiebre.sum() * quiebre_cost
        costo_inv = stock_base[p] * inv_cost * semanas
        ordenes_realizadas = (ordenes > 0).sum()
        costo_o = ordenes_realizadas * orden_fijo
        utilidad = ingreso_total - costo_q - costo_inv - costo_o

        utilidad_total += utilidad
        demanda_total += demanda.sum()
        demanda_satisfecha += venta.sum()
        total_quiebres += quiebre.sum()
        total_ordenes += ordenes_realizadas
        costo_total_inv += costo_inv
        costo_total_q += costo_q
        costo_total_o += costo_o
        dias_prom_inventario.append(stock_base[p] / max(1, demanda.mean()))

    nivel_servicio = (demanda_satisfecha / demanda_total) * 100 if demanda_total > 0 else 0
    dias_prom = np.mean(dias_prom_inventario)

    return {
        "Tienda": tienda,
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

# --- Calcular y guardar KPIs ---
kpi_t1 = calcular_kpis_dynamic(q_t1, o_t1, precios_t1, ingresos_t1, stock_base_t1, "Tienda 1")
kpi_t2 = calcular_kpis_dynamic(q_t2, o_t2, precios_t2, ingresos_t2, stock_base_t2, "Tienda 2")
df_kpis = pd.DataFrame([kpi_t1, kpi_t2])

# --- Guardar Excel completo ---
with pd.ExcelWriter("Caso_Base/KPIs_Forecast_4_Semanas.xlsx", engine="xlsxwriter") as writer:
    df_kpis.to_excel(writer, sheet_name="Resumen KPIs", index=False)

    pd.DataFrame({
        "Producto": product_names,
        "Stock Base T1": [stock_base_t1[p] for p in product_names],
        "Precio Prom. T1": [price_avg_t1[p] for p in product_names],
        "Stock Base T2": [stock_base_t2[p] for p in product_names],
        "Precio Prom. T2": [price_avg_t2[p] for p in product_names],
    }).to_excel(writer, sheet_name="Stock y Precios", index=False)

    q_t1.to_excel(writer, sheet_name="Quiebres T1", index=False)
    o_t1.to_excel(writer, sheet_name="Órdenes T1", index=False)
    precios_t1.to_excel(writer, sheet_name="Precios T1", index=False)
    ingresos_t1.to_excel(writer, sheet_name="Ingresos T1", index=False)

    q_t2.to_excel(writer, sheet_name="Quiebres T2", index=False)
    o_t2.to_excel(writer, sheet_name="Órdenes T2", index=False)
    precios_t2.to_excel(writer, sheet_name="Precios T2", index=False)
    ingresos_t2.to_excel(writer, sheet_name="Ingresos T2", index=False)

# --- Consola ---
print("\n✅ Archivo generado: KPIs_Forecast_4_Semanas.xlsx")
print("\nResumen de KPI’s de simulación futura (4 semanas):\n")
print(df_kpis.round(2))
