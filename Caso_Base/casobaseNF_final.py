import pandas as pd
import numpy as np
import os # For path handling

# --- Cargar datos ---
file_path = "Datos_v1.xlsx"
# Create directory if it doesn't exist
output_dir = "Caso_Base_Resultados"
os.makedirs(output_dir, exist_ok=True)

xls = pd.ExcelFile(file_path)
df_t1 = pd.read_excel(xls, sheet_name="Datos Tienda 1", skiprows=5)
df_t2 = pd.read_excel(xls, sheet_name="Datos tienda 2", skiprows=5)

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
        if demand_data.empty: # Handle cases with no demand data for a product
            stock_base[product_names[i]] = 0
        else:
            stock_base[product_names[i]] = np.percentile(demand_data, 90)
        
        price_data = pd.to_numeric(df[price_col], errors='coerce').dropna()
        if price_data.empty:
            price_avg[product_names[i]] = 0 # Or some default price
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
    
    # inventario[p] will store the inventory at the START of each week for product p
    # The first element is the initial stock_base
    inventario_hist = {p: [stock_base_dict[p]] for p in productos} 
    
    quiebres, ordenes, precios, ingresos = {}, {}, {}, {}
    inventario_inicial_semana = {p: [] for p in productos} # Stores inv_ant for each week
    inventario_final_antes_repo_semana = {p: [] for p in productos} # Stores stock_post for each week

    for p in productos:
        quiebres[p], ordenes[p], precios[p], ingresos[p] = [], [], [], []

    for t in range(semanas):
        for p in productos:
            demanda = demanda_df.loc[t, p] if pd.notna(demanda_df.loc[t, p]) else 0
            inv_ant = inventario_hist[p][-1] # Inventory at the start of week t

            inventario_inicial_semana[p].append(inv_ant) # Log initial inventory for the week

            # Precio dinámico según nivel de inventario
            if inv_ant > 1.2 * stock_base_dict[p]:
                precio_actual = precios_prom[p] * 0.8
            elif inv_ant < 0.8 * stock_base_dict[p] and stock_base_dict[p] > 0 : # avoid division by zero if stock_base is 0
                precio_actual = precios_prom[p] * 1.2
            else:
                precio_actual = precios_prom[p]

            vendido = min(inv_ant, demanda)
            quiebre = max(demanda - inv_ant, 0)
            stock_post = inv_ant - vendido # Inventory at end of week t, BEFORE replenishment

            inventario_final_antes_repo_semana[p].append(stock_post) # Log final inventory before repo

            # Reposición escalonada
            if stock_base_dict[p] > 0: # Only order if there's a base stock defined
                if stock_post < 0.6 * stock_base_dict[p]:
                    nueva_orden = stock_base_dict[p] # Order up to base_stock
                elif stock_post < 0.9 * stock_base_dict[p]:
                    nueva_orden = 0.5 * (stock_base_dict[p] - stock_post)
                else:
                    nueva_orden = 0.2 * (stock_base_dict[p] - stock_post)
                nueva_orden = max(0, nueva_orden) # Ensure order is not negative
            else:
                nueva_orden = 0


            inventario_hist[p].append(stock_post + nueva_orden) # Inventory for start of NEXT week
            quiebres[p].append(quiebre)
            ordenes[p].append(nueva_orden)
            precios[p].append(precio_actual)
            ingresos[p].append(vendido * precio_actual)
            
    # inventario_hist[p][:-1] would be the initial inventories for week 0 to N-1
    # but we've already captured that in inventario_inicial_semana more directly.
    
    return (
        pd.DataFrame(quiebres),
        pd.DataFrame(ordenes),
        pd.DataFrame(precios),
        pd.DataFrame(ingresos),
        pd.DataFrame(inventario_inicial_semana),         # New: Inv at start of week t
        pd.DataFrame(inventario_final_antes_repo_semana) # New: Inv at end of week t (before repo)
    )

# --- Simular ---
q_t1, o_t1, precios_t1, ingresos_t1, inv_ini_t1, inv_fin_t1 = simular_base_stock_con_precio(demanda_t1_4sem, stock_base_t1, price_avg_t1)
q_t2, o_t2, precios_t2, ingresos_t2, inv_ini_t2, inv_fin_t2 = simular_base_stock_con_precio(demanda_t2_4sem, stock_base_t2, price_avg_t2)


# --- KPIs ---
def calcular_kpis_dynamic(demanda_df, quiebres_df, ordenes_df, precios_df, ingresos_df, stock_base, tienda):
    productos = demanda_df.columns
    semanas = len(demanda_df)

    utilidad_total = demanda_total = demanda_satisfecha = 0
    total_quiebres = total_ordenes = costo_total_inv = costo_total_q = costo_total_o = 0
    dias_prom_inventario = []

    for p in productos:
        costo = costo_unitario[p]
        inv_cost = 0.1 * costo # Costo de mantener inventario por unidad por semana
        orden_fijo = costo_fijo_orden[p]

        demanda = demanda_df[p].fillna(0)
        quiebre = quiebres_df[p]
        ordenes_col = ordenes_df[p] # Renamed to avoid conflict
        ingresos_col = ingresos_df[p] # Renamed
        precios_col = precios_df[p]   # Renamed

        venta = demanda - quiebre
        ingreso_total_prod = ingresos_col.sum()
        
        # Costo de quiebre: 10% del precio promedio de VENTA de ese producto esa semana
        # If all prices are 0 (e.g. no sales, no price data), quiebre_cost_unit should be 0
        # Ensure precios_col.mean() doesn't fail on all NaNs or produce NaN
        valid_prices = precios_col.dropna()
        avg_price_for_quiebre = valid_prices.mean() if not valid_prices.empty else 0
        quiebre_cost_unit = 0.1 * avg_price_for_quiebre if avg_price_for_quiebre > 0 else 0
        
        costo_q_prod = quiebre.sum() * quiebre_cost_unit
        
        # Costo de inventario: basado en el stock_base objetivo, no el real fluctuante
        # Esto es una simplificación. Un cálculo más preciso usaría el inventario promedio semanal.
        # Para el KPI, se asume que se *intenta* mantener stock_base[p]
        costo_inv_prod = stock_base[p] * inv_cost * semanas 
        
        ordenes_realizadas_prod = (ordenes_col > 1e-9).sum() # Consider orders effectively > 0
        costo_o_prod = ordenes_realizadas_prod * orden_fijo
        
        # Costo de los productos vendidos (COGS)
        costo_productos_vendidos = venta.sum() * costo_unitario[p]

        utilidad_prod = ingreso_total_prod - costo_productos_vendidos - costo_q_prod - costo_inv_prod - costo_o_prod

        utilidad_total += utilidad_prod
        demanda_total += demanda.sum()
        demanda_satisfecha += venta.sum()
        total_quiebres += quiebre.sum()
        total_ordenes += ordenes_realizadas_prod
        costo_total_inv += costo_inv_prod
        costo_total_q += costo_q_prod
        costo_total_o += costo_o_prod
        
        # Días promedio de inventario: stock_base / demanda_diaria_promedio
        # Demanda diaria promedio = demanda_semanal_promedio / 7
        demanda_semanal_promedio_prod = demanda.mean()
        if demanda_semanal_promedio_prod > 0:
            dias_inv_prod = stock_base[p] / (demanda_semanal_promedio_prod / 7) # Asumiendo 7 días/semana
        else:
            dias_inv_prod = np.inf if stock_base[p] > 0 else 0 # Infinito si hay stock y no demanda, 0 si no hay stock
        dias_prom_inventario.append(dias_inv_prod)

    nivel_servicio = (demanda_satisfecha / demanda_total) * 100 if demanda_total > 0 else 0
    
    # Manejar inf y nan en dias_prom_inventario
    valid_dias_inv = [d for d in dias_prom_inventario if np.isfinite(d)]
    dias_prom = np.mean(valid_dias_inv) if valid_dias_inv else 0


    return {
        "Tienda": tienda,
        "Utilidad Total (4 sem)": utilidad_total,
        "Demanda Total": demanda_total,
        "Demanda Satisfecha": demanda_satisfecha,
        "Demanda Insatisfecha": total_quiebres,
        "Nivel de Servicio (%)": nivel_servicio,
        "Días Prom. Inventario": dias_prom, # Días de cobertura
        "Órdenes Emitidas": total_ordenes,
        "Costo Inventario (basado en Stock Base)": costo_total_inv,
        "Costo Ordenamiento": costo_total_o,
        "Costo Demanda Insatisfecha": costo_total_q
    }

# --- Ejecutar KPIs ---
kpi_t1 = calcular_kpis_dynamic(demanda_t1_4sem, q_t1, o_t1, precios_t1, ingresos_t1, stock_base_t1, "Tienda 1")
kpi_t2 = calcular_kpis_dynamic(demanda_t2_4sem, q_t2, o_t2, precios_t2, ingresos_t2, stock_base_t2, "Tienda 2")
df_kpis = pd.DataFrame([kpi_t1, kpi_t2])

# --- Crear hoja con inventarios iniciales ---
inventarios_iniciales_df = pd.DataFrame({
    "Producto": product_names,
    "Inventario Inicial Tienda 1 (Stock Base)": [stock_base_t1[p] for p in product_names],
    "Inventario Inicial Tienda 2 (Stock Base)": [stock_base_t2[p] for p in product_names]
})

# --- Exportar a Excel ---
excel_kpi_path = os.path.join(output_dir, "KPIs_2025_Sim_4_Semanas.xlsx")
with pd.ExcelWriter(excel_kpi_path, engine="xlsxwriter") as writer:
    df_kpis.to_excel(writer, sheet_name="Resumen KPIs", index=False)
    demanda_t1_4sem.to_excel(writer, sheet_name="Demanda T1", index=False)
    demanda_t2_4sem.to_excel(writer, sheet_name="Demanda T2", index=False)
    q_t1.to_excel(writer, sheet_name="Quiebres T1", index=False)
    o_t1.to_excel(writer, sheet_name="Órdenes T1", index=False)
    precios_t1.to_excel(writer, sheet_name="Precios T1", index=False)
    ingresos_t1.to_excel(writer, sheet_name="Ingresos T1", index=False)
    inv_ini_t1.to_excel(writer, sheet_name="Inv Ini T1", index=False)
    inv_fin_t1.to_excel(writer, sheet_name="Inv Fin T1", index=False)
    q_t2.to_excel(writer, sheet_name="Quiebres T2", index=False)
    o_t2.to_excel(writer, sheet_name="Órdenes T2", index=False)
    precios_t2.to_excel(writer, sheet_name="Precios T2", index=False)
    ingresos_t2.to_excel(writer, sheet_name="Ingresos T2", index=False)
    inv_ini_t2.to_excel(writer, sheet_name="Inv Ini T2", index=False)
    inv_fin_t2.to_excel(writer, sheet_name="Inv Fin T2", index=False)
    inventarios_iniciales_df.to_excel(writer, sheet_name="Inventarios Base", index=False)

print(f"\n✅ Simulación completada. Archivo de KPIs generado: {excel_kpi_path}")
print(df_kpis.round(2))

# --- Preparar datos para el formato CSV solicitado ---
datos_planificacion_semanal = []
num_semanas = len(demanda_t1_4sem) # Asumimos ambas tiendas tienen mismas semanas de test

# Tienda 1 (tienda_idx = 0)
for semana_idx in range(num_semanas): # semana_idx es 0-indexed
    for prod_idx, prod_name in enumerate(product_names):
        demanda_real_sem = demanda_t1_4sem.loc[semana_idx, prod_name] if pd.notna(demanda_t1_4sem.loc[semana_idx, prod_name]) else 0
        
        datos_planificacion_semanal.append({
            "semana_año": semana_idx + 1,
            "producto_idx": prod_idx,
            "tienda_idx": 0,
            "precio_optimo": precios_t1.loc[semana_idx, prod_name],
            "pedido_optimo_sem1_horizonte": o_t1.loc[semana_idx, prod_name],
            "demanda_promedio_sem1_horizonte": demanda_real_sem, # Usamos la demanda real de la semana
            "shortage_promedio_sem1_horizonte": q_t1.loc[semana_idx, prod_name],
            "inventario_inicial_sem1_horizonte": inv_ini_t1.loc[semana_idx, prod_name],
            "inventario_final_sem1_horizonte": inv_fin_t1.loc[semana_idx, prod_name]
        })

# Tienda 2 (tienda_idx = 1)
for semana_idx in range(num_semanas):
    for prod_idx, prod_name in enumerate(product_names):
        demanda_real_sem = demanda_t2_4sem.loc[semana_idx, prod_name] if pd.notna(demanda_t2_4sem.loc[semana_idx, prod_name]) else 0
        
        datos_planificacion_semanal.append({
            "semana_año": semana_idx + 1,
            "producto_idx": prod_idx,
            "tienda_idx": 1,
            "precio_optimo": precios_t2.loc[semana_idx, prod_name],
            "pedido_optimo_sem1_horizonte": o_t2.loc[semana_idx, prod_name],
            "demanda_promedio_sem1_horizonte": demanda_real_sem, # Usamos la demanda real de la semana
            "shortage_promedio_sem1_horizonte": q_t2.loc[semana_idx, prod_name],
            "inventario_inicial_sem1_horizonte": inv_ini_t2.loc[semana_idx, prod_name],
            "inventario_final_sem1_horizonte": inv_fin_t2.loc[semana_idx, prod_name]
        })

df_planificacion = pd.DataFrame(datos_planificacion_semanal)

# Definir el orden de las columnas para el CSV
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

print(f"\n✅ Archivo de planificación semanal generado: {csv_output_path}")
print("\nPrimeras filas del archivo de planificación:")
print(df_planificacion.head())
print("\nÚltimas filas del archivo de planificación:")
print(df_planificacion.tail())