import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import time
import math
import os
from utils import read_csv_with_comma_decimal, cargar_precios_iniciales_csv 

CACHED_STATIC_PARAMS = None
GLOBAL_EVAL_SEED = None 

def set_global_eval_seed(seed): 
    global GLOBAL_EVAL_SEED
    GLOBAL_EVAL_SEED = seed

def load_static_params_once(ruta_datos, Q_val=10, L_val=2):
    global CACHED_STATIC_PARAMS
    if CACHED_STATIC_PARAMS is not None:
        return CACHED_STATIC_PARAMS

    df_params_gen = read_csv_with_comma_decimal(os.path.join(ruta_datos, "General_Parameters.csv"))
    params_gen_dict = {row['parametro']: row['valor'] for _, row in df_params_gen.iterrows()}

    Q = int(params_gen_dict.get('Q', Q_val))
    L = int(params_gen_dict.get('L', L_val))
    N_scenarios = int(params_gen_dict.get('N', 50))
    T_periods_horizon = 4 # Horizonte de optimización del subproblema (fijo en 4 semanas)

    if Q != Q_val or L != L_val:
        raise ValueError(f"Se esperan Q={Q_val} y L={L_val}.")

    c = {q: float(params_gen_dict.get(f'c_{q}', 0)) for q in range(Q)}
    K = {q: float(params_gen_dict.get(f'K_{q}', 0)) for q in range(Q)}
    IF_cap = {l: float(params_gen_dict.get(f'IF_{l}', 1000)) for l in range(L)} 
    
    # El I_initial de General_Parameters.csv es para la *primera semana global del año*.
    # Las semanas subsiguientes usarán el inventario final calculado.
    I_initial_global = {(q, l): float(params_gen_dict.get(f'I_inicial_{q}_{l}', 0)) for q in range(Q) for l in range(L)}
    
    MinOrder = {q: float(params_gen_dict.get(f'MinOrder_{q}', 0)) for q in range(Q)}
    
    dl = {}
    for q_idx in range(Q):
        for l_idx in range(L):
            dl_value = params_gen_dict.get(f'dl_{q_idx}_{l_idx}', 0)
            dl[(q_idx, l_idx)] = float(dl_value)

    precios_base_np = cargar_precios_iniciales_csv(os.path.join(ruta_datos, "Precios_Iniciales.csv"), Q, L)

    df_asoc = read_csv_with_comma_decimal(os.path.join(ruta_datos, "asociaciones_semana_dist.csv"))
    df_asoc['q_idx'] = df_asoc['producto'].apply(lambda x: int(x.split('_')[1]) - 1)
    df_asoc['l_idx'] = df_asoc['Tienda'] - 1
    grupo_qlt_map = {}
    max_semana_asoc = df_asoc['nro_semana'].max()

    for q_idx in range(Q):
        for l_idx in range(L):
            for t_planning_horizon in range(1, T_periods_horizon + 1): # Para el horizonte de 4 semanas
                # Mapear semana_planificacion_actual (1 a N_SEMANAS_PLANIFICACION) a la semana del año
                # Esto es crucial si la optimización del DRL se hace para semana_año > 1
                # Por ahora, asumimos que el optimizador siempre ve las semanas 1,2,3,4 del *año*
                # para los parámetros de demanda si no se pasa la semana_año_actual.
                # Esta parte necesitará más refinamiento si `asociaciones_semana_dist.csv`
                # debe usarse relativo a la semana_año_actual que se está optimizando.
                # Por simplicidad, para este subproblema, vamos a asumir que t_planning_horizon (1 a 4)
                # se refiere a las semanas del año 1 a 4 para cargar los parámetros de demanda.
                # O, mejor, que los parámetros de demanda para el horizonte de 4 semanas
                # se cargan relativos a la `semana_año_actual` que el DRL está optimizando.
                # Para este cambio, vamos a asumir que el DRL optimiza semana_año N,
                # y el subproblema de optimización mira N, N+1, N+2, N+3.

                # La clave es cómo 't_idx' en el optimizador (1 a T_periods_horizon)
                # se mapea a la 'nro_semana' en `asociaciones_semana_dist.csv`.
                # Por ahora, para mantenerlo simple, vamos a necesitar que `prepare_dynamic_params`
                # reciba `semana_año_actual_optimizando`
                pass # La lógica de mapeo de semana se hará en prepare_dynamic_params


    dist_param_files = {
        0: os.path.join(ruta_datos, "par_dist_t1.csv"), 
        1: os.path.join(ruta_datos, "par_dist_t2.csv")  
    }
    dist_params_processed = {0: {}, 1: {}}
    # ... (carga de dist_params_processed sin cambios) ...
    for l_idx, filepath in dist_param_files.items():
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archivo de parámetros de distribución no encontrado: {filepath}")
        df_dist = read_csv_with_comma_decimal(filepath)
        df_dist['q_idx'] = df_dist['producto'].apply(lambda x: int(x.split('_')[1]) - 1)
        for _, row in df_dist.iterrows():
            q_idx = int(row['q_idx'])
            grupo = row['grupo']
            if q_idx not in dist_params_processed[l_idx]:
                dist_params_processed[l_idx][q_idx] = {}
            dist_params_processed[l_idx][q_idx][grupo] = {
                'alfa': float(row['alfa']), 'gamma': float(row['gamma']),
                'rho': float(row['rho']), 'theta': float(row['theta'])
            }
    
    CACHED_STATIC_PARAMS = {
        "Q": Q, "L": L, "N_scenarios": N_scenarios, "T_periods_horizon": T_periods_horizon,
        "c": c, "K": K, "IF_cap": IF_cap, "I_initial_global": I_initial_global, 
        "MinOrder": MinOrder, "dl": dl,
        "precios_base_np": precios_base_np, 
        "df_asoc": df_asoc, # Guardar el DataFrame completo para el mapeo dinámico
        "max_semana_asoc": max_semana_asoc,
        "dist_params_processed": dist_params_processed
    }
    return CACHED_STATIC_PARAMS


def prepare_dynamic_params(precios_semana_actual_np, semana_año_actual_optimizando, static_params):
    """
    precios_semana_actual_np: Precios para la semana_año_actual_optimizando (semana t=1 del horizonte)
    semana_año_actual_optimizando: La semana del año para la que se están fijando precios (1-indexed).
    """
    Q = static_params["Q"]
    L = static_params["L"]
    T_horizon = static_params["T_periods_horizon"]
    precios_base_np = static_params["precios_base_np"] # Precios para semanas > 1 del horizonte
    df_asoc = static_params["df_asoc"]
    max_semana_asoc = static_params["max_semana_asoc"]
    dist_params_processed = static_params["dist_params_processed"]

    p_qlt_horizon = {} # Precios para el horizonte de T_horizon semanas
    mu_calculated_horizon = {}
    sigma_calculated_horizon = {}

    for q_idx in range(Q):
        for l_idx in range(L):
            # Semana t=1 del HORIZONTE usa los precios_semana_actual_np (del DRL)
            p_qlt_horizon[(q_idx, l_idx, 1)] = precios_semana_actual_np[q_idx, l_idx]
            
            # Semanas t=2, 3, 4 del HORIZONTE usan precios_base_np
            for t_h in range(2, T_horizon + 1): # t_h es 2, 3, 4
                p_qlt_horizon[(q_idx, l_idx, t_h)] = precios_base_np[q_idx, l_idx]
    
    # Calcular mu y sigma para cada semana del horizonte de planificación
    for t_h in range(1, T_horizon + 1): # t_h = 1, 2, 3, 4 (índice dentro del horizonte)
        semana_del_año_para_parametros = semana_año_actual_optimizando + t_h - 1
        # Asegurar que no exceda el máximo de semanas en asociaciones_semana_dist.csv
        semana_del_año_para_parametros = min(semana_del_año_para_parametros, max_semana_asoc)

        for q_idx in range(Q):
            for l_idx in range(L):
                precio_actual_horizonte = p_qlt_horizon[(q_idx, l_idx, t_h)]
                
                # Encontrar grupo para (q_idx, l_idx, semana_del_año_para_parametros)
                # La columna 'nro_semana' en df_asoc es 1-indexed
                filtro_asoc = (df_asoc['q_idx'] == q_idx) & \
                              (df_asoc['l_idx'] == l_idx) & \
                              (df_asoc['nro_semana'] == semana_del_año_para_parametros)
                
                asoc_match = df_asoc[filtro_asoc]

                if asoc_match.empty:
                    raise ValueError(f"No se encontró asociación de grupo para q={q_idx}, l={l_idx}, semana_año={semana_del_año_para_parametros}. Verifique df_asoc.")
                
                grupo_actual_str = asoc_match['grupo'].iloc[0]

                if q_idx not in dist_params_processed[l_idx] or \
                   grupo_actual_str not in dist_params_processed[l_idx][q_idx]:
                    raise ValueError(f"No se encontraron parámetros de distribución para q={q_idx}, l={l_idx}, t_horizon={t_h}, grupo='{grupo_actual_str}'.")

                params_grupo = dist_params_processed[l_idx][q_idx][grupo_actual_str]
                alpha_val, gamma_val, rho_val, theta_val = params_grupo['alfa'], params_grupo['gamma'], params_grupo['rho'], params_grupo['theta']
                
                current_mu = gamma_val * rho_val * math.exp(-alpha_val * precio_actual_horizonte)
                # Clave para el diccionario es (q, l, t_h) donde t_h es el paso en el horizonte
                mu_calculated_horizon[(q_idx, l_idx, t_h)] = current_mu 
                
                term_in_paren = 1.0 + (current_mu / theta_val) if abs(theta_val) > 1e-9 else 1.0
                sigma_sq = current_mu * max(0.0, term_in_paren)
                sigma_calculated_horizon[(q_idx, l_idx, t_h)] = math.sqrt(max(0.0, sigma_sq))
                
    return p_qlt_horizon, mu_calculated_horizon, sigma_calculated_horizon


def generate_scenarios(mu, sigma, N_scenarios, Q, L, T_periods_horizon, use_eval_seed=False):
    global GLOBAL_EVAL_SEED
    if use_eval_seed and GLOBAL_EVAL_SEED is not None:
        np.random.seed(GLOBAL_EVAL_SEED)
    else:
        np.random.seed(int(time.time()) % (2**32 -1)) 
        
    scenarios = {}
    for i in range(N_scenarios):
        scenarios[i] = {}
        for q_idx in range(Q):
            for l_idx in range(L):
                for t_h in range(1, T_periods_horizon + 1): # Iterar sobre el horizonte
                    key = (q_idx, l_idx, t_h) # Clave es (q,l, paso_horizonte)
                    demand_sample = np.random.normal(mu[key], sigma[key])
                    scenarios[i][key] = max(0, int(demand_sample)) 
    return scenarios

# MODIFICADO: Acepta inventario_inicial_dict y devuelve más datos
def solve_optimization_problem(p_qlt_horizon, mu_calculated_horizon, sigma_calculated_horizon, 
                               inventario_inicial_actual_dict, # {(q,l): valor} para la semana t=1 del horizonte
                               static_params, use_eval_seed=False):
    Q = static_params["Q"]
    L = static_params["L"]
    N = static_params["N_scenarios"]
    T_horizon = static_params["T_periods_horizon"] # Horizonte del subproblema (e.g., 4 semanas)
    IF_cap = static_params["IF_cap"]
    c_cost = static_params["c"] # Renombrado para evitar conflicto con var local
    K_cost = static_params["K"] # Renombrado
    dl_cost = static_params["dl"] # Renombrado
    MinOrder_val = static_params["MinOrder"] # Renombrado

    scenarios = generate_scenarios(mu_calculated_horizon, sigma_calculated_horizon, N, Q, L, T_horizon, use_eval_seed=use_eval_seed)
    
    model = gp.Model("Retail_Inventory_Opt_Sequential")
    model.setParam('OutputFlag', 0) 
    
    o, y_bin, I_inv, Y_sales, U_shortage = {}, {}, {}, {}, {} 

    for q_idx in range(Q):
        for l_idx in range(L):
            for t_h in range(1, T_horizon + 1):
                o[q_idx,l_idx,t_h] = model.addVar(vtype=GRB.CONTINUOUS, name=f"o_{q_idx}_{l_idx}_{t_h}", lb=0)
                y_bin[q_idx,l_idx,t_h] = model.addVar(vtype=GRB.BINARY, name=f"y_bin_{q_idx}_{l_idx}_{t_h}")
                I_inv[q_idx,l_idx,t_h] = model.addVar(vtype=GRB.CONTINUOUS, name=f"I_inv_{q_idx}_{l_idx}_{t_h}", lb=0)
                for i_scen in range(N):
                    Y_sales[q_idx,l_idx,t_h,i_scen] = model.addVar(vtype=GRB.CONTINUOUS, name=f"Y_sales_{q_idx}_{l_idx}_{t_h}_{i_scen}", lb=0)
                    U_shortage[q_idx,l_idx,t_h,i_scen] = model.addVar(vtype=GRB.CONTINUOUS, name=f"U_shortage_{q_idx}_{l_idx}_{t_h}_{i_scen}", lb=0)

    M_large = 1e6
    for q_idx in range(Q):
        for l_idx in range(L):
            for t_h in range(1, T_horizon + 1):
                # Inventario inicial para t_h=1 es el pasado como argumento
                # Para t_h > 1, es el I_inv del final del periodo anterior del horizonte
                inventario_inicio_periodo_val = inventario_inicial_actual_dict[q_idx,l_idx] if t_h == 1 else I_inv[q_idx,l_idx,t_h-1]
                
                inventario_disp_antes_demanda = inventario_inicio_periodo_val + o[q_idx,l_idx,t_h]

                model.addConstr(o[q_idx,l_idx,t_h] >= MinOrder_val[q_idx] * y_bin[q_idx,l_idx,t_h])
                model.addConstr(o[q_idx,l_idx,t_h] <= M_large * y_bin[q_idx,l_idx,t_h])
                model.addConstr(inventario_disp_antes_demanda <= IF_cap[l_idx], name=f"Capacidad_Prod_{q_idx}_{l_idx}_{t_h}")

                sum_Y_para_promedio = gp.LinExpr()
                for i_scen in range(N):
                    demanda_escenario = scenarios[i_scen][(q_idx,l_idx,t_h)] # Demanda para este paso del horizonte
                    model.addConstr(Y_sales[q_idx,l_idx,t_h,i_scen] <= demanda_escenario)
                    model.addConstr(Y_sales[q_idx,l_idx,t_h,i_scen] <= inventario_disp_antes_demanda)
                    model.addConstr(U_shortage[q_idx,l_idx,t_h,i_scen] >= demanda_escenario - inventario_disp_antes_demanda)
                    sum_Y_para_promedio += Y_sales[q_idx,l_idx,t_h,i_scen]
                
                model.addConstr(I_inv[q_idx,l_idx,t_h] == inventario_disp_antes_demanda - (1/N if N > 0 else 1) * sum_Y_para_promedio)
    
    expected_revenue = gp.quicksum(Y_sales[q,l,t,i] * p_qlt_horizon[q,l,t] for q in range(Q) for l in range(L) for t in range(1,T_horizon+1) for i in range(N)) / (N if N > 0 else 1)
    expected_ordering_cost = gp.quicksum(c_cost[q] * o[q,l,t] for q in range(Q) for l in range(L) for t in range(1,T_horizon+1))
    expected_fixed_cost = gp.quicksum(K_cost[q] * y_bin[q,l,t] for q in range(Q) for l in range(L) for t in range(1,T_horizon+1))
    expected_shortage_cost = gp.quicksum(dl_cost[q,l] * U_shortage[q,l,t,i] for q in range(Q) for l in range(L) for t in range(1,T_horizon+1) for i in range(N)) / (N if N > 0 else 1)
    
    model.setObjective(expected_revenue - expected_ordering_cost - expected_fixed_cost - expected_shortage_cost, GRB.MAXIMIZE)
    model.optimize()
    
    # Inicializar diccionarios para resultados
    resultados = {
        "utilidad_total_horizonte": -float('inf'),
        "pedidos_semana1": {}, # (q,l) -> valor
        "demanda_promedio_semana1": {}, # (q,l) -> valor
        "shortage_promedio_semana1": {}, # (q,l) -> valor
        "inventario_final_semana1": {} # (q,l) -> valor
    }

    if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
        resultados["utilidad_total_horizonte"] = model.ObjVal
        # Extraer datos para la *primera semana del horizonte* (t_h=1)
        for q_idx in range(Q):
            for l_idx in range(L):
                resultados["pedidos_semana1"][(q_idx,l_idx)] = o[q_idx,l_idx,1].X
                
                avg_sales_s1 = sum(Y_sales[q_idx,l_idx,1,i_scen].X for i_scen in range(N)) / (N if N > 0 else 1)
                resultados["demanda_promedio_semana1"][(q_idx,l_idx)] = avg_sales_s1
                
                avg_shortage_s1 = sum(U_shortage[q_idx,l_idx,1,i_scen].X for i_scen in range(N)) / (N if N > 0 else 1)
                resultados["shortage_promedio_semana1"][(q_idx,l_idx)] = avg_shortage_s1
                
                resultados["inventario_final_semana1"][(q_idx,l_idx)] = I_inv[q_idx,l_idx,1].X
    else:
        status_msg = f"Modelo infactible ({'eval' if use_eval_seed else 'train'})" if model.status == GRB.INFEASIBLE else f"Estado Gurobi {model.status} ({'eval' if use_eval_seed else 'train'})"
        print(f"Error en optimizador: {status_msg}")
        # Rellenar con valores nulos o ceros para que el DRL pueda continuar con una mala recompensa
        for q_idx in range(Q):
            for l_idx in range(L):
                resultados["pedidos_semana1"][(q_idx,l_idx)] = 0.0
                resultados["demanda_promedio_semana1"][(q_idx,l_idx)] = 0.0
                resultados["shortage_promedio_semana1"][(q_idx,l_idx)] = 0.0 # O un valor alto si se quiere penalizar más
                # Inventario final podría ser el inicial si no hay operación
                resultados["inventario_final_semana1"][(q_idx,l_idx)] = inventario_inicial_actual_dict.get((q_idx,l_idx), 0.0)


    model.dispose()
    return resultados


# MODIFICADO: Acepta inv_inicial_semana_actual y semana_año_actual
def calcular_resultados_optimizacion(precios_semana_actual_np, 
                                    inventario_inicial_semana_actual_dict, 
                                    semana_año_actual_optimizando, # 1-indexed
                                    ruta_datos, 
                                    Q_val=10, L_val=2, use_eval_seed=False):
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    # print(f"[{timestamp} Optimizador.calcular_resultados_optimizacion({'EVAL' if use_eval_seed else 'TRAIN'})] Semana Año: {semana_año_actual_optimizando}, P0T0: {precios_semana_actual_np[0,0]:.2f}")
    
    opt_start_time = time.time()
    resultados_opt = None
    try:
        static_params = load_static_params_once(ruta_datos, Q_val, L_val)
        p_qlt_horizon, mu_calculated_horizon, sigma_calculated_horizon = prepare_dynamic_params(
            precios_semana_actual_np, semana_año_actual_optimizando, static_params
        )
        resultados_opt = solve_optimization_problem(
            p_qlt_horizon, mu_calculated_horizon, sigma_calculated_horizon, 
            inventario_inicial_semana_actual_dict, static_params, use_eval_seed=use_eval_seed
        )
    except FileNotFoundError as e:
        print(f"[{timestamp} Optimizador] Error crítico al cargar archivos: {e}")
        # Devolver estructura de resultados con valores por defecto/error
    except ValueError as e:
        print(f"[{timestamp} Optimizador] Error en datos o parámetros: {e}")
    except gp.GurobiError as e:
        print(f"[{timestamp} Optimizador] Error de Gurobi: {e.code} - {e.message}")
    except Exception as e:
        print(f"[{timestamp} Optimizador] Excepción inesperada: {e}")
        import traceback
        traceback.print_exc()

    # Si resultados_opt no se pudo calcular, inicializarlo con valores de error/default
    if resultados_opt is None:
        resultados_opt = {
            "utilidad_total_horizonte": -float('inf'), "pedidos_semana1": {},
            "demanda_promedio_semana1": {}, "shortage_promedio_semana1": {},
            "inventario_final_semana1": {}
        }
        for q_idx in range(Q_val):
            for l_idx in range(L_val):
                resultados_opt["pedidos_semana1"][(q_idx,l_idx)] = 0.0
                resultados_opt["demanda_promedio_semana1"][(q_idx,l_idx)] = 0.0
                resultados_opt["shortage_promedio_semana1"][(q_idx,l_idx)] = 0.0
                resultados_opt["inventario_final_semana1"][(q_idx,l_idx)] = inventario_inicial_semana_actual_dict.get((q_idx,l_idx), 0.0)

    opt_duration = time.time() - opt_start_time
    if opt_duration > 1.5 or resultados_opt["utilidad_total_horizonte"] < -1e8 : 
        # Obtener los precios base para comparar
        precios_base = static_params["precios_base_np"]
        
        # Crear un resumen de cambios de precios
        cambios_precios = []
        for q_idx in range(Q_val):
            for l_idx in range(L_val):
                precio_actual = precios_semana_actual_np[q_idx, l_idx]
                precio_base = precios_base[q_idx, l_idx]
                variacion = precio_actual - precio_base
                variacion_pct = (variacion / precio_base) * 100 if precio_base > 0 else 0
                
                # Solo incluir si hay cambio significativo
                if abs(variacion_pct) > 0.1:  # Umbral de 0.1% para considerar un cambio
                    cambios_precios.append(f"P{q_idx}T{l_idx}: {precio_actual:.2f} (Δ{variacion_pct:+.2f}%)")
        
        # Mostrar todos los cambios sin limitación
        cambios_str = ", ".join(cambios_precios)
            
        print(f"[{timestamp} Optimizador.calcular_resultados_optimizacion({'EVAL' if use_eval_seed else 'TRAIN'})] "
              f"FIN. Utilidad: {resultados_opt['utilidad_total_horizonte']:.2f}. "
              f"Duración: {opt_duration:.2f}s. Cambios: {cambios_str}")
    return resultados_opt


if __name__ == "__main__":
    # ... (El main de prueba de optimizador.py necesitará actualizarse para pasar el inventario inicial
    #      y la semana_año_actual_optimizando, y para manejar la nueva estructura de 'resultados_opt')
    print("Testeando optimizador.py directamente...")
    RutaDatos = "parametros" 
    N_PRODUCTOS_TEST = 10
    N_TIENDAS_TEST = 2

    # Cargar inventario inicial global para la primera prueba
    static_params_test = load_static_params_once(RutaDatos, N_PRODUCTOS_TEST, N_TIENDAS_TEST)
    inv_ini_test_sem1 = static_params_test["I_initial_global"]

    precios_test_sem1 = static_params_test["precios_base_np"] # Usar precios base para el test

    print(f"\nTest para Semana del Año 1 con Inventario Global Inicial:")
    resultados_sem1 = calcular_resultados_optimizacion(
        precios_test_sem1, inv_ini_test_sem1, 1, RutaDatos,
        N_PRODUCTOS_TEST, N_TIENDAS_TEST, use_eval_seed=False
    )
    print(f"  Utilidad Horizonte: {resultados_sem1['utilidad_total_horizonte']:.2f}")
    print(f"  Pedido P0T0 Sem1: {resultados_sem1['pedidos_semana1'].get((0,0),0):.2f}")
    print(f"  Inv. Fin P0T0 Sem1: {resultados_sem1['inventario_final_semana1'].get((0,0),0):.2f}")

    if resultados_sem1['utilidad_total_horizonte'] > -float('inf'):
        inv_ini_test_sem2 = resultados_sem1['inventario_final_semana1']
        precios_test_sem2 = precios_test_sem1 * 1.05 # Modificar precios ligeramente para la semana 2

        print(f"\nTest para Semana del Año 2 con Inventario Final de Semana 1:")
        resultados_sem2 = calcular_resultados_optimizacion(
            precios_test_sem2, inv_ini_test_sem2, 2, RutaDatos,
            N_PRODUCTOS_TEST, N_TIENDAS_TEST, use_eval_seed=False
        )
        print(f"  Utilidad Horizonte: {resultados_sem2['utilidad_total_horizonte']:.2f}")
        print(f"  Pedido P0T0 Sem1 del Opt: {resultados_sem2['pedidos_semana1'].get((0,0),0):.2f}")
        print(f"  Inv. Fin P0T0 Sem1 del Opt: {resultados_sem2['inventario_final_semana1'].get((0,0),0):.2f}")