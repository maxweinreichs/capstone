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
COSTO_TRANSPORTE_ENTRE_TIENDAS = 3.8e6  # 3.8 millones como constante

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
    T_periods_horizon = 4

    if Q != Q_val or L != L_val:
        raise ValueError(f"Se esperan Q={Q_val} y L={L_val}.")

    c = {q: float(params_gen_dict.get(f'c_{q}', 0)) for q in range(Q)}
    K = {q: float(params_gen_dict.get(f'K_{q}', 0)) for q in range(Q)}
    IF_cap = {l: float(params_gen_dict.get(f'IF_{l}', 1000)) for l in range(L)} 
    
    I_initial_global = {(q, l): float(params_gen_dict.get(f'I_inicial_{q}_{l}', 0)) for q in range(Q) for l in range(L)}
    
    MinOrder = {q: float(params_gen_dict.get(f'MinOrder_{q}', 0)) for q in range(Q)}
    
    precios_base_np = cargar_precios_iniciales_csv(os.path.join(ruta_datos, "Precios_Iniciales.csv"), Q, L)

    df_asoc = read_csv_with_comma_decimal(os.path.join(ruta_datos, "asociaciones_semana_dist.csv"))
    df_asoc['q_idx'] = df_asoc['producto'].apply(lambda x: int(x.split('_')[1]) - 1)
    df_asoc['l_idx'] = df_asoc['Tienda'] - 1
    grupo_qlt_map = {}
    max_semana_asoc = df_asoc['nro_semana'].max()

    dist_param_files = {
        0: os.path.join(ruta_datos, "par_dist_t1.csv"), 
        1: os.path.join(ruta_datos, "par_dist_t2.csv")  
    }
    dist_params_processed = {0: {}, 1: {}}
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
        "MinOrder": MinOrder,
        "precios_base_np": precios_base_np, 
        "df_asoc": df_asoc,
        "max_semana_asoc": max_semana_asoc,
        "dist_params_processed": dist_params_processed
    }
    return CACHED_STATIC_PARAMS


def prepare_dynamic_params(precios_semana_actual_np, semana_año_actual_optimizando, static_params):
    Q = static_params["Q"]
    L = static_params["L"]
    T_horizon = static_params["T_periods_horizon"]
    precios_base_np = static_params["precios_base_np"]
    df_asoc = static_params["df_asoc"]
    max_semana_asoc = static_params["max_semana_asoc"]
    dist_params_processed = static_params["dist_params_processed"]

    p_qlt_horizon = {}
    mu_calculated_horizon = {}
    sigma_calculated_horizon = {}

    # Agregar restricción de no arbitrariedad
    for q_idx in range(Q):
        for t_h in [1]:  # Solo para la semana actual
            precio_max = max(precios_semana_actual_np[q_idx, l] for l in range(L))
            precio_min = min(precios_semana_actual_np[q_idx, l] for l in range(L))
            if precio_max - precio_min > COSTO_TRANSPORTE_ENTRE_TIENDAS:
                # Ajustar precios para cumplir con la restricción
                precio_promedio = np.mean(precios_semana_actual_np[q_idx, :])
                for l in range(L):
                    precios_semana_actual_np[q_idx, l] = precio_promedio

    for q_idx in range(Q):
        for l_idx in range(L):
            p_qlt_horizon[(q_idx, l_idx, 1)] = precios_semana_actual_np[q_idx, l_idx]
            for t_h in range(2, T_horizon + 1):
                p_qlt_horizon[(q_idx, l_idx, t_h)] = precios_base_np[q_idx, l_idx]
    
    for t_h in range(1, T_horizon + 1):
        semana_del_año_para_parametros = semana_año_actual_optimizando + t_h - 1
        semana_del_año_para_parametros = min(semana_del_año_para_parametros, max_semana_asoc)

        for q_idx in range(Q):
            for l_idx in range(L):
                precio_actual_horizonte = p_qlt_horizon[(q_idx, l_idx, t_h)]
                
                filtro_asoc = (df_asoc['q_idx'] == q_idx) & \
                              (df_asoc['l_idx'] == l_idx) & \
                              (df_asoc['nro_semana'] == semana_del_año_para_parametros)
                
                asoc_match = df_asoc[filtro_asoc]

                if asoc_match.empty:
                    raise ValueError(f"No se encontró asociación de grupo para q={q_idx}, l={l_idx}, semana_año={semana_del_año_para_parametros}.")
                
                grupo_actual_str = asoc_match['grupo'].iloc[0]

                if q_idx not in dist_params_processed[l_idx] or \
                   grupo_actual_str not in dist_params_processed[l_idx][q_idx]:
                    raise ValueError(f"No se encontraron parámetros de distribución para q={q_idx}, l={l_idx}, t_horizon={t_h}, grupo='{grupo_actual_str}'.")

                params_grupo = dist_params_processed[l_idx][q_idx][grupo_actual_str]
                alpha_val, gamma_val, rho_val, theta_val = params_grupo['alfa'], params_grupo['gamma'], params_grupo['rho'], params_grupo['theta']
                
                current_mu = gamma_val * rho_val * math.exp(-alpha_val * precio_actual_horizonte)
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
                for t_h in range(1, T_periods_horizon + 1):
                    key = (q_idx, l_idx, t_h)
                    demand_sample = np.random.normal(mu[key], sigma[key])
                    scenarios[i][key] = max(0, int(demand_sample)) 
    return scenarios

def solve_optimization_problem(p_qlt_horizon, mu_calculated_horizon, sigma_calculated_horizon, 
                               inventario_inicial_actual_dict, 
                               static_params, use_eval_seed=False):
    Q = static_params["Q"]
    L = static_params["L"]
    N = static_params["N_scenarios"]
    T_horizon = static_params["T_periods_horizon"]
    IF_cap = static_params["IF_cap"]
    c_cost = static_params["c"]
    K_cost = static_params["K"]
    MinOrder_val = static_params["MinOrder"]

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
                inventario_inicio_periodo_val = inventario_inicial_actual_dict[q_idx,l_idx] if t_h == 1 else I_inv[q_idx,l_idx,t_h-1]
                inventario_disp_antes_demanda = inventario_inicio_periodo_val + o[q_idx,l_idx,t_h]

                model.addConstr(o[q_idx,l_idx,t_h] >= MinOrder_val[q_idx] * y_bin[q_idx,l_idx,t_h])
                model.addConstr(o[q_idx,l_idx,t_h] <= M_large * y_bin[q_idx,l_idx,t_h])
                model.addConstr(inventario_disp_antes_demanda <= IF_cap[l_idx], name=f"Capacidad_Prod_{q_idx}_{l_idx}_{t_h}")

                sum_Y_para_promedio = gp.LinExpr()
                for i_scen in range(N):
                    demanda_escenario = scenarios[i_scen][(q_idx,l_idx,t_h)]
                    model.addConstr(Y_sales[q_idx,l_idx,t_h,i_scen] <= demanda_escenario)
                    model.addConstr(Y_sales[q_idx,l_idx,t_h,i_scen] <= inventario_disp_antes_demanda)
                    model.addConstr(U_shortage[q_idx,l_idx,t_h,i_scen] >= demanda_escenario - inventario_disp_antes_demanda)
                    sum_Y_para_promedio += Y_sales[q_idx,l_idx,t_h,i_scen]
                
                model.addConstr(I_inv[q_idx,l_idx,t_h] == inventario_disp_antes_demanda - (1/N if N > 0 else 1) * sum_Y_para_promedio)
    
    # ========== COSTOS MODIFICADOS ========== #
    expected_revenue = gp.quicksum(Y_sales[q,l,t,i] * p_qlt_horizon[q,l,t] for q in range(Q) for l in range(L) for t in range(1,T_horizon+1) for i in range(N)) / (N if N > 0 else 1)
    expected_ordering_cost = gp.quicksum(c_cost[q] * o[q,l,t] for q in range(Q) for l in range(L) for t in range(1,T_horizon+1))
    expected_fixed_cost = gp.quicksum(K_cost[q] * y_bin[q,l,t] for q in range(Q) for l in range(L) for t in range(1,T_horizon+1))
    
    # Costo de inventario (10% del costo unitario)
    expected_inventory_cost = gp.quicksum(
        0.1 * c_cost[q] * I_inv[q,l,t]
        for q in range(Q)
        for l in range(L)
        for t in range(1, T_horizon+1)
    )
    
    # Costo de demanda insatisfecha (10% del precio de venta)
    expected_shortage_cost = gp.quicksum(
        0.1 * p_qlt_horizon[q,l,t] * U_shortage[q,l,t,i]
        for q in range(Q)
        for l in range(L)
        for t in range(1, T_horizon+1)
        for i in range(N)
    ) / (N if N > 0 else 1)
    
    model.setObjective(
        expected_revenue 
        - expected_ordering_cost 
        - expected_fixed_cost 
        - expected_shortage_cost 
        - expected_inventory_cost,
        GRB.MAXIMIZE
    )
    # ========== FIN COSTOS MODIFICADOS ========== #
    
    model.optimize()
    
    resultados = {
        "utilidad_total_horizonte": -float('inf'),
        "pedidos_semana1": {},
        "demanda_promedio_semana1": {},
        "shortage_promedio_semana1": {},
        "inventario_final_semana1": {}
    }

    if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
        resultados["utilidad_total_horizonte"] = model.ObjVal
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
        for q_idx in range(Q):
            for l_idx in range(L):
                resultados["pedidos_semana1"][(q_idx,l_idx)] = 0.0
                resultados["demanda_promedio_semana1"][(q_idx,l_idx)] = 0.0
                resultados["shortage_promedio_semana1"][(q_idx,l_idx)] = 0.0
                resultados["inventario_final_semana1"][(q_idx,l_idx)] = inventario_inicial_actual_dict.get((q_idx,l_idx), 0.0)

    model.dispose()
    return resultados


def calcular_resultados_optimizacion(precios_semana_actual_np, 
                                    inventario_inicial_semana_actual_dict, 
                                    semana_año_actual_optimizando,
                                    ruta_datos, 
                                    Q_val=10, L_val=2, use_eval_seed=False):
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    
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
    except ValueError as e:
        print(f"[{timestamp} Optimizador] Error en datos o parámetros: {e}")
    except gp.GurobiError as e:
        print(f"[{timestamp} Optimizador] Error de Gurobi: {e.code} - {e.message}")
    except Exception as e:
        print(f"[{timestamp} Optimizador] Excepción inesperada: {e}")
        import traceback
        traceback.print_exc()

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
         print(f"[{timestamp} Optimizador.calcular_resultados_optimizacion({'EVAL' if use_eval_seed else 'TRAIN'})] FIN. Utilidad: {resultados_opt['utilidad_total_horizonte']:.2f}. Duración: {opt_duration:.2f}s. P0T0: {precios_semana_actual_np[0,0]:.2f}")


    resultados_opt["mu_calculado_horizonte"] = mu_calculated_horizon
    resultados_opt["sigma_calculado_horizonte"] = sigma_calculated_horizon

    return resultados_opt

if __name__ == "__main__":
    print("Testeando optimizador.py directamente...")
    RutaDatos = "parametros" 
    N_PRODUCTOS_TEST = 10
    N_TIENDAS_TEST = 2

    static_params_test = load_static_params_once(RutaDatos, N_PRODUCTOS_TEST, N_TIENDAS_TEST)
    inv_ini_test_sem1 = static_params_test["I_initial_global"]
    precios_test_sem1 = static_params_test["precios_base_np"]

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
        precios_test_sem2 = precios_test_sem1 * 1.05

        print(f"\nTest para Semana del Año 2 con Inventario Final de Semana 1:")
        resultados_sem2 = calcular_resultados_optimizacion(
            precios_test_sem2, inv_ini_test_sem2, 2, RutaDatos,
            N_PRODUCTOS_TEST, N_TIENDAS_TEST, use_eval_seed=False
        )
        print(f"  Utilidad Horizonte: {resultados_sem2['utilidad_total_horizonte']:.2f}")
        print(f"  Pedido P0T0 Sem1 del Opt: {resultados_sem2['pedidos_semana1'].get((0,0),0):.2f}")
        print(f"  Inv. Fin P0T0 Sem1 del Opt: {resultados_sem2['inventario_final_semana1'].get((0,0),0):.2f}")