import re
import unicodedata
import pandas as pd
import numpy as np
import gurobipy as gp
from pathlib import Path
import os

# ---------- Configuration ----------
BASE_PARAMS_FILE = Path("datos_base_mejorado - Sheet1.csv")
PARAMS_DIR = Path("parametros")

T_WEEKS = 4
N_PRODUCTS = 10
N_STORES = 2
N_SCENARIOS = 10 # Keep it manageable for testing, increase for "production" runs

Q_list = list(range(1, N_PRODUCTS + 1))
L_list = list(range(1, N_STORES + 1))
S_list = list(range(N_SCENARIOS))

q_idx = {q: q - 1 for q in Q_list}
l_idx = {l: l - 1 for l in L_list}
s_idx = {s: s for s in S_list}
t_idx = {t: t - 1 for t in range(1, T_WEEKS + 1)}

# ---------- Helper Functions ----------
def clean_text(x: str) -> str:
    txt = unicodedata.normalize("NFKD", str(x))
    txt = "".join(c for c in txt if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", txt.lower()).strip()

def find_row_in_df(df, patterns):
    pat = re.compile("|".join(patterns), re.IGNORECASE)
    for i, cell_val in df.iloc[:, 0].items():
        if pat.search(clean_text(cell_val)):
            return i
    raise ValueError(f"Patterns {patterns} not found in DataFrame's first column.")

def pad_array(vec, name, target_len=N_PRODUCTS):
    current_len = len(vec)
    if current_len < target_len:
        miss = target_len - current_len
        valid_vec = vec[~np.isnan(vec)]
        mean_val = np.mean(valid_vec) if len(valid_vec) > 0 else 0.0
        
        if not isinstance(mean_val, (int, float)):
            print(f"Warning: Mean value for padding '{name}' is not a scalar ({mean_val}). Defaulting to 0.0.")
            mean_val = 0.0

        vec_padded = np.concatenate([vec, np.full(miss, mean_val)])
        # print(f"⚠ '{name}' has {current_len} entries; padded {miss} with mean/zero ({mean_val:.2f}).") # Less verbose
        return vec_padded
    elif current_len > target_len:
        # print(f"⚠ '{name}' has {current_len} entries; truncated to {target_len}.") # Less verbose
        return vec[:target_len]
    return vec

def parse_numeric(s):
    s_str = str(s).strip()
    num_dots = s_str.count('.')
    num_commas = s_str.count(',')

    if num_dots == 1 and num_commas == 0:
        pass
    elif num_commas == 1 and num_dots == 0:
        s_str = s_str.replace(',', '.')
    elif num_commas >= 1 and num_dots == 1:
        if s_str.find(',') < s_str.find('.'): # 1,234.56
             s_str = s_str.replace(',', '')
        else: # 1.234,56 -- this case is less common if dot is thousands
             s_str = s_str.replace('.', '').replace(',', '.')
    elif num_dots >= 1 and num_commas == 1:
        if s_str.find('.') < s_str.find(','): # 1.234,56
            s_str = s_str.replace('.', '').replace(',', '.')
        else: # 1,234.56 -- this case is less common if comma is thousands
            s_str = s_str.replace(',', '') # Should be caught by previous case
    elif num_dots > 1 :
        s_str = s_str.replace('.', '')
    elif num_commas > 1:
        s_str = s_str.replace(',', '')
    try:
        return float(s_str)
    except ValueError:
        return np.nan

# ---------- 1. Load Base Parameters ----------
print("\n──────── Loading Base Parameters ────────")
base_df = pd.read_csv(BASE_PARAMS_FILE, header=None, encoding="utf-8")
# ... (find_row_in_df calls as before) ...
row_cost = find_row_in_df(base_df, ["costo.*prod", "costo.*may"])
row_fixed = find_row_in_df(base_df, ["costo.*fijo"])
row_min_order = find_row_in_df(base_df, ["min.*orden", "cant.*min", "mín"])
row_cap1 = find_row_in_df(base_df, ["cap.*tienda.*1"])
row_cap2 = find_row_in_df(base_df, ["cap.*tienda.*2"])
row_transport = find_row_in_df(base_df, ["costo.*transp"])

raw_c_series = base_df.iloc[row_cost, 1:N_PRODUCTS+1].apply(parse_numeric)
raw_c = raw_c_series.values
c_q_arr = pad_array(raw_c, "c_q")

raw_K_series = base_df.iloc[row_fixed, 1:N_PRODUCTS+1].apply(parse_numeric)
raw_K = raw_K_series.values
K_q_arr = pad_array(raw_K, "K_q")

raw_M_series = base_df.iloc[row_min_order, 1:N_PRODUCTS+1].apply(parse_numeric)
raw_M = raw_M_series.values
M_q_arr = pad_array(raw_M, "M_q")

c_q_arr_no_nan = np.nan_to_num(c_q_arr)
h_q_arr = 0.10 * c_q_arr_no_nan
w_q_arr = 0.50 * c_q_arr_no_nan

cap_l_input = [base_df.iloc[row_cap1, 1], base_df.iloc[row_cap2, 1]]
cap_l_arr = [parse_numeric(x) for x in cap_l_input]
cap_l_arr = [x if pd.notna(x) and x > 0 else 1_000_000.0 for x in cap_l_arr]

TRANSP_COST_input = base_df.iloc[row_transport, 1]
TRANSP_COST = parse_numeric(TRANSP_COST_input)
if pd.isna(TRANSP_COST): TRANSP_COST = 0.0


print(f"c_q_arr: {c_q_arr}")
print(f"K_q_arr: {K_q_arr}")
print(f"M_q_arr: {M_q_arr}")
# ... (other prints)
print(f"cap_l_arr: {cap_l_arr}")
print(f"TRANSP_COST: {TRANSP_COST}")

# ---------- 2. Load Demand Parameters ... ----------
print("\n──────── Loading Demand Parameters & Week Groups ────────")
demand_params_store_product_group = {1: {}, 2: {}}
week_group_map_store_week = {1: {}, 2: {}}

for store_id in L_list:
    param_file = PARAMS_DIR / f"parametros_capstone - param_tienda{store_id}.csv"
    if not param_file.exists():
        # Try checking current working directory as a fallback
        param_file_cwd = Path(f"parametros_capstone - param_tienda{store_id}.csv")
        if not param_file_cwd.exists():
            raise FileNotFoundError(f"Parameter file {param_file} (or {param_file_cwd}) not found.")
        else:
            param_file = param_file_cwd # Use file from CWD
            
    df_params = pd.read_csv(param_file)
    
    required_cols = {'producto', 'grupo', 'gamma', 'rho', 'alfa', 'theta (simbolo raro)'}
    if not required_cols.issubset(df_params.columns):
        raise ValueError(f"File {param_file} missing one of required columns: {required_cols}")

    for _, row in df_params.iterrows():
        prod_str = str(row['producto']).replace('producto_', '')
        try:
            prod_id = int(prod_str)
            if prod_id not in Q_list:
                 continue
        except ValueError:
            # print(f"Warning: Could not parse product_id from '{row['producto']}' in {param_file}. Skipping.")
            continue

        group = str(row['grupo']).lower()
        if prod_id not in demand_params_store_product_group[store_id]:
            demand_params_store_product_group[store_id][prod_id] = {}
        
        try:
            gamma_val = float(row['gamma'])
            rho_val = float(row['rho'])
            alpha_param_val = float(row['alfa'])
            theta_val = float(row['theta (simbolo raro)'])
        except ValueError as e:
            # print(f"Error parsing demand params for P{prod_id}, S{store_id}, G{group}: {e}. Skipping row.")
            continue
            
        if theta_val < 1e-6:
            theta_val = 1e-6
        if rho_val <= 0:
            rho_val = 1e-6

        demand_params_store_product_group[store_id][prod_id][group] = {
            'gamma': gamma_val,
            'rho': rho_val,
            'alpha_param': alpha_param_val,
            'theta': theta_val
        }

    group_file = PARAMS_DIR / f"parametros_capstone - grupos_tienda{store_id}.csv"
    if not group_file.exists():
        group_file_cwd = Path(f"parametros_capstone - grupos_tienda{store_id}.csv")
        if not group_file_cwd.exists():
            raise FileNotFoundError(f"Group mapping file {group_file} (or {group_file_cwd}) not found.")
        else:
            group_file = group_file_cwd

    df_groups = pd.read_csv(group_file)
    if not {'nro_semana', 'grupo'}.issubset(df_groups.columns):
         raise ValueError(f"File {group_file} missing 'nro_semana' or 'grupo' column.")

    current_store_week_groups = {}
    for _, row in df_groups.iterrows():
        try:
            week_num = int(row['nro_semana'])
            group_name = str(row['grupo']).lower()
            if week_num not in current_store_week_groups:
                current_store_week_groups[week_num] = group_name
        except ValueError:
            # print(f"Warning: Skipping row in {group_file} due to parsing error: {row}")
            continue
            
    last_known_group_for_store = 'medio'
    for t_planning_week in range(1, T_WEEKS + 1):
        if t_planning_week in current_store_week_groups:
            week_group_map_store_week[store_id][t_planning_week] = current_store_week_groups[t_planning_week]
            last_known_group_for_store = current_store_week_groups[t_planning_week]
        else:
            week_group_map_store_week[store_id][t_planning_week] = last_known_group_for_store


def get_demand_params_for_week(product_id, store_id, week_t):
    group = week_group_map_store_week[store_id].get(week_t)
    if not group:
        group = 'medio' 
    
    product_params = demand_params_store_product_group.get(store_id, {}).get(product_id, {})
    group_specific_params = product_params.get(group)

    if not group_specific_params:
        return {'gamma': 1.0, 'rho': 100.0, 'alpha_param': 0.01, 'theta': 1000.0}
    return group_specific_params

# ---------- 3. Pre-generate Random Shocks ----------
np.random.seed(42)
random_shocks_splt = np.random.normal(0, 1, size=(N_SCENARIOS, N_PRODUCTS, N_STORES, T_WEEKS))
print(f"\n✅ Generated {N_SCENARIOS} sets of random_shocks_splt for SAA.")

# ---------- 4. Gurobi Model ----------
print("\n──────── Building Gurobi Model ────────")
m = gp.Model("pricing_inv_saa_final")
m.setParam("NonConvex", 2)
m.setParam("TimeLimit", 300)
m.setParam("MIPGap", 0.05) # A 5% gap is often acceptable for hard problems

p_qlt = m.addVars(Q_list, L_list, range(1, T_WEEKS + 1), lb=0.0, name="price")
o_qlt = m.addVars(Q_list, L_list, range(1, T_WEEKS + 1), lb=0.0, name="order_qty")
y_qlt_is_ordering = m.addVars(Q_list, L_list, range(1, T_WEEKS + 1), vtype=gp.GRB.BINARY, name="is_ordering")

mu_demand_qlt = m.addVars(Q_list, L_list, range(1, T_WEEKS + 1), lb=0.0, name="mean_demand")
std_dev_demand_qlt = m.addVars(Q_list, L_list, range(1, T_WEEKS + 1), lb=1e-6, name="std_dev_demand")

I_sqlt = m.addVars(S_list, Q_list, L_list, range(T_WEEKS + 1), lb=0.0, name="inventory")
D_sold_sqlt = m.addVars(S_list, Q_list, L_list, range(1, T_WEEKS + 1), lb=0.0, name="demand_sold")

for s_val in S_list:
    for q_val in Q_list:
        for l_val in L_list:
            initial_inv_val = cap_l_arr[l_idx[l_val]] / (2 * N_PRODUCTS) if N_PRODUCTS > 0 and cap_l_arr[l_idx[l_val]] > 0 else 0.0
            m.addConstr(I_sqlt[s_val, q_val, l_val, 0] == initial_inv_val)

for q_val in Q_list:
    for l_val in L_list:
        for t_val in range(1, T_WEEKS + 1):
            params = get_demand_params_for_week(q_val, l_val, t_val)
            
            arg_exp_qlt = m.addVar(lb=-70, ub=70, name=f"arg_exp_q{q_val}_l{l_val}_t{t_val}") # exp(70) is large, exp(-70) very small
            m.addConstr(arg_exp_qlt == -params['alpha_param'] * p_qlt[q_val,l_val,t_val])

            exp_val_qlt = m.addVar(lb=1e-30, ub=gp.GRB.INFINITY, name=f"exp_val_q{q_val}_l{l_val}_t{t_val}")
            m.addGenConstrExp(arg_exp_qlt, exp_val_qlt)
            
            m.addConstr(mu_demand_qlt[q_val,l_val,t_val] == params['gamma'] * params['rho'] * exp_val_qlt)
            
            mu_squared_qlt = m.addVar(lb=0, name=f"mu_sq_q{q_val}_l{l_val}_t{t_val}")
            m.addGenConstrPow(mu_demand_qlt[q_val,l_val,t_val], mu_squared_qlt, 2.0)
            
            variance_val_qlt = m.addVar(lb=1e-6, name=f"var_q{q_val}_l{l_val}_t{t_val}")
            m.addConstr(variance_val_qlt == mu_demand_qlt[q_val,l_val,t_val] + mu_squared_qlt / params['theta'])
            
            m.addGenConstrPow(variance_val_qlt, std_dev_demand_qlt[q_val,l_val,t_val], 0.5)

for s_val in S_list:
    for q_val in Q_list:
        for l_val in L_list:
            for t_val in range(1, T_WEEKS + 1):
                shock = random_shocks_splt[s_idx[s_val], q_idx[q_val], l_idx[l_val], t_idx[t_val]]
                
                potential_demand_sqlt = m.addVar(lb=-gp.GRB.INFINITY, name=f"pot_D_s{s_val}_q{q_val}_l{l_val}_t{t_val}")
                m.addConstr(potential_demand_sqlt == mu_demand_qlt[q_val,l_val,t_val] + shock * std_dev_demand_qlt[q_val,l_val,t_val])
                
                actual_demand_realization_sqlt = m.addVar(lb=0.0, name=f"actual_D_s{s_val}_q{q_val}_l{l_val}_t{t_val}")
                m.addGenConstrMax(actual_demand_realization_sqlt, [potential_demand_sqlt], constant=0.0)

                available_sqlt_var = m.addVar(lb=0.0, name=f"avail_var_s{s_val}_q{q_val}_l{l_val}_t{t_val}")
                m.addConstr(available_sqlt_var == I_sqlt[s_val, q_val, l_val, t_val-1] + o_qlt[q_val, l_val, t_val])
                
                m.addGenConstrMin(D_sold_sqlt[s_val, q_val, l_val, t_val], [available_sqlt_var, actual_demand_realization_sqlt])
                
                m.addConstr(I_sqlt[s_val, q_val, l_val, t_val] == available_sqlt_var - D_sold_sqlt[s_val, q_val, l_val, t_val])

for s_val in S_list:
    for l_val in L_list:
        for t_val in range(1, T_WEEKS + 1):
            m.addConstr(gp.quicksum(I_sqlt[s_val, q_val, l_val, t_val] for q_val in Q_list) <= cap_l_arr[l_idx[l_val]])

for q_val in Q_list:
    for l_val in L_list:
        for t_val in range(1, T_WEEKS + 1):
            m.addConstr(o_qlt[q_val, l_val, t_val] >= M_q_arr[q_idx[q_val]] * y_qlt_is_ordering[q_val, l_val, t_val])
            max_order_val = cap_l_arr[l_idx[l_val]] if cap_l_arr[l_idx[l_val]] > 0 else 1e7
            m.addConstr(o_qlt[q_val, l_val, t_val] <= max_order_val * y_qlt_is_ordering[q_val, l_val, t_val])

for q_val in Q_list:
    for t_val in range(1, T_WEEKS + 1):
        min_price_q = 1.05 * c_q_arr_no_nan[q_idx[q_val]] if c_q_arr_no_nan[q_idx[q_val]] > 0 else 0.01
        for l_val in L_list:
            m.addConstr(p_qlt[q_val, l_val, t_val] >= min_price_q)
        
        if N_STORES == 2 and len(L_list) >=2 :
            m.addConstr(p_qlt[q_val, L_list[0], t_val] - p_qlt[q_val, L_list[1], t_val] <= TRANSP_COST)
            m.addConstr(p_qlt[q_val, L_list[1], t_val] - p_qlt[q_val, L_list[0], t_val] <= TRANSP_COST)

total_expected_scenario_profit = gp.LinExpr()
first_stage_ordering_costs = gp.LinExpr()

for s_val in S_list:
    scenario_revenue = gp.quicksum(p_qlt[q,l,t] * D_sold_sqlt[s_val,q,l,t] for q in Q_list for l in L_list for t in range(1, T_WEEKS + 1))
    scenario_holding_cost = gp.quicksum(h_q_arr[q_idx[q]] * I_sqlt[s_val,q,l,t] for q in Q_list for l in L_list for t in range(1, T_WEEKS + 1))
    scenario_salvage_value = gp.quicksum(w_q_arr[q_idx[q]] * I_sqlt[s_val,q,l,T_WEEKS] for q in Q_list for l in L_list)
    total_expected_scenario_profit += (1.0/N_SCENARIOS) * (scenario_revenue - scenario_holding_cost + scenario_salvage_value)

for q_val in Q_list:
    for l_val in L_list:
        for t_val in range(1, T_WEEKS + 1):
            first_stage_ordering_costs += c_q_arr_no_nan[q_idx[q_val]] * o_qlt[q_val,l_val,t_val] + \
                                          K_q_arr[q_idx[q_val]] * y_qlt_is_ordering[q_val,l_val,t_val]

m.setObjective(total_expected_scenario_profit - first_stage_ordering_costs, gp.GRB.MAXIMIZE)

print("\n──────── Optimizing Model ────────")
m.optimize()

output_filename_suffix = f"s{N_SCENARIOS}_t{T_WEEKS}"

if m.Status == gp.GRB.OPTIMAL or m.Status == gp.GRB.SUBOPTIMAL or m.Status == gp.GRB.TIME_LIMIT:
    if m.Status == gp.GRB.TIME_LIMIT:
        print(f"⚠ Time limit reached. Using best found solution.")
    elif m.Status != gp.GRB.OPTIMAL: # Covers SUBOPTIMAL
        print(f"⚠ Optimal solution not found. Status: {m.Status}. Using best found solution.")
    
    if m.SolCount > 0: # Check if any solution is available
        print(f"\n💰 Expected Utility (SAA): {m.ObjVal:,.2f}\n")
        
        resultados_politica = []
        for t_val in range(1, T_WEEKS + 1):
            for q_val in Q_list:
                for l_val in L_list:
                    try:
                        price_val = p_qlt[q_val, l_val, t_val].X
                        order_val = o_qlt[q_val, l_val, t_val].X
                        y_order_val = y_qlt_is_ordering[q_val, l_val, t_val].X
                        mean_demand_val = mu_demand_qlt[q_val, l_val, t_val].X
                        
                        # Calculate average D_sold for this (q,l,t) across scenarios
                        avg_d_sold = 0
                        if D_sold_sqlt[S_list[0], q_val, l_val, t_val].X is not None: # Check if solution exists
                             avg_d_sold = sum(D_sold_sqlt[s, q_val, l_val, t_val].X for s in S_list) / N_SCENARIOS
                        
                    except AttributeError: # If variables were not solved
                        price_val, order_val, y_order_val, mean_demand_val, avg_d_sold = 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
                    
                    resultados_politica.append({
                        "producto": q_val,
                        "tienda": l_val,
                        "semana": t_val,
                        "precio_optimo": f"{price_val:.2f}" if isinstance(price_val, float) else price_val,
                        "orden_optima": f"{order_val:.2f}" if isinstance(order_val, float) else order_val,
                        "y_ordena": int(y_order_val) if isinstance(y_order_val, float) else y_order_val,
                        "demanda_media_esperada_mu": f"{mean_demand_val:.2f}" if isinstance(mean_demand_val, float) else mean_demand_val,
                        "promedio_unidades_vendidas": f"{avg_d_sold:.2f}" if isinstance(avg_d_sold, float) else avg_d_sold
                    })
        df_politica = pd.DataFrame(resultados_politica)
        policy_csv_path = f"politica_optima_saa_{output_filename_suffix}.csv"
        df_politica.to_csv(policy_csv_path, index=False)
        print(f"✅ Política óptima (precios y órdenes) guardada en {policy_csv_path}")
        
        if not df_politica.empty and not all(df_politica["precio_optimo"] == 'N/A'):
            print("\n🔎 Snippet: Decisiones y demandas resultantes (promedio sobre escenarios para ventas)")
            print(df_politica[['producto', 'tienda', 'semana', 'precio_optimo', 'orden_optima', 'demanda_media_esperada_mu', 'promedio_unidades_vendidas']].head(15).to_string(index=False))
        else:
            print("ℹ No policy data to display (variables might not have values).")
    else:
        print("⚠ No feasible solution found within the time limit or due to other issues.")

else: # Handle other statuses like INFEASIBLE, UNBOUNDED explicitly
    print(f"⚠ No solution found. Status Gurobi: {m.Status}")
    if m.Status == gp.GRB.INFEASIBLE: # Corrected attribute
        print("Computing IIS to find cause of infeasibility...")
        try:
            m.computeIIS()
            iis_file = f"model_iis_{output_filename_suffix}.ilp"
            m.write(iis_file)
            print(f"IIS written to {iis_file}. Please examine this file to debug.")
        except gp.GurobiError as e:
            print(f"Could not compute IIS: {e}")
    elif m.Status == gp.GRB.UNBOUNDED:
        print("Model is unbounded.")
    elif m.Status == 4: # GRB.INFEASIBLE_OR_UNBOUNDED
        print("Model is infeasible or unbounded. Computing IIS if possible...")
        try:
            m.computeIIS() # This will only work if it's truly infeasible
            iis_file = f"model_iis_status4_{output_filename_suffix}.ilp"
            m.write(iis_file)
            print(f"IIS (if model was infeasible) written to {iis_file}.")
        except gp.GurobiError:
             print("Could not compute IIS (model might be unbounded or other issue).")


print("\n──────── Fin del Script ────────")