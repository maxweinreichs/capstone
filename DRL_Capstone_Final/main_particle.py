import numpy as np
import os
import time
import pandas as pd
from utils import guardar_precios_optimos_csv
from optimizador import calcular_resultados_optimizacion, load_static_params_once, set_global_eval_seed
from particle_filter import particle_filter_optimization_multi_resample

# NUEVO: importar el módulo de análisis
from analizar_utilidad import calcular_utilidad_total, guardar_demanda_real_simulada

def optimize_prices_for_week(optimizer_caller, ruta_datos, n_productos, n_tiendas, static_params, current_inventory, semana_idx):
    """
    Optimize prices for a week using particle filter approach with multi-stage resampling.
    """
    print(f"  Optimizando precios usando filtro de partículas con resampling múltiple...")
    
    N_PARTICLES = 50  # Reduced from 1000 to 50 for more efficient exploration with liberal policy
    
    # Use base prices from static parameters
    precios_base = static_params["precios_base_np"]
    
    # Run particle filter optimization with multi-stage resampling
    best_prices, best_score = particle_filter_optimization_multi_resample(
        n_particles=N_PARTICLES,
        n_productos=n_productos,
        n_tiendas=n_tiendas,
        precios_base=precios_base,
        evaluate_fn=optimizer_caller,
        semana_idx=semana_idx
    )
    
    # Calculate detailed results for the best particle
    resultados_detallados = calcular_resultados_optimizacion(
        best_prices, current_inventory,
        semana_idx, ruta_datos, n_productos, n_tiendas,
        use_eval_seed=True
    )
    
    # Print detailed utility information
    print("\n=== Resultados Detallados de la Mejor Partícula ===")
    print(f"  Claves disponibles en resultados_detallados: {list(resultados_detallados.keys())}")
    print(f"  Utilidad Total del Horizonte: {resultados_detallados['utilidad_total_horizonte']:.2f}")
    print("============================================\n")
    
    return best_prices

def main():
    RUTA_DATOS = "parametros"
    N_PRODUCTOS = 10
    N_TIENDAS = 2
    N_SEMANAS_PLANIFICACION = 4

    static_opt_params = load_static_params_once(RUTA_DATOS, N_PRODUCTOS, N_TIENDAS)
    current_inventory_for_opt = static_opt_params["I_initial_global"] 

    all_weeks_data = []

    for semana_idx_año in range(1, N_SEMANAS_PLANIFICACION + 1):
        timestamp_main = time.strftime("%H:%M:%S", time.localtime())
        print(f"\n[{timestamp_main} Main] === Optimizando Precios para Semana del Año: {semana_idx_año} ===")

        def optimizer_caller_for_particle(precios_np_semana_actual):
            resultados = calcular_resultados_optimizacion(
                precios_np_semana_actual, current_inventory_for_opt,
                semana_idx_año, RUTA_DATOS, N_PRODUCTOS, N_TIENDAS,
                use_eval_seed=False 
            )
            return resultados["utilidad_total_horizonte"]

        precios_optimos_semana_actual_np = optimize_prices_for_week(
            optimizer_caller_for_particle, 
            RUTA_DATOS, N_PRODUCTOS, N_TIENDAS,
            static_opt_params,
            current_inventory_for_opt,
            semana_idx_año
        )
        
        print(f"  Precios óptimos PF para Semana {semana_idx_año} (P0T0): {precios_optimos_semana_actual_np[0,0]:.2f}")

        set_global_eval_seed(12345 + semana_idx_año)
        print(f"  Calculando resultados detallados para Semana {semana_idx_año} con precios PF óptimos...")
        resultados_detallados_semana = calcular_resultados_optimizacion(
            precios_optimos_semana_actual_np, current_inventory_for_opt,
            semana_idx_año, RUTA_DATOS, N_PRODUCTOS, N_TIENDAS,
            use_eval_seed=True
        )

        # NUEVO: guardar demanda real simulada + estimada
        guardar_demanda_real_simulada(
            semana=semana_idx_año,
            mu_dict=resultados_detallados_semana["mu_calculado_horizonte"],
            sigma_dict=resultados_detallados_semana["sigma_calculado_horizonte"],
            n_muestras=3,
            guardar_mu=True
        )

        for q in range(N_PRODUCTOS):
            for l in range(N_TIENDAS):
                all_weeks_data.append({
                    "semana_año": semana_idx_año, "producto_idx": q, "tienda_idx": l,
                    "precio_optimo": precios_optimos_semana_actual_np[q, l],
                    "pedido_optimo_sem1_horizonte": resultados_detallados_semana["pedidos_semana1"].get((q,l), 0),
                    "demanda_promedio_sem1_horizonte": resultados_detallados_semana["demanda_promedio_semana1"].get((q,l), 0),
                    "shortage_promedio_sem1_horizonte": resultados_detallados_semana["shortage_promedio_semana1"].get((q,l), 0),
                    "inventario_inicial_sem1_horizonte": current_inventory_for_opt.get((q,l),0),
                    "inventario_final_sem1_horizonte": resultados_detallados_semana["inventario_final_semana1"].get((q,l), 0)
                })
        
        current_inventory_for_opt = resultados_detallados_semana["inventario_final_semana1"]
        print(f"  Inventario final para P0T0 de Semana {semana_idx_año} (será inicial de S{semana_idx_año+1}): {current_inventory_for_opt.get((0,0),0):.2f}")

    df_resultados_completos = pd.DataFrame(all_weeks_data)
    ruta_salida_csv = os.path.join("resultados", "Planificacion_Semanal_Optima_PF.csv")
    os.makedirs("resultados", exist_ok=True)
    df_resultados_completos.to_csv(ruta_salida_csv, index=False, sep=';', decimal='.')
    print(f"\n¡Planificación secuencial completada para {N_SEMANAS_PLANIFICACION} semanas!")
    print(f"Resultados detallados guardados en: {ruta_salida_csv}")

    # NUEVO: Llamar a analizar_utilidad
    print("\n=== Análisis Final de Utilidad ===")
    utilidad_total, utilidad_por_semana = calcular_utilidad_total(ruta_salida_csv)
    print(f"Utilidad total obtenida: ${utilidad_total:,.0f}")
    print("Utilidad por semana:")
    print(utilidad_por_semana)

if __name__ == "__main__":
    if not os.path.exists("parametros"):
        print("ADVERTENCIA: La carpeta 'parametros' no existe.")
    main()