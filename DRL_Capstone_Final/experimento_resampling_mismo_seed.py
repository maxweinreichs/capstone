"""
Experimento de Comparación de Políticas de Resampling - MISMO SEED
Ejecuta todas las políticas con EXACTAMENTE el mismo seed para comparación completamente justa.
"""
import numpy as np
import pandas as pd
import os
import time
import shutil
from datetime import datetime

# Imports del sistema existente
from utils import guardar_precios_optimos_csv
from optimizador import calcular_resultados_optimizacion, load_static_params_once, set_global_eval_seed
from resampling_policies import get_policy_function
from particle_filter import set_global_particle_seed, reset_seed_counter
from analizar_utilidad import calcular_utilidad_total, guardar_demanda_real_simulada, set_analizar_utilidad_seed

def ejecutar_optimizacion_con_politica(politica_nombre, n_particles, seed_global, 
                                      ruta_datos, n_productos, n_tiendas, n_semanas):
    """
    Ejecuta la optimización completa usando una política específica de resampling.
    Retorna métricas de desempeño y la utilidad real calculada.
    """
    print(f"\n{'='*80}")
    print(f"EJECUTANDO: {politica_nombre.upper()} con {n_particles} partículas (seed: {seed_global})")
    print(f"{'='*80}")
    
    # Establecer seeds para reproducibilidad
    set_global_particle_seed(seed_global)
    reset_seed_counter()
    
    # Cargar parámetros estáticos
    static_opt_params = load_static_params_once(ruta_datos, n_productos, n_tiendas)
    current_inventory_for_opt = static_opt_params["I_initial_global"]
    
    # Obtener función de política
    policy_function = get_policy_function(politica_nombre)
    
    # Métricas de desempeño
    tiempo_inicio_total = time.time()
    metricas_por_semana = []
    all_weeks_data = []
    utilidades_optimizador = []
    
    for semana_idx_año in range(1, n_semanas + 1):
        print(f"\n[{time.strftime('%H:%M:%S')}] === Optimizando Semana {semana_idx_año} ===")
        
        tiempo_inicio_semana = time.time()
        
        # Función de evaluación para el optimizador
        def optimizer_caller_for_particle(precios_np_semana_actual):
            resultados = calcular_resultados_optimizacion(
                precios_np_semana_actual, current_inventory_for_opt,
                semana_idx_año, ruta_datos, n_productos, n_tiendas,
                use_eval_seed=False 
            )
            return resultados["utilidad_total_horizonte"]
        
        # Ejecutar optimización con la política específica
        precios_optimos_semana_actual_np, mejor_utilidad_optimizador = policy_function(
            n_particles=n_particles,
            n_productos=n_productos,
            n_tiendas=n_tiendas,
            precios_base=static_opt_params["precios_base_np"],
            evaluate_fn=optimizer_caller_for_particle,
            semana_idx=semana_idx_año,
            costo_transporte=3.8
        )
        
        tiempo_optimizacion = time.time() - tiempo_inicio_semana
        utilidades_optimizador.append(mejor_utilidad_optimizador)
        
        print(f"  ⏱️  Tiempo optimización semana {semana_idx_año}: {tiempo_optimizacion:.2f}s")
        print(f"  💰 Mejor utilidad optimizador: {mejor_utilidad_optimizador:.2f}")
        print(f"  💲 Precio P0T0: {precios_optimos_semana_actual_np[0,0]:.2f}")
        
        # Calcular resultados detallados para esta semana
        set_global_eval_seed(12345 + semana_idx_año)
        resultados_detallados_semana = calcular_resultados_optimizacion(
            precios_optimos_semana_actual_np, current_inventory_for_opt,
            semana_idx_año, ruta_datos, n_productos, n_tiendas,
            use_eval_seed=True
        )
        
        # Guardar demanda real simulada
        guardar_demanda_real_simulada(
            semana=semana_idx_año,
            mu_dict=resultados_detallados_semana["mu_calculado_horizonte"],
            sigma_dict=resultados_detallados_semana["sigma_calculado_horizonte"],
            n_muestras=3,
            guardar_mu=True
        )
        
        # Guardar datos de la semana para CSV temporal
        for q in range(n_productos):
            for l in range(n_tiendas):
                all_weeks_data.append({
                    "semana_año": semana_idx_año, "producto_idx": q, "tienda_idx": l,
                    "precio_optimo": precios_optimos_semana_actual_np[q, l],
                    "pedido_optimo_sem1_horizonte": resultados_detallados_semana["pedidos_semana1"].get((q,l), 0),
                    "demanda_promedio_sem1_horizonte": resultados_detallados_semana["demanda_promedio_semana1"].get((q,l), 0),
                    "shortage_promedio_sem1_horizonte": resultados_detallados_semana["shortage_promedio_semana1"].get((q,l), 0),
                    "inventario_inicial_sem1_horizonte": current_inventory_for_opt.get((q,l),0),
                    "inventario_final_sem1_horizonte": resultados_detallados_semana["inventario_final_semana1"].get((q,l), 0)
                })
        
        # Actualizar inventario para próxima semana
        current_inventory_for_opt = resultados_detallados_semana["inventario_final_semana1"]
        
        # Métricas de la semana
        metricas_por_semana.append({
            "semana": semana_idx_año,
            "tiempo_optimizacion_seg": tiempo_optimizacion,
            "utilidad_optimizador": mejor_utilidad_optimizador,
            "precio_p0t0": precios_optimos_semana_actual_np[0,0]
        })
    
    tiempo_total = time.time() - tiempo_inicio_total
    
    # Generar CSV temporal para analizar_utilidad.py
    df_resultados_temp = pd.DataFrame(all_weeks_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_temp_path = f"resultados/temp_experimento_{politica_nombre}_{n_particles}_{timestamp}.csv"
    os.makedirs("resultados", exist_ok=True)
    df_resultados_temp.to_csv(csv_temp_path, index=False, sep=';', decimal='.')
    
    print(f"\n📊 CSV temporal generado: {csv_temp_path}")
    
    # Calcular utilidad REAL usando analizar_utilidad.py
    print(f"🔍 Calculando utilidad real con analizar_utilidad.py...")
    try:
        utilidad_real_total, utilidad_por_semana_real = calcular_utilidad_total(
            path_csv=csv_temp_path,
            path_parametros="parametros/General_Parameters.csv",
            exportar_csv=False  # No exportar CSV detallado por ahora
        )
        
        print(f"✅ Utilidad real calculada: {utilidad_real_total:.2f}")
        
    except Exception as e:
        print(f"❌ Error calculando utilidad real: {e}")
        utilidad_real_total = -999999
        utilidad_por_semana_real = pd.Series([np.nan] * n_semanas)
    
    # Limpiar archivo temporal
    if os.path.exists(csv_temp_path):
        os.remove(csv_temp_path)
        print(f"🗑️  Archivo temporal eliminado: {csv_temp_path}")
    
    # Compilar métricas finales
    metricas_finales = {
        "politica": politica_nombre,
        "n_particulas_inicial": n_particles,
        "seed_global": seed_global,
        "tiempo_total_seg": tiempo_total,
        "utilidad_real_total": utilidad_real_total,
        "utilidad_optimizador_promedio": np.mean(utilidades_optimizador),
        "utilidad_optimizador_mejor": max(utilidades_optimizador),
        "utilidad_optimizador_std": np.std(utilidades_optimizador),
        "gap_utilidad_real_vs_optimizador": utilidad_real_total - np.mean(utilidades_optimizador),
        "precio_p0t0_promedio": np.mean([m["precio_p0t0"] for m in metricas_por_semana]),
        "tiempo_promedio_por_semana": tiempo_total / n_semanas,
        # Métricas por semana individual
        "utilidad_real_semana_1": utilidad_por_semana_real.iloc[0] if len(utilidad_por_semana_real) > 0 else np.nan,
        "utilidad_real_semana_2": utilidad_por_semana_real.iloc[1] if len(utilidad_por_semana_real) > 1 else np.nan,
        "utilidad_real_semana_3": utilidad_por_semana_real.iloc[2] if len(utilidad_por_semana_real) > 2 else np.nan,
        "utilidad_real_semana_4": utilidad_por_semana_real.iloc[3] if len(utilidad_por_semana_real) > 3 else np.nan,
    }
    
    print(f"\n📈 RESUMEN FINAL:")
    print(f"   Utilidad Real Total: {utilidad_real_total:,.2f}")
    print(f"   Utilidad Optimizador Promedio: {np.mean(utilidades_optimizador):,.2f}")
    print(f"   Gap Real vs Optimizador: {metricas_finales['gap_utilidad_real_vs_optimizador']:,.2f}")
    print(f"   Tiempo Total: {tiempo_total:.2f} segundos")
    
    return metricas_finales


def main():
    """
    Función principal que ejecuta el experimento completo con MISMO SEED para todas las configuraciones.
    """
    print("🚀 INICIANDO EXPERIMENTO DE POLÍTICAS DE RESAMPLING - MISMO SEED")
    print("=" * 80)
    print("⚠️  TODAS LAS POLÍTICAS USARÁN EXACTAMENTE EL MISMO SEED PARA COMPARACIÓN JUSTA")
    print("=" * 80)
    
    # Configuración del experimento
    RUTA_DATOS = "parametros"
    N_PRODUCTOS = 10
    N_TIENDAS = 2
    N_SEMANAS_PLANIFICACION = 4
    SEED_BASE = 42  # Seed fijo para TODAS las configuraciones
    
    # Configuraciones a probar
    configuraciones = [
        {"politica": "sin_resampling", "n_particulas_lista": [25, 50, 100]},
        {"politica": "actual", "n_particulas_lista": [25, 50, 100]},
        {"politica": "agresivo", "n_particulas_lista": [25, 50, 100]},
        {"politica": "conservador", "n_particulas_lista": [25, 50, 100]},
        {"politica": "adaptativo", "n_particulas_lista": [25, 50, 100]}
    ]
    
    # Verificar que existen los archivos necesarios
    if not os.path.exists(RUTA_DATOS):
        print(f"❌ ERROR: La carpeta '{RUTA_DATOS}' no existe.")
        return
    
    print(f"📋 CONFIGURACIÓN DEL EXPERIMENTO:")
    print(f"   Productos: {N_PRODUCTOS}")
    print(f"   Tiendas: {N_TIENDAS}")
    print(f"   Semanas: {N_SEMANAS_PLANIFICACION}")
    print(f"   🎲 Seed ÚNICO para TODAS las configuraciones: {SEED_BASE}")
    print(f"   Total de Configuraciones: {sum(len(c['n_particulas_lista']) for c in configuraciones)}")
    print()
    
    # Ejecutar experimento
    resultados_experimento = []
    tiempo_inicio_experimento = time.time()
    
    # CONFIGURACIÓN GLOBAL DE SEEDS
    set_global_eval_seed(SEED_BASE)  # Para optimizador
    set_analizar_utilidad_seed(SEED_BASE)  # Para simulaciones de demanda real
    
    for i, config in enumerate(configuraciones, 1):
        politica = config["politica"]
        n_particulas_lista = config["n_particulas_lista"]
        
        print(f"\n{'🔄' * 20}")
        print(f"POLÍTICA {i}/{len(configuraciones)}: {politica.upper()}")
        print(f"{'🔄' * 20}")
        
        for j, n_particulas in enumerate(n_particulas_lista, 1):
            print(f"\n--- Configuración {j}/{len(n_particulas_lista)} para {politica} ---")
            
            # ✅ TODAS LAS CONFIGURACIONES USAN EL MISMO SEED
            seed_config = SEED_BASE  # Mismo seed para TODAS las políticas y configuraciones
            print(f"🎲 Usando seed común: {seed_config}")
            
            try:
                # Limpiar archivos de demanda real previos
                demanda_real_path = "resultados/demanda_real.csv"
                if os.path.exists(demanda_real_path):
                    os.remove(demanda_real_path)
                    print(f"🗑️  Limpiado: {demanda_real_path}")
                
                # Ejecutar optimización
                metricas = ejecutar_optimizacion_con_politica(
                    politica_nombre=politica,
                    n_particles=n_particulas,
                    seed_global=seed_config,
                    ruta_datos=RUTA_DATOS,
                    n_productos=N_PRODUCTOS,
                    n_tiendas=N_TIENDAS,
                    n_semanas=N_SEMANAS_PLANIFICACION
                )
                
                resultados_experimento.append(metricas)
                
            except Exception as e:
                print(f"❌ ERROR en {politica} con {n_particulas} partículas: {e}")
                # Agregar resultado de error
                metricas_error = {
                    "politica": politica,
                    "n_particulas_inicial": n_particulas,
                    "seed_global": seed_config,
                    "tiempo_total_seg": np.nan,
                    "utilidad_real_total": -999999,
                    "error": str(e)
                }
                resultados_experimento.append(metricas_error)
    
    tiempo_total_experimento = time.time() - tiempo_inicio_experimento
    
    # Guardar resultados en CSV
    df_resultados = pd.DataFrame(resultados_experimento)
    timestamp_final = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_resultados_path = f"resultados/experimento_resampling_mismo_seed_{timestamp_final}.csv"
    
    os.makedirs("resultados", exist_ok=True)
    df_resultados.to_csv(csv_resultados_path, index=False, sep=';', decimal=',')
    
    print(f"\n{'🎉' * 30}")
    print("EXPERIMENTO COMPLETADO - MISMO SEED")
    print(f"{'🎉' * 30}")
    print(f"⏱️  Tiempo Total del Experimento: {tiempo_total_experimento/60:.1f} minutos")
    print(f"📊 Resultados guardados en: {csv_resultados_path}")
    print(f"📈 Total de configuraciones ejecutadas: {len(resultados_experimento)}")
    print(f"🎲 TODAS las configuraciones usaron seed: {SEED_BASE}")
    
    # Mostrar resumen de mejores resultados
    print(f"\n📋 RESUMEN DE UTILIDADES (MISMO SEED):")
    if not df_resultados.empty and 'utilidad_real_total' in df_resultados.columns:
        df_valid = df_resultados[df_resultados['utilidad_real_total'] > -999999]
        if not df_valid.empty:
            df_resumen = df_valid.groupby(['politica', 'n_particulas_inicial'])['utilidad_real_total'].max().reset_index()
            df_resumen = df_resumen.sort_values('utilidad_real_total', ascending=False)
            print(df_resumen.to_string(index=False))
        else:
            print("⚠️  No se obtuvieron resultados válidos.")
    
    print(f"\n✅ Archivo de resultados: {csv_resultados_path}")
    print("🔍 Usa este CSV para hacer tus análisis y gráficos con comparación completamente justa.")


if __name__ == "__main__":
    main() 