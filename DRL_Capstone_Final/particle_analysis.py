import numpy as np
import pandas as pd
import time
import os
from optimizador import calcular_resultados_optimizacion, load_static_params_once, set_global_eval_seed
from particle_filter import particle_filter_optimization_multi_resample

class ParticleAnalyzer:
    """
    Clase para analizar el comportamiento del filtro de partículas con diferentes configuraciones.
    """
    
    def __init__(self, ruta_datos, n_productos, n_tiendas):
        self.ruta_datos = ruta_datos
        self.n_productos = n_productos
        self.n_tiendas = n_tiendas
        self.metrics_data = []
        
    def analyze_particle_counts(self, particle_counts, n_semanas=4, base_seed=12345):
        """
        Analiza el comportamiento del filtro de partículas con diferentes cantidades iniciales.
        
        Args:
            particle_counts: Lista de cantidades de partículas a probar (ej: [10, 20, 50, 100])
            n_semanas: Número de semanas a simular
            base_seed: Seed base para reproducibilidad
        """
        print(f"\n🔬 === ANÁLISIS DE FILTRO DE PARTÍCULAS ===")
        print(f"Probando cantidades de partículas: {particle_counts}")
        print(f"Semanas a simular: {n_semanas}")
        print(f"Seed base: {base_seed}")
        
        # Cargar parámetros estáticos una vez
        static_params = load_static_params_once(self.ruta_datos, self.n_productos, self.n_tiendas)
        
        for n_particles in particle_counts:
            print(f"\n📊 === Probando con {n_particles} partículas iniciales ===")
            
            # Ejecutar experimento con esta cantidad de partículas
            experiment_data = self._run_experiment(
                n_particles=n_particles,
                n_semanas=n_semanas,
                static_params=static_params,
                base_seed=base_seed
            )
            
            # Agregar los datos del experimento a la lista
            self.metrics_data.extend(experiment_data)
            
        return self.metrics_data
    
    def _run_experiment(self, n_particles, n_semanas, static_params, base_seed):
        """
        Ejecuta un experimento completo con una cantidad específica de partículas.
        """
        experiment_data = []
        current_inventory = static_params["I_initial_global"].copy()
        
        for semana_idx in range(1, n_semanas + 1):
            print(f"  🗓️  Semana {semana_idx} con {n_particles} partículas...")
            
            # Configurar seed determinístico para esta combinación
            experiment_seed = base_seed + (n_particles * 1000) + semana_idx
            np.random.seed(experiment_seed)
            set_global_eval_seed(experiment_seed)
            
            # Crear función de evaluación
            def optimizer_caller(precios_np):
                resultados = calcular_resultados_optimizacion(
                    precios_np, current_inventory,
                    semana_idx, self.ruta_datos, self.n_productos, self.n_tiendas,
                    use_eval_seed=False
                )
                return resultados["utilidad_total_horizonte"]
            
            # Ejecutar optimización con métricas detalladas
            start_time = time.time()
            best_prices, best_score, detailed_metrics = self._optimize_with_metrics(
                n_particles=n_particles,
                optimizer_caller=optimizer_caller,
                static_params=static_params,
                current_inventory=current_inventory,
                semana_idx=semana_idx
            )
            optimization_time = time.time() - start_time
            
            # Calcular resultados detallados para la mejor solución
            set_global_eval_seed(experiment_seed)
            resultados_detallados = calcular_resultados_optimizacion(
                best_prices, current_inventory,
                semana_idx, self.ruta_datos, self.n_productos, self.n_tiendas,
                use_eval_seed=True
            )
            
            # Recopilar métricas
            week_metrics = {
                'n_particles': n_particles,
                'semana': semana_idx,
                'experiment_seed': experiment_seed,
                'optimization_time_seconds': optimization_time,
                'best_utility': best_score,
                'best_price_p0t0': best_prices[0, 0],
                'total_particles_evaluated': detailed_metrics['total_particles_evaluated'],
                'initial_stage_particles': detailed_metrics['initial_stage_particles'],
                'resampling_rounds': detailed_metrics['resampling_rounds'],
                'best_initial_utility': detailed_metrics['best_initial_utility'],
                'improvement_from_resampling': best_score - detailed_metrics['best_initial_utility'],
                'improvement_percentage': ((best_score - detailed_metrics['best_initial_utility']) / abs(detailed_metrics['best_initial_utility'])) * 100 if detailed_metrics['best_initial_utility'] != 0 else 0,
                'particles_per_round': detailed_metrics['particles_per_round'],
                'utility_per_round': detailed_metrics['utility_per_round'],
                'final_inventory_p0t0': resultados_detallados["inventario_final_semana1"].get((0,0), 0),
                'demanda_promedio_p0t0': resultados_detallados["demanda_promedio_semana1"].get((0,0), 0),
                'shortage_promedio_p0t0': resultados_detallados["shortage_promedio_semana1"].get((0,0), 0)
            }
            
            experiment_data.append(week_metrics)
            
            # Actualizar inventario para la siguiente semana
            current_inventory = resultados_detallados["inventario_final_semana1"]
            
            print(f"    ✅ Completado en {optimization_time:.2f}s - Utilidad: {best_score:.2f}")
            
        return experiment_data
    
    def _optimize_with_metrics(self, n_particles, optimizer_caller, static_params, current_inventory, semana_idx):
        """
        Ejecuta la optimización capturando métricas detalladas del proceso.
        """
        # Preparar datos para métricas
        detailed_metrics = {
            'total_particles_evaluated': 0,
            'initial_stage_particles': n_particles,
            'resampling_rounds': 0,
            'best_initial_utility': -float('inf'),
            'particles_per_round': [],
            'utility_per_round': []
        }
        
        # Ejecutar optimización con instrumentación
        best_prices, best_score = self._instrumented_particle_filter(
            n_particles=n_particles,
            n_productos=self.n_productos,
            n_tiendas=self.n_tiendas,
            precios_base=static_params["precios_base_np"],
            evaluate_fn=optimizer_caller,
            semana_idx=semana_idx,
            metrics_collector=detailed_metrics
        )
        
        return best_prices, best_score, detailed_metrics
    
    def _instrumented_particle_filter(self, n_particles, n_productos, n_tiendas, 
                                    precios_base, evaluate_fn, semana_idx, metrics_collector):
        """
        Versión instrumentada del filtro de partículas que captura métricas detalladas.
        """
        from particle_filter import initialize_historical_prices, get_historical_prices_matrix, generate_particle_historical, generate_particle_from_base
        
        # Inicializar datos históricos
        precios_historicos_data = initialize_historical_prices()
        precios_historicos_semana_completa = get_historical_prices_matrix(
            precios_historicos_data, semana_idx, n_productos, n_tiendas
        )
        
        # Determinar precios de partida
        if precios_historicos_semana_completa is not None:
            precios_partida_tienda0_ref = precios_historicos_semana_completa[:, 0]
        else:
            if precios_base.ndim == 2:
                precios_partida_tienda0_ref = precios_base[:, 0]
            elif precios_base.ndim == 1 and len(precios_base) == n_productos:
                precios_partida_tienda0_ref = precios_base
            else:
                precios_partida_tienda0_ref = np.full((n_productos,), 30.0)
        
        # Etapa inicial
        initial_particles = []
        initial_scores = []
        
        for i in range(n_particles):
            p = generate_particle_historical(n_productos, n_tiendas, precios_partida_tienda0_ref, 
                                           semana_idx, variation_factor=0.3)
            score = evaluate_fn(p)
            initial_particles.append(p)
            initial_scores.append(score)
            metrics_collector['total_particles_evaluated'] += 1
        
        if initial_scores:
            metrics_collector['best_initial_utility'] = max(initial_scores)
            metrics_collector['particles_per_round'].append(('Initial', n_particles))
            metrics_collector['utility_per_round'].append(('Initial', max(initial_scores)))
        
        # Configuración de rondas de resampling
        rondas_config = [
            {"nombre": "Round 1", "n_particulas_ronda": 25, "n_mejores_base": 10, "variation_factor": 0.20},
            {"nombre": "Round 2", "n_particulas_ronda": 22, "n_mejores_base": 8,  "variation_factor": 0.10},
            {"nombre": "Round 3", "n_particulas_ronda": 18, "n_mejores_base": 6,  "variation_factor": 0.05}
        ]
        
        all_particles_acum = list(initial_particles)
        all_scores_acum = list(initial_scores)
        
        # Ejecutar rondas de resampling
        for ronda in rondas_config:
            metrics_collector['resampling_rounds'] += 1
            current_round_particles = []
            current_round_scores = []
            
            sorted_indices_acum = np.argsort(all_scores_acum)[::-1]
            n_bases_disponibles = min(ronda['n_mejores_base'], len(all_particles_acum))
            
            if n_bases_disponibles == 0:
                continue
                
            best_indices_base = sorted_indices_acum[:n_bases_disponibles]
            
            for i in range(ronda['n_particulas_ronda']):
                base_idx_actual = best_indices_base[i % n_bases_disponibles]
                base_particle_actual = all_particles_acum[base_idx_actual]
                
                p = generate_particle_from_base(base_particle_actual, 
                                              variation_factor=ronda['variation_factor'])
                score = evaluate_fn(p)
                current_round_particles.append(p)
                current_round_scores.append(score)
                metrics_collector['total_particles_evaluated'] += 1
            
            all_particles_acum.extend(current_round_particles)
            all_scores_acum.extend(current_round_scores)
            
            if current_round_scores:
                round_best = max(current_round_scores)
                metrics_collector['particles_per_round'].append((ronda['nombre'], ronda['n_particulas_ronda']))
                metrics_collector['utility_per_round'].append((ronda['nombre'], round_best))
        
        # Encontrar la mejor partícula
        if all_scores_acum:
            best_idx_final = np.argmax(all_scores_acum)
            best_particle_final = all_particles_acum[best_idx_final]
            best_score_final = all_scores_acum[best_idx_final]
        else:
            # Fallback
            best_particle_final = generate_particle_historical(n_productos, n_tiendas, precios_partida_tienda0_ref, 
                                                             semana_idx, variation_factor=0.0)
            best_score_final = -float('inf')
        
        return best_particle_final, best_score_final
    
    def save_results(self, filename="particle_analysis_results.csv"):
        """
        Guarda los resultados del análisis en un archivo CSV.
        """
        if not self.metrics_data:
            print("⚠️ No hay datos para guardar.")
            return
        
        df = pd.DataFrame(self.metrics_data)
        
        # Crear directorio de resultados si no existe
        os.makedirs("resultados", exist_ok=True)
        filepath = os.path.join("resultados", filename)
        
        # Guardar con formato específico
        df.to_csv(filepath, index=False, sep=';', decimal='.')
        
        print(f"\n💾 Resultados guardados en: {filepath}")
        print(f"📊 Total de registros: {len(df)}")
        
        # Mostrar resumen estadístico
        self._print_summary_stats(df)
        
        return filepath
    
    def _print_summary_stats(self, df):
        """
        Imprime estadísticas resumen de los resultados.
        """
        print("\n📈 === RESUMEN ESTADÍSTICO ===")
        
        for n_particles in df['n_particles'].unique():
            subset = df[df['n_particles'] == n_particles]
            avg_utility = subset['best_utility'].mean()
            avg_time = subset['optimization_time_seconds'].mean()
            avg_improvement = subset['improvement_from_resampling'].mean()
            avg_particles_total = subset['total_particles_evaluated'].mean()
            
            print(f"\n🔹 {n_particles} partículas iniciales:")
            print(f"  • Utilidad promedio: {avg_utility:,.2f}")
            print(f"  • Tiempo promedio: {avg_time:.2f}s")
            print(f"  • Mejora por resampling: {avg_improvement:,.2f} ({avg_improvement/avg_utility*100:.1f}%)")
            print(f"  • Partículas evaluadas total: {avg_particles_total:.0f}")
            print(f"  • Eficiencia (utilidad/tiempo): {avg_utility/avg_time:.2f}") 