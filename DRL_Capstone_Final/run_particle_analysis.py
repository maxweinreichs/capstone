#!/usr/bin/env python3
"""
Script para analizar el comportamiento del filtro de partículas con diferentes cantidades iniciales.
Ejecuta experimentos reproducibles usando el mismo seed para comparar eficiencia del resampling.
"""

import os
import sys
import time
from particle_analysis import ParticleAnalyzer

def main():
    """
    Función principal que ejecuta el análisis de partículas.
    """
    print("🚀 === ANÁLISIS DE EFICIENCIA DEL FILTRO DE PARTÍCULAS ===")
    print("Este análisis evalúa cómo el resampling afecta la eficiencia del algoritmo")
    print("comparando diferentes cantidades de partículas iniciales.\n")
    
    # Configuración del experimento
    RUTA_DATOS = "parametros"
    N_PRODUCTOS = 10
    N_TIENDAS = 2
    N_SEMANAS = 4
    BASE_SEED = 12345
    
    # Cantidades de partículas a probar
    PARTICLE_COUNTS = [10, 20, 50, 100]
    
    print("📋 Configuración del experimento:")
    print(f"  • Ruta de datos: {RUTA_DATOS}")
    print(f"  • Productos: {N_PRODUCTOS}")
    print(f"  • Tiendas: {N_TIENDAS}")
    print(f"  • Semanas a simular: {N_SEMANAS}")
    print(f"  • Cantidades de partículas: {PARTICLE_COUNTS}")
    print(f"  • Seed base: {BASE_SEED}")
    
    # Verificar que existe la carpeta de parámetros
    if not os.path.exists(RUTA_DATOS):
        print(f"\n❌ ERROR: La carpeta '{RUTA_DATOS}' no existe.")
        print("Por favor, asegúrate de que los archivos de parámetros estén disponibles.")
        return
    
    # Inicializar analizador
    analyzer = ParticleAnalyzer(
        ruta_datos=RUTA_DATOS,
        n_productos=N_PRODUCTOS,
        n_tiendas=N_TIENDAS
    )
    
    # Ejecutar análisis
    start_time = time.time()
    
    try:
        print(f"\n⏳ Iniciando análisis... (esto puede tomar varios minutos)")
        metrics_data = analyzer.analyze_particle_counts(
            particle_counts=PARTICLE_COUNTS,
            n_semanas=N_SEMANAS,
            base_seed=BASE_SEED
        )
        
        # Guardar resultados
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"particle_analysis_{timestamp}.csv"
        filepath = analyzer.save_results(filename)
        
        total_time = time.time() - start_time
        print(f"\n🎯 === ANÁLISIS COMPLETADO ===")
        print(f"⏱️  Tiempo total: {total_time:.2f} segundos ({total_time/60:.1f} minutos)")
        print(f"📁 Archivo de resultados: {filepath}")
        
        # Mostrar insights clave
        print_key_insights(metrics_data)
        
    except Exception as e:
        print(f"\n❌ ERROR durante el análisis: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n✅ Análisis completado exitosamente!")
    print(f"📊 Puedes usar el archivo CSV para crear gráficos y análisis adicionales.")

def print_key_insights(metrics_data):
    """
    Imprime insights clave del análisis.
    """
    if not metrics_data:
        return
        
    import pandas as pd
    df = pd.DataFrame(metrics_data)
    
    print(f"\n🔍 === INSIGHTS CLAVE ===")
    
    # Análisis de eficiencia por cantidad de partículas
    efficiency_analysis = []
    for n_particles in sorted(df['n_particles'].unique()):
        subset = df[df['n_particles'] == n_particles]
        avg_utility = subset['best_utility'].mean()
        avg_time = subset['optimization_time_seconds'].mean()
        avg_improvement = subset['improvement_from_resampling'].mean()
        avg_total_particles = subset['total_particles_evaluated'].mean()
        
        efficiency_analysis.append({
            'n_particles': n_particles,
            'avg_utility': avg_utility,
            'avg_time': avg_time,
            'utility_per_second': avg_utility / avg_time if avg_time > 0 else 0,
            'avg_improvement': avg_improvement,
            'avg_total_particles': avg_total_particles,
            'particles_per_second': avg_total_particles / avg_time if avg_time > 0 else 0
        })
    
    # Encontrar la configuración más eficiente
    best_efficiency = max(efficiency_analysis, key=lambda x: x['utility_per_second'])
    best_absolute = max(efficiency_analysis, key=lambda x: x['avg_utility'])
    best_improvement = max(efficiency_analysis, key=lambda x: x['avg_improvement'])
    
    print(f"🏆 Mejor eficiencia (utilidad/tiempo): {best_efficiency['n_particles']} partículas")
    print(f"   → {best_efficiency['utility_per_second']:.2f} utilidad por segundo")
    
    print(f"🎯 Mejor utilidad absoluta: {best_absolute['n_particles']} partículas")
    print(f"   → {best_absolute['avg_utility']:,.2f} utilidad promedio")
    
    print(f"📈 Mejor mejora por resampling: {best_improvement['n_particles']} partículas")
    print(f"   → {best_improvement['avg_improvement']:,.2f} mejora promedio")
    
    # Análisis de valor del resampling
    total_improvements = [x['avg_improvement'] for x in efficiency_analysis if x['avg_improvement'] > 0]
    if total_improvements:
        avg_improvement_across_all = sum(total_improvements) / len(total_improvements)
        print(f"\n💡 El resampling aporta en promedio {avg_improvement_across_all:,.2f} puntos de utilidad")
        
        # Análisis de ROI del resampling
        print(f"\n📊 ROI del Resampling (por configuración):")
        for config in efficiency_analysis:
            if config['avg_improvement'] > 0:
                # Calcular partículas "extra" por resampling (total - inicial)
                extra_particles = config['avg_total_particles'] - config['n_particles']
                roi = config['avg_improvement'] / extra_particles if extra_particles > 0 else 0
                print(f"   • {config['n_particles']} partículas: {roi:.3f} utilidad por partícula extra")

if __name__ == "__main__":
    main() 