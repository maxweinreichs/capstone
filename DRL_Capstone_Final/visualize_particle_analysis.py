#!/usr/bin/env python3
"""
Script para generar visualizaciones de los resultados del análisis de partículas.
Crea gráficos útiles para analizar la eficiencia del resampling.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
from pathlib import Path

def load_analysis_results(filepath):
    """
    Carga los resultados del análisis desde un archivo CSV.
    """
    try:
        df = pd.read_csv(filepath, sep=';', decimal='.')
        print(f"✅ Datos cargados exitosamente: {len(df)} registros")
        return df
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return None

def create_efficiency_comparison(df, output_dir):
    """
    Crea gráficos de comparación de eficiencia entre diferentes cantidades de partículas.
    """
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análisis de Eficiencia del Filtro de Partículas', fontsize=16, fontweight='bold')
    
    # 1. Utilidad promedio por cantidad de partículas
    ax1 = axes[0, 0]
    utility_by_particles = df.groupby('n_particles')['best_utility'].agg(['mean', 'std']).reset_index()
    ax1.bar(utility_by_particles['n_particles'], utility_by_particles['mean'], 
            yerr=utility_by_particles['std'], capsize=5, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Número de Partículas Iniciales')
    ax1.set_ylabel('Utilidad Promedio')
    ax1.set_title('Utilidad Promedio por Configuración')
    ax1.grid(True, alpha=0.3)
    
    # 2. Tiempo de optimización
    ax2 = axes[0, 1]
    time_by_particles = df.groupby('n_particles')['optimization_time_seconds'].agg(['mean', 'std']).reset_index()
    ax2.bar(time_by_particles['n_particles'], time_by_particles['mean'], 
            yerr=time_by_particles['std'], capsize=5, alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Número de Partículas Iniciales')
    ax2.set_ylabel('Tiempo (segundos)')
    ax2.set_title('Tiempo de Optimización')
    ax2.grid(True, alpha=0.3)
    
    # 3. Eficiencia (utilidad por segundo)
    ax3 = axes[1, 0]
    df['efficiency'] = df['best_utility'] / df['optimization_time_seconds']
    efficiency_by_particles = df.groupby('n_particles')['efficiency'].agg(['mean', 'std']).reset_index()
    ax3.bar(efficiency_by_particles['n_particles'], efficiency_by_particles['mean'], 
            yerr=efficiency_by_particles['std'], capsize=5, alpha=0.7, color='lightgreen')
    ax3.set_xlabel('Número de Partículas Iniciales')
    ax3.set_ylabel('Utilidad por Segundo')
    ax3.set_title('Eficiencia (Utilidad/Tiempo)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Mejora por resampling
    ax4 = axes[1, 1]
    improvement_by_particles = df.groupby('n_particles')['improvement_from_resampling'].agg(['mean', 'std']).reset_index()
    ax4.bar(improvement_by_particles['n_particles'], improvement_by_particles['mean'], 
            yerr=improvement_by_particles['std'], capsize=5, alpha=0.7, color='orange')
    ax4.set_xlabel('Número de Partículas Iniciales')
    ax4.set_ylabel('Mejora de Utilidad')
    ax4.set_title('Mejora por Resampling')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'efficiency_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Gráfico de eficiencia guardado: {filepath}")

def create_resampling_analysis(df, output_dir):
    """
    Crea análisis detallado del valor del resampling.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análisis del Valor del Resampling', fontsize=16, fontweight='bold')
    
    # 1. Distribución de mejoras por resampling
    ax1 = axes[0, 0]
    for n_particles in sorted(df['n_particles'].unique()):
        subset = df[df['n_particles'] == n_particles]
        ax1.hist(subset['improvement_from_resampling'], alpha=0.6, 
                label=f'{n_particles} partículas', bins=15)
    ax1.set_xlabel('Mejora de Utilidad')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Distribución de Mejoras por Resampling')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. ROI del resampling (mejora por partícula extra)
    ax2 = axes[0, 1]
    df['extra_particles'] = df['total_particles_evaluated'] - df['n_particles']
    df['roi_resampling'] = df['improvement_from_resampling'] / df['extra_particles']
    roi_by_particles = df.groupby('n_particles')['roi_resampling'].agg(['mean', 'std']).reset_index()
    ax2.bar(roi_by_particles['n_particles'], roi_by_particles['mean'], 
            yerr=roi_by_particles['std'], capsize=5, alpha=0.7, color='purple')
    ax2.set_xlabel('Número de Partículas Iniciales')
    ax2.set_ylabel('Utilidad por Partícula Extra')
    ax2.set_title('ROI del Resampling')
    ax2.grid(True, alpha=0.3)
    
    # 3. Relación entre partículas iniciales y totales
    ax3 = axes[1, 0]
    for n_particles in sorted(df['n_particles'].unique()):
        subset = df[df['n_particles'] == n_particles]
        ax3.scatter(subset['n_particles'], subset['total_particles_evaluated'], 
                   alpha=0.6, label=f'{n_particles} iniciales', s=50)
    ax3.set_xlabel('Partículas Iniciales')
    ax3.set_ylabel('Partículas Totales Evaluadas')
    ax3.set_title('Partículas Iniciales vs. Totales')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Mejora porcentual por resampling
    ax4 = axes[1, 1]
    improvement_pct_by_particles = df.groupby('n_particles')['improvement_percentage'].agg(['mean', 'std']).reset_index()
    ax4.bar(improvement_pct_by_particles['n_particles'], improvement_pct_by_particles['mean'], 
            yerr=improvement_pct_by_particles['std'], capsize=5, alpha=0.7, color='teal')
    ax4.set_xlabel('Número de Partículas Iniciales')
    ax4.set_ylabel('Mejora Porcentual (%)')
    ax4.set_title('Mejora Porcentual por Resampling')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'resampling_analysis.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Gráfico de resampling guardado: {filepath}")

def create_convergence_analysis(df, output_dir):
    """
    Crea análisis de convergencia a lo largo de las semanas.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análisis de Convergencia por Semanas', fontsize=16, fontweight='bold')
    
    # 1. Utilidad por semana y configuración
    ax1 = axes[0, 0]
    for n_particles in sorted(df['n_particles'].unique()):
        subset = df[df['n_particles'] == n_particles]
        utility_by_week = subset.groupby('semana')['best_utility'].mean()
        ax1.plot(utility_by_week.index, utility_by_week.values, 
                marker='o', linewidth=2, label=f'{n_particles} partículas')
    ax1.set_xlabel('Semana')
    ax1.set_ylabel('Utilidad Promedio')
    ax1.set_title('Evolución de Utilidad por Semana')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Tiempo por semana
    ax2 = axes[0, 1]
    for n_particles in sorted(df['n_particles'].unique()):
        subset = df[df['n_particles'] == n_particles]
        time_by_week = subset.groupby('semana')['optimization_time_seconds'].mean()
        ax2.plot(time_by_week.index, time_by_week.values, 
                marker='s', linewidth=2, label=f'{n_particles} partículas')
    ax2.set_xlabel('Semana')
    ax2.set_ylabel('Tiempo Promedio (s)')
    ax2.set_title('Tiempo de Optimización por Semana')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Mejora por resampling por semana
    ax3 = axes[1, 0]
    for n_particles in sorted(df['n_particles'].unique()):
        subset = df[df['n_particles'] == n_particles]
        improvement_by_week = subset.groupby('semana')['improvement_from_resampling'].mean()
        ax3.plot(improvement_by_week.index, improvement_by_week.values, 
                marker='^', linewidth=2, label=f'{n_particles} partículas')
    ax3.set_xlabel('Semana')
    ax3.set_ylabel('Mejora Promedio')
    ax3.set_title('Mejora por Resampling por Semana')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Inventario final por semana (P0T0)
    ax4 = axes[1, 1]
    for n_particles in sorted(df['n_particles'].unique()):
        subset = df[df['n_particles'] == n_particles]
        inventory_by_week = subset.groupby('semana')['final_inventory_p0t0'].mean()
        ax4.plot(inventory_by_week.index, inventory_by_week.values, 
                marker='d', linewidth=2, label=f'{n_particles} partículas')
    ax4.set_xlabel('Semana')
    ax4.set_ylabel('Inventario Final Promedio')
    ax4.set_title('Inventario Final P0T0 por Semana')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'convergence_analysis.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Gráfico de convergencia guardado: {filepath}")

def create_summary_table(df, output_dir):
    """
    Crea una tabla resumen con las métricas más importantes.
    """
    summary_stats = []
    
    for n_particles in sorted(df['n_particles'].unique()):
        subset = df[df['n_particles'] == n_particles]
        
        stats = {
            'Partículas Iniciales': n_particles,
            'Utilidad Promedio': subset['best_utility'].mean(),
            'Desv. Est. Utilidad': subset['best_utility'].std(),
            'Tiempo Promedio (s)': subset['optimization_time_seconds'].mean(),
            'Eficiencia (Util/Tiempo)': (subset['best_utility'] / subset['optimization_time_seconds']).mean(),
            'Mejora por Resampling': subset['improvement_from_resampling'].mean(),
            'Mejora Porcentual (%)': subset['improvement_percentage'].mean(),
            'Partículas Totales': subset['total_particles_evaluated'].mean(),
            'ROI Resampling': (subset['improvement_from_resampling'] / (subset['total_particles_evaluated'] - subset['n_particles'])).mean()
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Guardar como CSV
    csv_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary_df.to_csv(csv_path, index=False, sep=';', decimal='.')
    
    # Crear visualización de la tabla
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Formatear números para mejor presentación
    formatted_data = summary_df.copy()
    for col in ['Utilidad Promedio', 'Desv. Est. Utilidad', 'Mejora por Resampling']:
        formatted_data[col] = formatted_data[col].apply(lambda x: f'{x:,.1f}')
    for col in ['Tiempo Promedio (s)', 'Eficiencia (Util/Tiempo)', 'ROI Resampling']:
        formatted_data[col] = formatted_data[col].apply(lambda x: f'{x:.2f}')
    for col in ['Mejora Porcentual (%)', 'Partículas Totales']:
        formatted_data[col] = formatted_data[col].apply(lambda x: f'{x:.1f}')
    
    table = ax.table(cellText=formatted_data.values, colLabels=formatted_data.columns,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    
    # Colorear encabezados
    for i in range(len(formatted_data.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Resumen Estadístico - Análisis de Filtro de Partículas', 
              fontsize=14, fontweight='bold', pad=20)
    
    table_path = os.path.join(output_dir, 'summary_table.png')
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Tabla resumen guardada: {table_path}")
    print(f"📄 CSV resumen guardado: {csv_path}")

def main():
    """
    Función principal para generar todas las visualizaciones.
    """
    print("📊 === GENERADOR DE VISUALIZACIONES ===")
    
    # Buscar archivos de análisis de partículas
    results_dir = "resultados"
    if not os.path.exists(results_dir):
        print(f"❌ El directorio '{results_dir}' no existe.")
        return
    
    # Buscar archivos CSV de análisis
    analysis_files = [f for f in os.listdir(results_dir) if f.startswith('particle_analysis_') and f.endswith('.csv')]
    
    if not analysis_files:
        print(f"❌ No se encontraron archivos de análisis en '{results_dir}'.")
        print("Ejecuta primero 'run_particle_analysis.py' para generar los datos.")
        return
    
    # Usar el archivo más reciente
    latest_file = sorted(analysis_files)[-1]
    filepath = os.path.join(results_dir, latest_file)
    print(f"📁 Usando archivo: {filepath}")
    
    # Cargar datos
    df = load_analysis_results(filepath)
    if df is None:
        return
    
    # Crear directorio para visualizaciones
    viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"\n🎨 Generando visualizaciones...")
    
    try:
        # Generar todos los gráficos
        create_efficiency_comparison(df, viz_dir)
        create_resampling_analysis(df, viz_dir)
        create_convergence_analysis(df, viz_dir)
        create_summary_table(df, viz_dir)
        
        print(f"\n✅ Visualizaciones completadas!")
        print(f"📁 Archivos guardados en: {viz_dir}")
        
    except Exception as e:
        print(f"❌ Error generando visualizaciones: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 