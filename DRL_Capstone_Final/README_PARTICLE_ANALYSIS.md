# Análisis de Eficiencia del Filtro de Partículas

Este módulo implementa un sistema completo para analizar la eficiencia del algoritmo de filtro de partículas con resampling múltiple, comparando diferentes cantidades de partículas iniciales.

## 🎯 Objetivo

Evaluar el valor que aporta el resampling y determinar si es posible reducir la cantidad de partículas iniciales manteniendo la calidad de los resultados.

## 📁 Archivos del Sistema

- **`particle_analysis.py`**: Clase principal para ejecutar experimentos comparativos
- **`run_particle_analysis.py`**: Script principal para ejecutar el análisis completo  
- **`visualize_particle_analysis.py`**: Generador de gráficos y visualizaciones
- **`README_PARTICLE_ANALYSIS.md`**: Esta documentación

## 🚀 Cómo Usar

### 1. Ejecutar el Análisis

```bash
python run_particle_analysis.py
```

Este script:
- Prueba configuraciones con 10, 20, 50 y 100 partículas iniciales
- Usa el mismo seed para reproducibilidad
- Simula 4 semanas de planificación para cada configuración
- Captura métricas detalladas de tiempo, utilidad y mejoras por resampling

### 2. Generar Visualizaciones

```bash
python visualize_particle_analysis.py
```

Genera automáticamente:
- Gráficos de comparación de eficiencia
- Análisis del valor del resampling  
- Evolución por semanas
- Tabla resumen con estadísticas clave

## 📊 Métricas Capturadas

### Métricas Principales
- **`n_particles`**: Número de partículas iniciales
- **`best_utility`**: Mejor utilidad encontrada
- **`optimization_time_seconds`**: Tiempo de optimización
- **`total_particles_evaluated`**: Partículas totales evaluadas
- **`best_initial_utility`**: Mejor utilidad de la etapa inicial
- **`improvement_from_resampling`**: Mejora absoluta por resampling
- **`improvement_percentage`**: Mejora porcentual por resampling

### Métricas de Eficiencia
- **Eficiencia**: Utilidad por segundo
- **ROI del Resampling**: Mejora por partícula adicional evaluada
- **Partículas por segundo**: Velocidad de evaluación

### Métricas de Negocio
- **`final_inventory_p0t0`**: Inventario final para producto 0, tienda 0
- **`demanda_promedio_p0t0`**: Demanda promedio
- **`shortage_promedio_p0t0`**: Escasez promedio

## 🔬 Configuración del Experimento

```python
# Cantidades de partículas a probar
PARTICLE_COUNTS = [10, 20, 50, 100]

# Configuración del problema
N_PRODUCTOS = 10
N_TIENDAS = 2  
N_SEMANAS = 4

# Semilla para reproducibilidad
BASE_SEED = 12345
```

### Generación Determinística de Seeds

Para cada combinación de partículas y semana:
```python
experiment_seed = base_seed + (n_particles * 1000) + semana_idx
```

Esto garantiza que cada experimento sea reproducible pero independiente.

## 📈 Análisis Esperados

### 1. **Eficiencia vs. Cantidad de Partículas**
- ¿Más partículas siempre es mejor?
- ¿Cuál es el punto de rendimientos decrecientes?

### 2. **Valor del Resampling**
- ¿Qué tanto mejora el resampling los resultados?
- ¿Es consistente esta mejora?

### 3. **ROI del Resampling**  
- ¿Vale la pena invertir en más evaluaciones de partículas?
- ¿Qué configuración ofrece mejor ROI?

### 4. **Convergencia Temporal**
- ¿Los patrones son consistentes a lo largo de las semanas?
- ¿Hay diferencias en comportamiento por semana?

## 📁 Estructura de Resultados

```
resultados/
├── particle_analysis_YYYYMMDD_HHMMSS.csv     # Datos completos del análisis
├── visualizations/
│   ├── efficiency_comparison.png              # Comparación de eficiencia
│   ├── resampling_analysis.png                # Análisis del resampling
│   ├── convergence_analysis.png               # Análisis por semanas
│   ├── summary_table.png                      # Tabla resumen visual
│   └── summary_statistics.csv                 # Estadísticas resumen
```

## 🔧 Personalización

### Modificar Cantidades de Partículas

En `run_particle_analysis.py`:
```python
PARTICLE_COUNTS = [5, 15, 30, 75, 150]  # Ejemplo personalizado
```

### Cambiar Duración del Experimento

```python
N_SEMANAS = 8  # Simular más semanas
```

### Agregar Nuevas Métricas

En `particle_analysis.py`, modifica el diccionario `week_metrics`:
```python
week_metrics = {
    # ... métricas existentes ...
    'nueva_metrica': calcular_nueva_metrica(),
}
```

## 📋 Interpretación de Resultados

### Indicadores Clave

1. **Mejor Eficiencia**: Configuración con mayor `utilidad/tiempo`
2. **Mejor ROI**: Configuración con mayor `mejora_resampling/partículas_extra`  
3. **Consistencia**: Menor desviación estándar en utilidades
4. **Escalabilidad**: Relación lineal/exponencial entre partículas y tiempo

### Ejemplo de Análisis

```
🏆 Mejor eficiencia: 20 partículas (1,250 utilidad/segundo)
🎯 Mejor utilidad absoluta: 100 partículas (125,000 utilidad promedio)  
📈 Mejor mejora por resampling: 10 partículas (8,500 mejora promedio)
💡 El resampling aporta en promedio 7,200 puntos de utilidad
```

## ⚠️ Consideraciones

1. **Tiempo de Ejecución**: El análisis completo puede tomar 10-30 minutos
2. **Reproducibilidad**: Usa seeds determinísticos para resultados consistentes
3. **Memoria**: Almacena todas las partículas evaluadas para análisis posterior
4. **Dependencias**: Requiere matplotlib, seaborn, pandas, numpy

## 🔄 Flujo Completo

1. **Preparación**: Verificar que existe carpeta `parametros/`
2. **Ejecución**: `python run_particle_analysis.py`
3. **Visualización**: `python visualize_particle_analysis.py`  
4. **Análisis**: Revisar CSV y gráficos generados
5. **Decisión**: Seleccionar configuración óptima basada en métricas

## 📞 Uso en Producción

Basado en los resultados, puedes modificar `main_particle.py`:

```python
# Si el análisis muestra que 20 partículas son óptimas:
N_PARTICLES = 20  # Cambiar de 50 a 20
```

Este enfoque permite optimizar tanto la calidad como la velocidad del algoritmo de optimización de precios. 