# Sistema de Optimización de Precios con DRL y SAA

Este proyecto implementa un sistema de optimización de precios para retail utilizando Aprendizaje por Refuerzo Profundo (DRL) y Aproximación por Promedio de Muestras (SAA).


#Para correr el proyecto:




          
# Instrucciones para Configurar el Entorno Virtual

## Requisitos Previos

Para ejecutar este proyecto, necesitarás:

- Python **3.10**
- pip (gestor de paquetes de Python)
- Acceso a terminal/línea de comandos

## Pasos para Crear un Entorno Virtual

### 1. Clonar el Repositorio

```bash
git clone <url-del-repositorio>
cd DRL_Capstone_Final
```

### 2. Crear un Entorno Virtual con Python 3.10

#### En macOS/Linux:

```bash
python3.10 -m venv venv_py310
```

#### En Windows:

```bash
python -m venv venv_py310
```

### 3. Activar el Entorno Virtual

#### En macOS/Linux:

```bash
source venv_py310/bin/activate
```

#### En Windows:

```bash
venv_py310\Scripts\activate
```

### 4. Instalar Dependencias

Una vez activado el entorno virtual, instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

### 5. Verificar la Instalación de Gurobi

Este proyecto requiere Gurobi Optimizer. Asegúrate de tener una licencia válida y que esté correctamente instalado:

```bash
python -c "import gurobipy; print(gurobipy.gurobi.version())"
```

### 6. Eliminar carpetas y Ejecutar el Proyecto

Se deben eliminar las carpetas sb3_logs, logs_best_model y logs_eval antes de cada ejecucion.
Con el entorno virtual activado, puedes ejecutar el proyecto:

```bash
python main.py
```

## Notas Importantes

- El entorno virtual debe ser creado con Python 3.10 específicamente para asegurar la compatibilidad con todas las dependencias.
- Para desactivar el entorno virtual cuando hayas terminado, simplemente ejecuta `deactivate` en la terminal.
- Si encuentras problemas con alguna dependencia específica, puedes instalarla manualmente con `pip install nombre-del-paquete`.
- Los resultados se guardarán en la carpeta `resultados/`.

## Solución de Problemas

Si encuentras errores relacionados con TensorFlow o PyTorch, asegúrate de que estás utilizando versiones compatibles con tu sistema operativo y hardware.

Para más información sobre los requisitos y uso del sistema, consulta las secciones correspondientes en el README del proyecto.

        Too many current requests. Your queue position is 1. Please wait for a while or switch to other models for a smoother experience.



## Descripción

El sistema busca encontrar políticas óptimas de precios para múltiples productos en múltiples tiendas a lo largo de un horizonte de planificación (por defecto, 52 semanas). Utiliza datos reales de parámetros de demanda, costos y temporadas para simular escenarios realistas.

### Características principales:

- **DRL (Deep Reinforcement Learning)**: Utiliza el algoritmo PPO (Proximal Policy Optimization) para aprender políticas de precios óptimas.
- **SAA (Sample Average Approximation)**: Genera múltiples escenarios de demanda para cada decisión de precios y optimiza el promedio, lo que proporciona robustez frente a la incertidumbre.
- **Optimización con Gurobi**: Resuelve subproblemas de optimización para determinar pedidos óptimos dados los precios y la demanda.
- **Datos reales**: Utiliza parámetros de distribución de demanda y costos cargados desde archivos CSV.
- **Múltiples productos y tiendas**: Soporta múltiples productos (9) y tiendas (2) con diferentes patrones de demanda.
- **Estacionalidad**: Considera diferentes temporadas (alta, media, baja) para cada producto y tienda a lo largo del año.

## Estructura del proyecto

- `main.py`: Punto de entrada principal que coordina el entrenamiento y evaluación del agente.
- `entorno_precio.py`: Implementa el entorno Gym para el problema de optimización de precios.
- `demanda.py`: Contiene las funciones para simular la demanda basada en precios y parámetros.
- `optimizador.py`: Implementa los algoritmos de optimización con Gurobi.
- `parametros/`: Directorio con archivos CSV que contienen los parámetros del sistema:
  - `Parametros_Base.csv`: Costos unitarios, costos fijos y capacidades.
  - `parametros_distribucion_tienda_1.csv` y `parametros_distribuciones_tienda_2.csv`: Parámetros de demanda.
  - `asociacion_semana_temporada.csv`: Mapeo entre semanas y tipos de temporada.

## Flujo de trabajo

1. **Entrenamiento del agente DRL**:
   - El agente aprende a fijar precios óptimos para maximizar la utilidad esperada.
   - Para cada acción (precios), se generan múltiples escenarios de demanda.
   - Se optimiza cada escenario y se devuelve la utilidad promedio como recompensa.

2. **Evaluación del agente**:
   - Se evalúa el agente entrenado en múltiples episodios.
   - Se generan visualizaciones y métricas de rendimiento.

3. **Visualización de resultados**:
   - Gráficos de utilidad total y por semana.
   - Mapas de calor de precios.
   - Comparación de demanda vs pedidos.
   - Política de precios óptima en formato CSV.

## Enfoque SAA (Sample Average Approximation)

El enfoque SAA permite manejar la incertidumbre en la demanda:

1. Para cada acción (vector de precios), se generan N escenarios de demanda.
2. Se resuelve el problema de optimización para cada escenario.
3. La recompensa es el promedio de las utilidades obtenidas en los N escenarios.

Este enfoque proporciona políticas robustas que funcionan bien en diferentes realizaciones de la demanda.

## Requisitos

- Python 3.6+
- Gurobi Optimizer
- PyTorch
- Stable Baselines3
- NumPy, Pandas, Matplotlib, Seaborn

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python main.py
```

## Resultados

El sistema genera:
- Modelo entrenado (`modelo_precios_anual.zip`)
- Checkpoints durante el entrenamiento (`checkpoints/`)
- Visualizaciones de resultados (`graficos/`)
- Política de precios óptima (`politica_precios.csv`)

## Implementación de SAA

La implementación de SAA se encuentra principalmente en:
- `entorno_precio.py`: Método `step()` que genera múltiples escenarios.
- `optimizador.py`: Función `resolver_multiples_escenarios()` que optimiza cada escenario.

## Notas

- El sistema está diseñado para ser flexible y adaptable a diferentes contextos de retail.
- Los parámetros como el número de escenarios, semanas y productos se pueden ajustar según las necesidades.
- El sistema imprime información detallada durante la ejecución para facilitar el seguimiento del proceso. 
