# Caso Base – Proyecto de Precios e Inventarios en Retail

Este proyecto implementa una política base para la gestión conjunta de precios e inventarios en un entorno de retail con 2 tiendas y 10 productos, utilizando datos históricos entre 2021 y 2025.

## 📋 Descripción del Enfoque

Se utiliza una política de **Base Stock con Precios Promedio Fijos**:
- Para cada producto en cada tienda, se define un **nivel de stock objetivo** equivalente al percentil 90 de la demanda histórica.
- El **precio de venta** se fija como el promedio histórico observado.
- Cada semana se reponen las unidades necesarias para alcanzar el stock objetivo.
- Se calculan los principales KPI's asociados a esta política para comparación con el modelo optimizado.

## 📁 Archivos principales

- `casobase.py`: Script principal para ejecutar la simulación del caso base y calcular KPIs.
- `Datos v1 copy.xlsx`: Excel con los datos históricos de demanda y precios por tienda.
- `KPIs_Caso_Base.xlsx`: Archivo generado automáticamente con el resumen de resultados por tienda.

## 📊 Indicadores Clave (KPI's) calculados

- **Utilidad Total**
- **Demanda Total y Satisfecha**
- **Demanda Insatisfecha**
- **Nivel de Servicio (%)**
- **Días Promedio en Inventario**
- **Órdenes Emitidas**
- **Costos: Inventario, Ordenamiento, Demanda Insatisfecha, Transporte**

### 📐 Supuestos del modelo

Este caso base se construye sobre los siguientes supuestos, basados en la información entregada en el problema:

#### 📦 Costos de productos
- Costo mayorista de cada producto según tabla entregada (en millones de pesos).
- Costo fijo por orden de cada producto también detallado por unidad.

#### 🧾 Costos adicionales
- **Costo de inventario**: 10% del costo mayorista del producto por unidad en stock.
- **Costo de demanda insatisfecha**: 10% del precio de venta promedio por unidad no atendida.
- **Valor residual del producto**: 50% del costo de ordenamiento (no considerado directamente en esta simulación base).

#### 🚚 Logística
- **Costo de transporte entre tiendas**: 3.8 M$/unidad  
  *(no incluido en el caso base porque no hay transferencias entre tiendas)*.
- **Capacidad de almacenamiento**:
  - Tienda 1: 175.000 unidades.
  - Tienda 2: 163.000 unidades.

#### 💸 Precios
- Precio mínimo de venta por producto debe garantizar un margen de al menos 5% sobre el costo.

#### ⚖️ Restricción de arbitraje
- La diferencia de precios entre tiendas no puede superar el costo de transporte (3.8 M$/unidad).

#### 📏 Restricciones de pedido
- Si se realiza un pedido de un producto, debe cumplir una cantidad mínima:

| Producto | Cant. Mín. Orden |
|----------|------------------|
| 1        | 300              |
| 2        | 700              |
| 3        | 500              |
| 4        | 260              |
| 5        | 900              |
| 6        | 350              |
| 7        | 1000             |
| 8        | 450              |
| 9        | 450              |
| 10       | 900              |


## 🚀 Cómo ejecutar

1. Instalar dependencias (una sola vez):

```bash
pip install pandas numpy openpyxl
```
2. Ejecutar Script
```bash
python casobase.py
```
3. Revisar el archivo generado en la misma carpeta.