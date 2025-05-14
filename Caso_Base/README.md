# Simulación de Política de Stock Base

Este proyecto contiene tres scripts en Python que simulan y evalúan el desempeño de una política de inventario tipo base stock en un entorno retail, utilizando un archivo de entrada `Datos v1.xlsx` con datos semanales de demanda y precio para 10 productos en dos tiendas.

## Archivos

### `casobaseUT.py` — **Caso Base con Uso Total de Datos**
- Entrena y simula utilizando todos los datos disponibles, sin separar entrenamiento/test.
- Calcula el stock base como el percentil 90 de la demanda histórica y usa el precio promedio observado.
- Simula la operación completa (demanda, órdenes, quiebres, inventario).
- Genera un archivo Excel `KPIs_Caso_Base_Completo.xlsx` con hojas que incluyen:
  - `Resumen KPIs` por tienda
  - `Stock y Precios` promedio
  - `Demanda`, `Quiebres` y `Órdenes` semana a semana para Tienda 1 y Tienda 2

### `casobasePP.py` — **Política Estática con Precio Promedio**
- Separa los datos hasta 2024 para calcular políticas (stock base y precios).
- Simula solamente el año 2025.
- Utiliza precio promedio por producto y repone siempre hasta el stock base.
- Genera el archivo `KPIs_2025_BasePolicy_Completo.xlsx` con las mismas hojas relevantes que `casobaseUT.py`, pero limitadas a las semanas del año 2025.

### `casobasePD.py` — **Política con Precio Dinámico y Reposición Parcial**
- Separa los datos hasta 2024 para definir políticas y simula solo el año 2025.
- Ajusta dinámicamente los precios según el inventario:
  - Si el inventario está bajo el 80% del stock base → **sube el precio un 20%**
  - Si el inventario supera el 120% del stock base → **baja el precio un 20%**
  - Si el inventario está entre 80% y 120% del stock base → se mantiene el precio promedio
- Implementa una **política de reposición parcial escalonada**:
  - Inventario < 60% del stock base → reponer completo
  - Inventario entre 60% y 90% → reponer el 50% del faltante
  - Inventario > 90% → reponer el 20% del faltante
- Genera `KPIs_2025_Precios_Dinamicos.xlsx` incluyendo:
  - `Resumen KPIs`
  - `Stock y Precios`
  - Hojas adicionales con `Precios`, `Ingresos`, `Quiebres`, `Órdenes` y `Demanda` por semana

## Requisitos

- Python 3.8+
- Bibliotecas:
  ```bash
  pip install pandas numpy xlsxwriter
  ```