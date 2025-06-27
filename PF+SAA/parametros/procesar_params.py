import pandas as pd
import numpy as np
import argparse
import os

# --- 0. Definición de Funciones y Argumentos ---

def calcular_rho(mu, gamma, alpha, precio):
    """
    Calcula rho_qml a partir de la fórmula:
    μ_qml = γ_qml * ρ_qml * e^(-α_ql * P_qml)
    ρ_qml = μ_qml / (γ_qml * e^(-α_ql * P_qml))
    """
    denominador = gamma * np.exp(-alpha * precio)
    if denominador == 0 or np.isinf(denominador) or np.isnan(denominador):
        print(f"Advertencia: Denominador inválido (cero, inf o nan) al calcular rho para mu={mu}, gamma={gamma}, alpha={alpha}, precio={precio}")
        return np.nan
    rho = mu / denominador
    if np.isinf(rho) or np.isnan(rho):
        print(f"Advertencia: Rho resultante es inf o nan para mu={mu}, gamma={gamma}, alpha={alpha}, precio={precio}")
        return np.nan
    return rho

def calcular_theta(mu, varianza):
    """
    Calcula theta_qmlt a partir de la fórmula:
    σ^2_qmlt = μ_qmlt * (1 + μ_qmlt / θ_qmlt)
    Despejando theta:
    θ = μ^2 / (σ^2 - μ)
    """
    if varianza - mu == 0:
        print(f"Advertencia: Varianza ({varianza}) - mu ({mu}) es cero. Theta podría ser infinito o indefinido.")
        return np.nan 
    
    theta_calc = (mu**2) / (varianza - mu)

    if theta_calc <= 0 or np.isinf(theta_calc) or np.isnan(theta_calc):
        print(f"Advertencia: Theta calculado no es positivo o es inf/nan (mu={mu}, varianza={varianza}, theta_calc={theta_calc}).")
        return np.nan
    return theta_calc

def main(alpha_valor_arg): # Renombrado el argumento para evitar confusión con la columna
    # --- 1. Recibir y Procesar el Archivo parametros_maestro.csv ---
    archivo_maestro = "parametros_maestro.csv"
    archivo_salida_t1 = "par_dist_t1.csv"
    archivo_salida_t2 = "par_dist_t2.csv"

    print(f"Cargando datos de '{archivo_maestro}'...")
    try:
        df_maestro = pd.read_csv(archivo_maestro, sep=';', decimal=',')
        
        cols_numericas = ['mu_historico', 'precio_historico', 'gamma', 'varianza']
        for col in cols_numericas:
            if df_maestro[col].dtype == 'object':
                df_maestro[col] = df_maestro[col].str.replace(',', '.', regex=False).astype(float)
            elif not pd.api.types.is_numeric_dtype(df_maestro[col]):
                df_maestro[col] = pd.to_numeric(df_maestro[col], errors='coerce')
        
        df_maestro['producto'] = df_maestro['producto'].astype(str)
        df_maestro['tienda'] = df_maestro['tienda'].astype(str)

    except FileNotFoundError:
        print(f"ERROR: El archivo '{archivo_maestro}' no fue encontrado. Asegúrate de que esté en el mismo directorio que el script.")
        return
    except Exception as e:
        print(f"Error al cargar o procesar '{archivo_maestro}': {e}")
        return

    print("Datos cargados exitosamente. Primeras filas:")
    print(df_maestro.head())

    # --- 2. Aplicar el Alpha Argumento ---
    # El alpha es el mismo para todos los productos, tiendas y grupos en esta lógica
    df_maestro['alfa'] = alpha_valor_arg # Nombre de columna 'alfa'
    print(f"\nSe aplicará un valor de alfa = {alpha_valor_arg} a todos los registros.")

    # --- 3. Calcular Rho para cada fila ---
    print("\nCalculando rho para cada fila...")
    df_maestro['rho'] = df_maestro.apply(
        lambda row: calcular_rho(
            row['mu_historico'], 
            row['gamma'], 
            row['alfa'], # Usar la nueva columna 'alfa'
            row['precio_historico']
        ), 
        axis=1
    )

    # --- 4. Calcular Theta para cada fila ---
    print("\nCalculando theta para cada fila...")
    df_maestro['theta'] = df_maestro.apply(
        lambda row: calcular_theta(
            row['mu_historico'], 
            row['varianza']
        ), 
        axis=1
    )
    
    if df_maestro['rho'].isnull().any():
        print("\nAdvertencia: Se generaron valores NaN para 'rho'. Revisa los datos de entrada y los cálculos.")
    if df_maestro['theta'].isnull().any():
        print("\nAdvertencia: Se generaron valores NaN para 'theta'. Revisa los datos de entrada y los cálculos.")

    print("\nTabla con alfa, rho y theta calculados (primeras filas):")
    print(df_maestro[['producto', 'tienda', 'grupo', 'alfa', 'rho', 'theta']].head())

    # --- 5. Sobrescribir los archivos par_dist_tX.csv ---
    
    # Preparar datos para Tienda 1
    df_t1 = df_maestro[df_maestro['tienda'] == '1'].copy()
    if not df_t1.empty:
        df_t1['producto_grupo'] = "producto_" + df_t1['producto'] 
        # Seleccionar y renombrar columnas para el output
        df_output_t1 = df_t1[['producto_grupo', 'grupo', 'alfa', 'gamma', 'rho', 'theta']].copy()
        df_output_t1.rename(columns={'producto_grupo': 'producto'}, inplace=True) 
        
        df_output_t1['producto_sort_key'] = df_output_t1['producto'].str.split('_').str[1].astype(int)
        df_output_t1 = df_output_t1.sort_values(by=['producto_sort_key', 'grupo']).drop(columns=['producto_sort_key'])
        
        print(f"\nGuardando datos procesados en '{archivo_salida_t1}'...")
        try:
            df_output_t1.to_csv(archivo_salida_t1, sep=';', decimal=',', index=False, float_format='%.8f')
            print(f"Archivo '{archivo_salida_t1}' guardado exitosamente.")
        except Exception as e:
            print(f"Error al guardar '{archivo_salida_t1}': {e}")
    else:
        print(f"No hay datos para la tienda 1 en '{archivo_maestro}'. No se generará '{archivo_salida_t1}'.")

    # Preparar datos para Tienda 2
    df_t2 = df_maestro[df_maestro['tienda'] == '2'].copy()
    if not df_t2.empty:
        df_t2['producto_grupo'] = "producto_" + df_t2['producto']
        # Seleccionar y renombrar columnas para el output
        df_output_t2 = df_t2[['producto_grupo', 'grupo', 'alfa', 'gamma', 'rho', 'theta']].copy()
        df_output_t2.rename(columns={'producto_grupo': 'producto'}, inplace=True)

        df_output_t2['producto_sort_key'] = df_output_t2['producto'].str.split('_').str[1].astype(int)
        df_output_t2 = df_output_t2.sort_values(by=['producto_sort_key', 'grupo']).drop(columns=['producto_sort_key'])

        print(f"\nGuardando datos procesados en '{archivo_salida_t2}'...")
        try:
            df_output_t2.to_csv(archivo_salida_t2, sep=';', decimal=',', index=False, float_format='%.8f')
            print(f"Archivo '{archivo_salida_t2}' guardado exitosamente.")
        except Exception as e:
            print(f"Error al guardar '{archivo_salida_t2}': {e}")
    else:
        print(f"No hay datos para la tienda 2 en '{archivo_maestro}'. No se generará '{archivo_salida_t2}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcula rho y theta y actualiza archivos de parámetros de distribución.")
    # El argumento de línea de comandos se sigue llamando 'alpha' internamente para el parser,
    # pero la variable que se usa en main() y la columna del DataFrame se llama 'alfa'.
    parser.add_argument("alpha", type=float, help="Valor de alpha (sensibilidad al precio) a utilizar.") 
    
    args = parser.parse_args()
    
    main(args.alpha) # Pasamos el valor a la función main