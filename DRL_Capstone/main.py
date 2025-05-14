import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from entorno_precio import PrecioPorPeriodoEnv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def entrenar_agente(n_semanas=52, total_timesteps=100000, n_escenarios=10, ruta_datos='parametros'):
    """
    Entrena el agente DRL.
    
    Args:
        n_semanas (int): Número de semanas en el horizonte
        total_timesteps (int): Número total de pasos de entrenamiento
        n_escenarios (int): Número de escenarios para SAA
        ruta_datos (str, optional): Ruta al directorio de datos
    
    Returns:
        modelo entrenado
    """
    # Crear entorno con horizonte de múltiples semanas
    env = PrecioPorPeriodoEnv(n_semanas=n_semanas, n_escenarios=n_escenarios, ruta_datos=ruta_datos)
    env = DummyVecEnv([lambda: env])
    
    # Crear directorio para checkpoints
    os.makedirs("./checkpoints/", exist_ok=True)
    
    # Crear callback para guardar checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="modelo_precios"
    )
    
    # Crear y entrenar modelo
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )
    
    # Entrenar
    print(f"Iniciando entrenamiento con {n_semanas} semanas y {n_escenarios} escenarios por paso...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    return model

def evaluar_agente(model, n_episodios=5, n_semanas=52, n_escenarios=10, ruta_datos='parametros'):
    """
    Evalúa el agente entrenado.
    
    Args:
        model: Modelo entrenado
        n_episodios (int): Número de episodios de evaluación
        n_semanas (int): Número de semanas en el horizonte
        n_escenarios (int): Número de escenarios para SAA
        ruta_datos (str, optional): Ruta al directorio de datos
    
    Returns:
        dict: Métricas de evaluación
    """
    env = PrecioPorPeriodoEnv(n_semanas=n_semanas, n_escenarios=n_escenarios, ruta_datos=ruta_datos)
    
    # Resultados por episodio
    resultados_episodios = []
    
    for episodio in range(n_episodios):
        print(f"\nEvaluando episodio {episodio+1}/{n_episodios}")
        obs = env.reset()
        done = False
        
        # Historial para este episodio
        historial_acciones = []
        historial_recompensas = []
        historial_info = []
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            
            historial_acciones.append(action)
            historial_recompensas.append(reward)
            historial_info.append(info)
        
        # Al final del episodio, guardar resultados completos
        resultados_episodios.append({
            'acciones': historial_acciones,
            'recompensas': historial_recompensas,
            'info': historial_info[-1]  # El último info contiene el historial completo
        })
    
    return resultados_episodios

def visualizar_resultados(resultados_episodios, n_semanas=52):
    """
    Visualiza los resultados de la evaluación.
    
    Args:
        resultados_episodios (list): Lista de resultados por episodio
        n_semanas (int): Número de semanas en el horizonte
    """
    # Crear directorio para gráficos
    import os
    os.makedirs("graficos", exist_ok=True)
    
    # Extraer utilidad total por episodio
    utilidades_totales = [res['info'].get('utilidad_total', 0) for res in resultados_episodios]
    
    # 1. Gráfico de utilidad total por episodio
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(utilidades_totales)), utilidades_totales)
    plt.title('Utilidad Total por Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Utilidad Total')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graficos/utilidad_total_episodios.png')
    plt.close()
    
    # Seleccionar el mejor episodio para análisis detallado
    if utilidades_totales:
        mejor_episodio = np.argmax(utilidades_totales)
        mejor_resultado = resultados_episodios[mejor_episodio]
        
        # Verificar si hay historial de utilidades
        if 'historial_utilidades' in mejor_resultado['info']:
            # 2. Gráfico de utilidad por semana para el mejor episodio
            utilidades_semanales = mejor_resultado['info']['historial_utilidades']
            plt.figure(figsize=(12, 6))
            plt.plot(range(n_semanas), utilidades_semanales)
            plt.title(f'Utilidad por Semana (Mejor Episodio: {mejor_episodio+1})')
            plt.xlabel('Semana')
            plt.ylabel('Utilidad')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('graficos/utilidad_semanal.png')
            plt.close()
            
            # 3. Precio promedio por semana para el mejor episodio
            if 'historial_precios' in mejor_resultado['info']:
                precios_semanales = mejor_resultado['info']['historial_precios']
                precios_promedio = np.mean(precios_semanales, axis=(1, 2))  # Promedio por semana
                
                plt.figure(figsize=(12, 6))
                plt.plot(range(n_semanas), precios_promedio)
                plt.title(f'Precio Promedio por Semana (Mejor Episodio: {mejor_episodio+1})')
                plt.xlabel('Semana')
                plt.ylabel('Precio Promedio')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('graficos/precio_promedio_semanal.png')
                plt.close()
                
                # 4. Mapa de calor de precios promedio por producto y semana
                precios_producto = np.mean(precios_semanales, axis=2)  # Promedio por producto y semana
                
                plt.figure(figsize=(15, 8))
                sns.heatmap(precios_producto.T, cmap='viridis', annot=False)
                plt.title(f'Precios por Producto y Semana (Mejor Episodio: {mejor_episodio+1})')
                plt.xlabel('Semana')
                plt.ylabel('Producto')
                plt.tight_layout()
                plt.savefig('graficos/mapa_calor_precios.png')
                plt.close()
            
            # 5. Demanda vs Pedidos para el mejor episodio
            if 'historial_demandas' in mejor_resultado['info'] and 'historial_pedidos' in mejor_resultado['info']:
                demandas = mejor_resultado['info']['historial_demandas']
                pedidos = mejor_resultado['info']['historial_pedidos']
                
                # Promediar sobre todas las semanas
                demanda_promedio = np.mean(demandas, axis=0).flatten()
                pedidos_promedio = np.mean(pedidos, axis=0).flatten()
                
                plt.figure(figsize=(10, 8))
                plt.scatter(demanda_promedio, pedidos_promedio, alpha=0.7)
                
                # Línea de referencia donde demanda = pedidos
                max_val = max(demanda_promedio.max(), pedidos_promedio.max())
                plt.plot([0, max_val], [0, max_val], 'r--')
                
                plt.title(f'Demanda vs Pedidos (Mejor Episodio: {mejor_episodio+1})')
                plt.xlabel('Demanda Promedio')
                plt.ylabel('Pedidos Promedio')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('graficos/demanda_vs_pedidos.png')
                plt.close()
                
                # 6. Guardar política de precios en CSV
                if 'historial_precios' in mejor_resultado['info']:
                    precios_df = pd.DataFrame(precios_semanales.reshape(n_semanas, -1))
                    precios_df.columns = [f'Producto{i+1}_Tienda{j+1}' for i in range(9) for j in range(2)]
                    precios_df.index.name = 'Semana'
                    precios_df.to_csv('politica_precios.csv')
                
                # 7. Resumen de métricas
                print("\nResumen del mejor episodio ({}):".format(mejor_episodio+1))
                print(f"Utilidad total: {utilidades_totales[mejor_episodio]:.2f}")
                print(f"Utilidad promedio por semana: {np.mean(utilidades_semanales):.2f}")
                print(f"Precio promedio: {np.mean(precios_semanales):.2f}")
                print(f"Demanda promedio: {np.mean(demandas):.2f}")
                print(f"Pedidos promedio: {np.mean(pedidos):.2f}")

def main():
    # Parámetros
    n_semanas = 52  # Horizonte de un año
    n_escenarios = 10  # Número de escenarios para SAA
    ruta_datos = 'parametros'  # Ruta a la carpeta de parámetros
    
    # Entrenar agente
    print(f"Entrenando agente para un horizonte de {n_semanas} semanas con {n_escenarios} escenarios por paso...")
    model = entrenar_agente(n_semanas=n_semanas, total_timesteps=100000, 
                           n_escenarios=n_escenarios, ruta_datos=ruta_datos)
    
    # Guardar modelo
    model.save("modelo_precios_anual")
    
    # Evaluar agente
    print("\nEvaluando agente...")
    resultados = evaluar_agente(model, n_episodios=5, n_semanas=n_semanas, 
                              n_escenarios=n_escenarios, ruta_datos=ruta_datos)
    
    # Visualizar resultados
    print("\nVisualizando resultados...")
    try:
        visualizar_resultados(resultados, n_semanas=n_semanas)
    except Exception as e:
        print(f"Error al visualizar resultados: {e}")
        # Imprimir resultados directamente
        for i, res in enumerate(resultados):
            print(f"Episodio {i + 1}:")
            print(f"  Acciones: {res['acciones']}")
            print(f"  Recompensas: {res['recompensas']}")
            print(f"  Info: {res['info']}")
    
    print("\n¡Proceso completado!")
    print("Archivos generados:")
    print("- modelo_precios_anual.zip (modelo guardado)")
    print("- checkpoints/ (checkpoints durante entrenamiento)")
    print("- graficos/ (visualizaciones de resultados)")
    print("- politica_precios.csv (política de precios óptima)")

if __name__ == "__main__":
    main() 