import numpy as np
import os
import time
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from entorno_precio import PrecioOptEnv 
from utils import guardar_precios_optimos_csv # Ahora guardaremos un CSV más completo
from optimizador import calcular_resultados_optimizacion, load_static_params_once, set_global_eval_seed

class StopTrainingOnWallTime(BaseCallback):
    # ... (sin cambios) ...
    def __init__(self, max_wall_time_seconds, verbose=0):
        super(StopTrainingOnWallTime, self).__init__(verbose)
        self.max_wall_time_seconds = max_wall_time_seconds
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        if self.verbose > 0:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            print(f"[{timestamp} StopTrainingOnWallTime] Iniciando contador de tiempo (max {self.max_wall_time_seconds}s).")

    def _on_step(self) -> bool:
        if time.time() - self.start_time > self.max_wall_time_seconds:
            if self.verbose > 0:
                timestamp = time.strftime("%H:%M:%S", time.localtime())
                print(f"[{timestamp} StopTrainingOnWallTime] Parando entrenamiento - Se alcanzó el tiempo máximo ({self.max_wall_time_seconds / 60:.1f} min).")
            return False
        return True

def train_drl_for_week(optimizer_caller_for_env, ruta_datos_gym, n_productos, n_tiendas, drl_params):
    """
    Entrena un agente DRL para una semana específica usando el optimizer_caller proporcionado.
    optimizer_caller_for_env es una función que toma precios_np y devuelve utilidad_horizonte.
    """
    print(f"  Entrenando DRL para la semana actual...")

    def make_env_local(is_eval=False):
        # El optimizer_caller ya está "preparado" con el inventario y semana correctos
        env = PrecioOptEnv(optimizer_callable=optimizer_caller_for_env, 
                           ruta_datos_gym=ruta_datos_gym, 
                           n_productos=n_productos, n_tiendas=n_tiendas)
        env = Monitor(env)
        return env

    vec_env_drl = DummyVecEnv([lambda: make_env_local(is_eval=False)])
    eval_env_drl = DummyVecEnv([lambda: make_env_local(is_eval=True)])

    # Usar el mismo logger para todas las semanas o crear uno nuevo? Por ahora, el mismo.
    # new_logger_drl = configure(f"./sb3_logs/semana_{drl_params['semana_actual_año']}/", ["stdout", "csv", "tensorboard"])


    stop_train_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=drl_params["patience"], verbose=1)
    
    eval_cb = EvalCallback(
        eval_env_drl, best_model_save_path=f'./logs_best_model/semana_{drl_params["semana_actual_año"]}/',
        log_path=f'./logs_eval/semana_{drl_params["semana_actual_año"]}/', eval_freq=drl_params["eval_freq"],
        n_eval_episodes=drl_params["n_eval_episodes"], deterministic=True, render=False,
        callback_after_eval=stop_train_cb)
    
    wall_time_cb = StopTrainingOnWallTime(max_wall_time_seconds=drl_params["max_wall_time_per_week"], verbose=1)

    model_drl = PPO(
        "MlpPolicy", vec_env_drl, verbose=0, # verbose=1 para debug de PPO
        learning_rate=drl_params["lr"], n_steps=drl_params["n_steps_ppo"],
        batch_size=drl_params["batch_size_ppo"], n_epochs=4, gamma=0.99, 
        ent_coef=drl_params["ent_coef_ppo"]
    )
    model_drl.set_logger(drl_params["logger"]) # Usar el logger global o uno específico

    model_drl.learn(total_timesteps=drl_params["total_timesteps_week"], callback=[eval_cb, wall_time_cb], log_interval=10)

    # Cargar el mejor modelo si existe para esta semana
    best_model_week_path = os.path.join(f'./logs_best_model/semana_{drl_params["semana_actual_año"]}/', 'best_model.zip')
    if os.path.exists(best_model_week_path):
        print(f"  Cargando mejor modelo para semana {drl_params['semana_actual_año']} desde: {best_model_week_path}")
        final_model = PPO.load(best_model_week_path, env=vec_env_drl)
    else:
        print(f"  Usando modelo final del entrenamiento para semana {drl_params['semana_actual_año']}.")
        final_model = model_drl
    
    obs_drl = vec_env_drl.reset()
    precios_opt_semana_np, _ = final_model.predict(obs_drl, deterministic=True)
    return precios_opt_semana_np[0].reshape(n_productos, n_tiendas)


def main():
    RUTA_DATOS = "parametros"
    N_PRODUCTOS = 10
    N_TIENDAS = 2
    N_SEMANAS_PLANIFICACION = 4 # Planificar para las primeras 4 semanas del año

    # --- PARÁMETROS GLOBALES DE DRL (por semana) ---
    drl_params_config = {
        "total_timesteps_week": 100, # Timesteps DRL para optimizar precios de UNA semana
        "max_wall_time_per_week": 4 * 60, # Límite de tiempo para el DRL de UNA semana
        "eval_freq": 20,
        "n_eval_episodes": 5,
        "patience": 3, # Paciencia para StopTrainingOnNoModelImprovement
        "lr": 0.0003,
        "n_steps_ppo": 20,
        "batch_size_ppo": 10,
        "ent_coef_ppo": 0.005,
    }

    os.makedirs("./sb3_logs/", exist_ok=True)
    global_logger = configure("./sb3_logs/", ["stdout", "csv", "tensorboard"])
    print(f"Logging global to {os.path.abspath('./sb3_logs/')}")
    drl_params_config["logger"] = global_logger # Pasar el logger a la función de entrenamiento

    # Cargar parámetros estáticos del optimizador una vez
    static_opt_params = load_static_params_once(RUTA_DATOS, N_PRODUCTOS, N_TIENDAS)
    current_inventory_for_opt = static_opt_params["I_initial_global"] # Comienza con el global

    # Lista para almacenar todos los resultados detallados
    all_weeks_data = []

    # --- Bucle Principal por Semanas de Planificación ---
    for semana_idx_año in range(1, N_SEMANAS_PLANIFICACION + 1):
        timestamp_main = time.strftime("%H:%M:%S", time.localtime())
        print(f"\n[{timestamp_main} Main] === Optimizando Precios para Semana del Año: {semana_idx_año} ===")
        
        drl_params_config["semana_actual_año"] = semana_idx_año # Para nombres de archivo de logs

        # Crear el callable para el optimizador que el entorno usará.
        # Este callable ya "conoce" el inventario actual y la semana del año.
        # El DRL solo le pasará los precios_propuestos_np.
        # Esta función debe devolver solo la utilidad del horizonte para el DRL.
        def optimizer_caller_for_drl_env(precios_np_semana_actual, use_eval_seed_flag=False):
            # `use_eval_seed_flag` se podría pasar desde el entorno si diferenciamos train/eval calls
            # Por ahora, el DRL no diferencia, así que es siempre False para la semilla de escenarios.
            # Si se usa un eval_env para el DRL y se quiere semilla fija, hay que propagar el seed.
            # Aquí, `use_eval_seed_flag` se refiere a si la llamada viene de una evaluación DRL.
            # Para la semilla de escenarios del optimizador, se puede usar GLOBAL_EVAL_SEED si se setea.
            # set_global_eval_seed(drl_params_config["logger"].get_dir() ... ) # Algo complejo

            resultados = calcular_resultados_optimizacion(
                precios_np_semana_actual,
                current_inventory_for_opt,
                semana_idx_año,
                RUTA_DATOS,
                N_PRODUCTOS, N_TIENDAS,
                use_eval_seed=False # O manejar esto si el entorno DRL puede indicar si es evaluación
            )
            return resultados["utilidad_total_horizonte"]

        # Entrenar DRL para encontrar la mejor política de precios para `semana_idx_año`
        # dados los `current_inventory_for_opt`
        precios_optimos_semana_actual_np = train_drl_for_week(
            optimizer_caller_for_drl_env, 
            RUTA_DATOS, N_PRODUCTOS, N_TIENDAS, 
            drl_params_config
        )
        
        print(f"  Precios óptimos DRL para Semana {semana_idx_año} (P0T0): {precios_optimos_semana_actual_np[0,0]:.2f}")

        # Una vez obtenidos los precios óptimos para la semana_idx_año,
        # ejecutar el optimizador UNA VEZ MÁS con esos precios para obtener todos los detalles
        # (pedidos, demanda, shortage, inventario final) para ESA semana_idx_año.
        # Usar use_eval_seed=True para consistencia si se quiere una "evaluación final" de la política.
        # set_global_eval_seed(12345 + semana_idx_año) # Semilla fija para esta ejecución final de la semana
        
        print(f"  Calculando resultados detallados para Semana {semana_idx_año} con precios DRL óptimos...")
        resultados_detallados_semana = calcular_resultados_optimizacion(
            precios_optimos_semana_actual_np,
            current_inventory_for_opt,
            semana_idx_año,
            RUTA_DATOS,
            N_PRODUCTOS, N_TIENDAS,
            use_eval_seed=True # Usar semilla fija para obtener resultados consistentes de esta política
        )

        # Recopilar datos para el CSV
        for q in range(N_PRODUCTOS):
            for l in range(N_TIENDAS):
                all_weeks_data.append({
                    "semana_año": semana_idx_año,
                    "producto_idx": q,
                    "tienda_idx": l,
                    "precio_optimo": precios_optimos_semana_actual_np[q, l],
                    "pedido_optimo_sem1_horizonte": resultados_detallados_semana["pedidos_semana1"].get((q,l), 0),
                    "demanda_promedio_sem1_horizonte": resultados_detallados_semana["demanda_promedio_semana1"].get((q,l), 0),
                    "shortage_promedio_sem1_horizonte": resultados_detallados_semana["shortage_promedio_semana1"].get((q,l), 0),
                    "inventario_inicial_sem1_horizonte": current_inventory_for_opt.get((q,l),0), # El que se usó para esta optimización
                    "inventario_final_sem1_horizonte": resultados_detallados_semana["inventario_final_semana1"].get((q,l), 0)
                })
        
        # Actualizar el inventario para la siguiente semana
        current_inventory_for_opt = resultados_detallados_semana["inventario_final_semana1"]
        print(f"  Inventario final para P0T0 de Semana {semana_idx_año} (será inicial de S{semana_idx_año+1}): {current_inventory_for_opt.get((0,0),0):.2f}")

    # --- Guardar Resultados Compilados ---
    df_resultados_completos = pd.DataFrame(all_weeks_data)
    ruta_salida_csv = os.path.join("resultados", "Planificacion_Semanal_Optima_DRL.csv")
    os.makedirs("resultados", exist_ok=True)
    df_resultados_completos.to_csv(ruta_salida_csv, index=False, sep=';', decimal=',')
    print(f"\n¡Planificación secuencial completada para {N_SEMANAS_PLANIFICACION} semanas!")
    print(f"Resultados detallados guardados en: {ruta_salida_csv}")
    print("Para ver logs de TensorBoard: tensorboard --logdir ./sb3_logs/")

if __name__ == "__main__":
    if not os.path.exists("parametros"):
        print("ADVERTENCIA: La carpeta 'parametros' no existe.")
    main()