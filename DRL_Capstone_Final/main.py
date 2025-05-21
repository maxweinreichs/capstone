import numpy as np
import os
import time
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
# Intenta importar configure de la nueva manera si SB3 es muy reciente
# from stable_baselines3.common.logger import configure as sb3_configure

from entorno_precio import PrecioOptEnv 
from utils import guardar_precios_optimos_csv
from optimizador import calcular_resultados_optimizacion, load_static_params_once, set_global_eval_seed

class StopTrainingOnWallTime(BaseCallback):
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
    print(f"  Entrenando DRL para la semana actual (año {drl_params['semana_actual_año']})...")

    def make_env_local(is_eval=False):
        env_id_suffix = "eval" if is_eval else f"train_s{drl_params['semana_actual_año']}"
        # El callable ya tiene el inventario y semana correctos
        env = PrecioOptEnv(optimizer_callable=optimizer_caller_for_env, 
                           ruta_datos_gym=ruta_datos_gym, 
                           n_productos=n_productos, n_tiendas=n_tiendas)
        # Monitor env para logueo de recompensas de episodio, crucial para EvalCallback y TensorBoard
        log_dir_monitor = f"./sb3_logs/monitor_logs_s{drl_params['semana_actual_año']}_{env_id_suffix}/"
        os.makedirs(log_dir_monitor, exist_ok=True)
        env = Monitor(env, log_dir_monitor, allow_early_resets=True)
        return env

    print(f"    Creando DummyVecEnv...")
    vec_env_drl = DummyVecEnv([lambda: make_env_local(is_eval=False)])
    eval_env_drl = DummyVecEnv([lambda: make_env_local(is_eval=True)])
    print(f"    Entornos creados.")

    # Callbacks
    # EvalCallback necesita que Monitor envuelva el eval_env para funcionar correctamente
    # El path de best_model_save_path debe existir
    os.makedirs(f'./logs_best_model/semana_{drl_params["semana_actual_año"]}/', exist_ok=True)
    os.makedirs(f'./logs_eval/semana_{drl_params["semana_actual_año"]}/', exist_ok=True)

    stop_train_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=drl_params["patience"], verbose=1)
    
    eval_cb = EvalCallback(
        eval_env_drl, 
        best_model_save_path=f'./logs_best_model/semana_{drl_params["semana_actual_año"]}/',
        log_path=f'./logs_eval/semana_{drl_params["semana_actual_año"]}/', 
        eval_freq=drl_params["eval_freq"],
        n_eval_episodes=drl_params["n_eval_episodes"], 
        deterministic=True, # Usar política determinística para evaluación
        render=False,
        callback_on_new_best=stop_train_cb) # Usar callback_on_new_best en lugar de callback_after_eval para StopTrainingOnNoModelImprovement
    
    wall_time_cb = StopTrainingOnWallTime(max_wall_time_seconds=drl_params["max_wall_time_per_week"], verbose=1)
    print(f"    Callbacks creados.")

    print(f"    Creando modelo PPO con LR={drl_params['lr']}, ent_coef={drl_params['ent_coef_ppo']}...")
    model_drl = PPO(
        "MlpPolicy", vec_env_drl, 
        verbose=1, # verbose=1 para debug de PPO en consola
        vf_coef=drl_params["vf_coef"],
        learning_rate=drl_params["lr"], 
        n_steps=drl_params["n_steps_ppo"],
        batch_size=drl_params["batch_size_ppo"], 
        n_epochs=drl_params["n_epochs"], 
        gamma=0.99, 
        ent_coef=drl_params["ent_coef_ppo"]
        # tensorboard_log="./sb3_logs/ppo_tensorboard/" # SB3 PPO usa el logger seteado por set_logger
    )
    print(f"    Modelo PPO creado.")
    
    # Asegurarse que el logger está bien configurado
    # El logger global ya debería estar configurado para escribir en ./sb3_logs/
    # y debería incluir el formato "tensorboard"
    model_drl.set_logger(drl_params["logger"])
    print(f"    Logger seteado en el modelo. Log dir: {drl_params['logger'].get_dir()}")

    print(f"    Iniciando model_drl.learn() con total_timesteps={drl_params['total_timesteps_week']}, log_interval={drl_params['log_interval']}...")
    
    try:
        model_drl.learn(
            total_timesteps=drl_params["total_timesteps_week"], 
            callback=[eval_cb, wall_time_cb], # Lista de callbacks
            log_interval=drl_params["log_interval"] # Loguear cada N rollouts
        )
    except Exception as e:
        print(f"ERROR durante model_drl.learn(): {e}")
        import traceback
        traceback.print_exc()
        # Si hay un error, devolvemos precios base como fallback
        static_p = load_static_params_once(ruta_datos_gym, n_productos, n_tiendas)
        return static_p["precios_base_np"]

    print(f"    model_drl.learn() completado.")

    best_model_week_path = os.path.join(f'./logs_best_model/semana_{drl_params["semana_actual_año"]}/', 'best_model.zip')
    final_model_to_use = model_drl # Usar el modelo final del entrenamiento por defecto
    if os.path.exists(best_model_week_path):
        print(f"  Cargando mejor modelo para semana {drl_params['semana_actual_año']} desde: {best_model_week_path}")
        # Necesitas pasar el mismo env o uno compatible al cargar.
        # vec_env_drl ya está definido y es compatible.
        try:
            final_model_to_use = PPO.load(best_model_week_path, env=vec_env_drl)
        except Exception as e:
            print(f"ERROR al cargar best_model: {e}. Usando modelo final del entrenamiento.")
            final_model_to_use = model_drl
    else:
        print(f"  No se encontró best_model.zip. Usando modelo final del entrenamiento para semana {drl_params['semana_actual_año']}.")
    
    # Usar un nuevo entorno para la predicción final para evitar problemas de estado
    pred_env = DummyVecEnv([lambda: make_env_local(is_eval=True)]) # O False, no debería importar mucho para solo predict
    obs_drl = pred_env.reset()
    precios_opt_semana_np, _ = final_model_to_use.predict(obs_drl, deterministic=True)
    pred_env.close() # Cerrar el entorno de predicción
    
    vec_env_drl.close() # Cerrar el entorno de entrenamiento
    eval_env_drl.close() # Cerrar el entorno de evaluacion

    return precios_opt_semana_np[0].reshape(n_productos, n_tiendas)


def main():
    RUTA_DATOS = "parametros"
    N_PRODUCTOS = 10
    N_TIENDAS = 2
    N_SEMANAS_PLANIFICACION = 1 # PROBAR CON 1 SEMANA PRIMERO PARA DEBUG PROFUNDO

    # --- PARÁMETROS GLOBALES DE DRL (por semana) ---
    # Ajustados para un debug más rápido y forzar cambios si es posible
    drl_params_config = {
        "total_timesteps_week": 256 * 40, # ~20 rollouts/actualizaciones de PPO
        "max_wall_time_per_week": 60 * 60, # 20 minutos, debería ser suficiente
        "eval_freq": 256 * 8,        # Evaluar cada 5 rollouts
        "n_eval_episodes": 5,
        "patience": 10,              # Paciencia para StopTrainingOnNoModelImprovement
        "lr": 0.05,               # LR más estándar para empezar de nuevo
        "n_steps_ppo": 256,
        "batch_size_ppo": 64,
        "ent_coef_ppo": 0.05,       # Exploración moderada pero significativa
        "n_epochs": 10,             # Más épocas por batch
        "vf_coef": 0.5,             # Default SB3
        "log_interval": 1           # Loguear cada rollout
    }

    # Configuración del Logger (CRUCIAL)
    log_path = "./sb3_logs/"
    os.makedirs(log_path, exist_ok=True)
    
    # Intenta con sb3_configure si configure da problemas o viceversa
    # from stable_baselines3.common.logger import configure as sb3_configure
    # global_logger = sb3_configure(log_path, ["stdout", "csv", "tensorboard"])
    
    # Mantén tu forma original si funcionaba antes del problema de imghdr
    print(f"Limpiando logs antiguos en {os.path.abspath(log_path)}...")
    for root, dirs, files in os.walk(log_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.makedirs(log_path, exist_ok=True) # Recrear la carpeta base si se eliminó
    print("Logs antiguos limpiados.")
    global_logger = configure(log_path, ["stdout", "csv", "tensorboard"]) 
    
    print(f"Logger configurado. Verificando carpeta de logs: {os.path.abspath(log_path)}")
    if os.path.exists(log_path):
        print(f"  La carpeta {log_path} existe.")
        try:
            with open(os.path.join(log_path, "test_write.txt"), "w") as f:
                f.write("test")
            print(f"  Escritura de prueba en {log_path} exitosa.")
            os.remove(os.path.join(log_path, "test_write.txt"))
        except Exception as e:
            print(f"  ERROR: No se pudo escribir en {log_path}. Error: {e}")
    else:
        print(f"  ERROR: La carpeta {log_path} NO existe después de configure().")
    
    print(f"Logging global to {os.path.abspath(log_path)}")
    drl_params_config["logger"] = global_logger

    static_opt_params = load_static_params_once(RUTA_DATOS, N_PRODUCTOS, N_TIENDAS)
    current_inventory_for_opt = static_opt_params["I_initial_global"] 

    all_weeks_data = []

    for semana_idx_año in range(1, N_SEMANAS_PLANIFICACION + 1):
        timestamp_main = time.strftime("%H:%M:%S", time.localtime())
        print(f"\n[{timestamp_main} Main] === Optimizando Precios para Semana del Año: {semana_idx_año} ===")
        
        drl_params_config["semana_actual_año"] = semana_idx_año

        def optimizer_caller_for_drl_env(precios_np_semana_actual, use_eval_seed_flag=False):
            resultados = calcular_resultados_optimizacion(
                precios_np_semana_actual, current_inventory_for_opt,
                semana_idx_año, RUTA_DATOS, N_PRODUCTOS, N_TIENDAS,
                use_eval_seed=False 
            )
            return resultados["utilidad_total_horizonte"]

        precios_optimos_semana_actual_np = train_drl_for_week(
            optimizer_caller_for_drl_env, 
            RUTA_DATOS, N_PRODUCTOS, N_TIENDAS, drl_params_config
        )
        
        print(f"  Precios óptimos DRL para Semana {semana_idx_año} (P0T0): {precios_optimos_semana_actual_np[0,0]:.2f}")

        set_global_eval_seed(12345 + semana_idx_año) # Semilla fija para la ejecución final de la semana
        print(f"  Calculando resultados detallados para Semana {semana_idx_año} con precios DRL óptimos...")
        resultados_detallados_semana = calcular_resultados_optimizacion(
            precios_optimos_semana_actual_np, current_inventory_for_opt,
            semana_idx_año, RUTA_DATOS, N_PRODUCTOS, N_TIENDAS,
            use_eval_seed=True
        )

        for q in range(N_PRODUCTOS):
            for l in range(N_TIENDAS):
                all_weeks_data.append({
                    "semana_año": semana_idx_año, "producto_idx": q, "tienda_idx": l,
                    "precio_optimo": precios_optimos_semana_actual_np[q, l],
                    "pedido_optimo_sem1_horizonte": resultados_detallados_semana["pedidos_semana1"].get((q,l), 0),
                    "demanda_promedio_sem1_horizonte": resultados_detallados_semana["demanda_promedio_semana1"].get((q,l), 0),
                    "shortage_promedio_sem1_horizonte": resultados_detallados_semana["shortage_promedio_semana1"].get((q,l), 0),
                    "inventario_inicial_sem1_horizonte": current_inventory_for_opt.get((q,l),0),
                    "inventario_final_sem1_horizonte": resultados_detallados_semana["inventario_final_semana1"].get((q,l), 0)
                })
        
        current_inventory_for_opt = resultados_detallados_semana["inventario_final_semana1"]
        print(f"  Inventario final para P0T0 de Semana {semana_idx_año} (será inicial de S{semana_idx_año+1}): {current_inventory_for_opt.get((0,0),0):.2f}")

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