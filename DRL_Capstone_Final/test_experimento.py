"""
Script de verificación rápida para el experimento de resampling.
Ejecuta una configuración pequeña para verificar que todo funciona.
"""
import sys
import os

# Verificar imports críticos
try:
    from particle_filter import set_global_particle_seed, reset_seed_counter
    print("✅ particle_filter importado correctamente")
except ImportError as e:
    print(f"❌ Error importando particle_filter: {e}")
    sys.exit(1)

try:
    from resampling_policies import get_policy_function
    print("✅ resampling_policies importado correctamente")
except ImportError as e:
    print(f"❌ Error importando resampling_policies: {e}")
    sys.exit(1)

try:
    from analizar_utilidad import calcular_utilidad_total
    print("✅ analizar_utilidad importado correctamente")
except ImportError as e:
    print(f"❌ Error importando analizar_utilidad: {e}")
    sys.exit(1)

try:
    from optimizador import load_static_params_once, calcular_resultados_optimizacion
    print("✅ optimizador importado correctamente")
except ImportError as e:
    print(f"❌ Error importando optimizador: {e}")
    sys.exit(1)

# Verificar archivos de datos
RUTA_DATOS = "parametros"
archivos_requeridos = [
    "General_Parameters.csv",
    "Precios_Iniciales.csv", 
    "asociaciones_semana_dist.csv",
    "par_dist_t1.csv",
    "par_dist_t2.csv"
]

print(f"\n🔍 Verificando archivos en {RUTA_DATOS}/:")
for archivo in archivos_requeridos:
    ruta_completa = os.path.join(RUTA_DATOS, archivo)
    if os.path.exists(ruta_completa):
        print(f"✅ {archivo}")
    else:
        print(f"❌ {archivo} - NO ENCONTRADO")

# Test rápido de una política
print(f"\n🧪 EJECUTANDO TEST RÁPIDO...")
try:
    # Configurar seed
    set_global_particle_seed(42)
    reset_seed_counter()
    
    # Cargar parámetros
    static_params = load_static_params_once(RUTA_DATOS, 10, 2)
    print("✅ Parámetros estáticos cargados")
    
    # Test de función de política
    policy_func = get_policy_function("sin_resampling")
    print("✅ Función de política obtenida")
    
    # Test muy básico (solo 3 partículas para velocidad)
    def dummy_evaluate(precios):
        return float(sum(sum(row) for row in precios))
    
    print("🔄 Ejecutando test con 3 partículas...")
    resultado, score = policy_func(
        n_particles=3,
        n_productos=10,
        n_tiendas=2,
        precios_base=static_params["precios_base_np"],
        evaluate_fn=dummy_evaluate,
        semana_idx=1,
        costo_transporte=3.8
    )
    
    print(f"✅ Test completado. Score: {score:.2f}")
    print(f"✅ Resultado shape: {resultado.shape}")
    
except Exception as e:
    print(f"❌ Error en test rápido: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n🎉 VERIFICACIÓN COMPLETADA - TODO LISTO PARA EL EXPERIMENTO")
print(f"▶️  Ejecuta: python experimento_resampling.py") 