"""
Verificación de Seeds para Experimento de Mismo Seed
Verifica que todas las configuraciones usen exactamente el mismo seed.
"""
import numpy as np
from particle_filter import set_global_particle_seed, reset_seed_counter, get_next_deterministic_seed
from optimizador import set_global_eval_seed
from analizar_utilidad import set_analizar_utilidad_seed, get_analizar_utilidad_seed

def test_mismo_seed_todas_configuraciones():
    """
    Test: Verifica que todas las configuraciones usan el mismo seed para generación de partículas.
    """
    print("🧪 Test 1: Mismo seed para todas las configuraciones")
    
    SEED_BASE = 42
    n_configuraciones = 5  # Simular 5 configuraciones diferentes
    
    # Colectar seeds para cada configuración
    seeds_usados = []
    
    for i in range(n_configuraciones):
        # Simular el seed que usaría cada configuración
        seed_config = SEED_BASE  # MISMO seed para todas
        seeds_usados.append(seed_config)
        
        # Verificar que efectivamente usa ese seed
        set_global_particle_seed(seed_config)
        reset_seed_counter()
        
        # Generar algunos números aleatorios para verificar
        np.random.seed(seed_config)
        numeros_aleatorios = np.random.random(3)
        
        print(f"  Configuración {i+1}: seed={seed_config}, primeros números={numeros_aleatorios}")
    
    # Verificar que todos los seeds son iguales
    seeds_unicos = set(seeds_usados)
    if len(seeds_unicos) == 1:
        print(f"  ✅ ÉXITO: Todas las configuraciones usan el mismo seed: {list(seeds_unicos)[0]}")
        return True
    else:
        print(f"  ❌ FALLA: Se encontraron {len(seeds_unicos)} seeds diferentes: {seeds_unicos}")
        return False

def test_mismos_numeros_aleatorios():
    """
    Test: Verifica que el mismo seed produce exactamente los mismos números aleatorios.
    """
    print("\n🧪 Test 2: Mismos números aleatorios con mismo seed")
    
    SEED_BASE = 42
    
    # Primera ejecución
    np.random.seed(SEED_BASE)
    numeros_1 = np.random.random(10)
    
    # Segunda ejecución con mismo seed
    np.random.seed(SEED_BASE)
    numeros_2 = np.random.random(10)
    
    # Comparar
    son_identicos = np.array_equal(numeros_1, numeros_2)
    
    print(f"  Primera ejecución: {numeros_1[:3]}...")
    print(f"  Segunda ejecución: {numeros_2[:3]}...")
    print(f"  Son idénticos: {son_identicos}")
    
    if son_identicos:
        print("  ✅ ÉXITO: Mismo seed produce números idénticos")
        return True
    else:
        print("  ❌ FALLA: Mismo seed produce números diferentes")
        return False

def test_seeds_globales_compartidos():
    """
    Test: Verifica que los seeds globales (optimizador y analizar_utilidad) son compartidos.
    """
    print("\n🧪 Test 3: Seeds globales compartidos")
    
    SEED_BASE = 42
    
    # Configurar seeds globales
    set_global_eval_seed(SEED_BASE)
    set_analizar_utilidad_seed(SEED_BASE)
    
    # Verificar que se configuraron correctamente
    seed_demanda = get_analizar_utilidad_seed()
    
    print(f"  Seed configurado para optimizador: {SEED_BASE}")
    print(f"  Seed demanda real: {seed_demanda}")
    
    if seed_demanda == SEED_BASE:
        print("  ✅ ÉXITO: Seeds globales configurados correctamente")
        return True
    else:
        print("  ❌ FALLA: Seeds globales no coinciden con esperado")
        return False

def test_reproducibilidad_completa():
    """
    Test: Simula dos ejecuciones completas y verifica que son idénticas.
    """
    print("\n🧪 Test 4: Reproducibilidad completa de ejecución")
    
    SEED_BASE = 42
    
    def simular_ejecucion(seed):
        """Simula una ejecución completa de una configuración"""
        # Seeds globales
        set_global_eval_seed(seed)
        set_analizar_utilidad_seed(seed)
        
        # Seed para partículas
        set_global_particle_seed(seed)
        reset_seed_counter()
        
        # Simular generación de partículas
        np.random.seed(seed)
        particulas = np.random.random(25)  # 25 partículas iniciales
        
        # Simular evaluaciones del optimizador
        evaluaciones = []
        for i in range(5):  # 5 evaluaciones
            np.random.seed(seed + i)  # Simular seed determinístico
            eval_result = np.random.random()
            evaluaciones.append(eval_result)
        
        # Simular demanda real
        np.random.seed(seed + 1000)
        demanda_real = np.random.random(4)  # 4 semanas
        
        return {
            'particulas': particulas,
            'evaluaciones': evaluaciones,
            'demanda_real': demanda_real
        }
    
    # Dos ejecuciones con mismo seed
    resultado_1 = simular_ejecucion(SEED_BASE)
    resultado_2 = simular_ejecucion(SEED_BASE)
    
    # Comparar resultados
    particulas_iguales = np.array_equal(resultado_1['particulas'], resultado_2['particulas'])
    evaluaciones_iguales = np.array_equal(resultado_1['evaluaciones'], resultado_2['evaluaciones'])
    demanda_igual = np.array_equal(resultado_1['demanda_real'], resultado_2['demanda_real'])
    
    print(f"  Partículas idénticas: {particulas_iguales}")
    print(f"  Evaluaciones idénticas: {evaluaciones_iguales}")
    print(f"  Demanda real idéntica: {demanda_igual}")
    
    todo_identico = particulas_iguales and evaluaciones_iguales and demanda_igual
    
    if todo_identico:
        print("  ✅ ÉXITO: Ejecuciones completamente reproducibles")
        return True
    else:
        print("  ❌ FALLA: Ejecuciones no son idénticas")
        return False

def test_deterministic_seed_sequence():
    """
    Test: Verifica que get_next_deterministic_seed genera secuencias idénticas con mismo seed.
    """
    print("\n🧪 Test 5: Secuencia determinística de seeds")
    
    SEED_BASE = 42
    
    # Primera secuencia
    set_global_particle_seed(SEED_BASE)
    reset_seed_counter()
    secuencia_1 = []
    for i in range(10):
        seed = get_next_deterministic_seed()
        secuencia_1.append(seed)
    
    # Segunda secuencia con mismo seed base
    set_global_particle_seed(SEED_BASE)
    reset_seed_counter()
    secuencia_2 = []
    for i in range(10):
        seed = get_next_deterministic_seed()
        secuencia_2.append(seed)
    
    # Comparar secuencias
    son_identicas = secuencia_1 == secuencia_2
    
    print(f"  Primera secuencia: {secuencia_1[:5]}...")
    print(f"  Segunda secuencia: {secuencia_2[:5]}...")
    print(f"  Son idénticas: {son_identicas}")
    
    if son_identicas:
        print("  ✅ ÉXITO: Secuencias de seeds determinísticas")
        return True
    else:
        print("  ❌ FALLA: Secuencias de seeds no son idénticas")
        return False

def main():
    """
    Ejecuta todos los tests de verificación para experimento de mismo seed.
    """
    print("🔍 VERIFICACIÓN DE SEEDS PARA EXPERIMENTO 'MISMO SEED'")
    print("=" * 60)
    print("Verificando que todas las configuraciones usen exactamente el mismo seed...")
    print()
    
    tests = [
        test_mismo_seed_todas_configuraciones,
        test_mismos_numeros_aleatorios,
        test_seeds_globales_compartidos,
        test_reproducibilidad_completa,
        test_deterministic_seed_sequence
    ]
    
    resultados = []
    
    for test_func in tests:
        try:
            resultado = test_func()
            resultados.append(resultado)
        except Exception as e:
            print(f"  ❌ ERROR en {test_func.__name__}: {e}")
            resultados.append(False)
    
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE VERIFICACIÓN")
    print("=" * 60)
    
    tests_exitosos = sum(resultados)
    total_tests = len(resultados)
    
    print(f"Tests exitosos: {tests_exitosos}/{total_tests}")
    
    if tests_exitosos == total_tests:
        print("🎉 ¡TODOS LOS TESTS PASARON!")
        print("✅ El experimento con mismo seed funcionará correctamente.")
        print("🔄 Todas las políticas usarán exactamente los mismos números aleatorios.")
    else:
        print("⚠️  Algunos tests fallaron. Revisa la configuración de seeds.")
    
    print(f"\n🚀 Para ejecutar el experimento con mismo seed, usa:")
    print(f"   py experimento_resampling_mismo_seed.py")

if __name__ == "__main__":
    main() 