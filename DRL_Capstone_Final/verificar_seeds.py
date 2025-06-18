#!/usr/bin/env python3
"""
Script de verificación del manejo de seeds en el experimento de resampling.

Este script verifica que:
1. Las partículas se generen de forma determinística
2. Las evaluaciones del optimizador sean reproducibles
3. Las simulaciones de demanda real sean consistentes
4. Diferentes configuraciones con el mismo seed den resultados idénticos
"""

import numpy as np
import time
import os
import sys

# Agregar el directorio actual al path si es necesario
if '.' not in sys.path:
    sys.path.append('.')

from particle_filter import (
    set_global_particle_seed, reset_seed_counter, get_next_deterministic_seed,
    generate_particle, generate_particle_historical
)
from optimizador import set_global_eval_seed, calcular_resultados_optimizacion, load_static_params_once
from analizar_utilidad import set_analizar_utilidad_seed, guardar_demanda_real_simulada

def test_particle_seed_consistency():
    """Verifica que la generación de partículas sea consistente con el mismo seed."""
    print("🧪 Test 1: Consistencia de seeds en generación de partículas")
    
    # Configuración
    n_productos, n_tiendas = 10, 2
    precios_base = np.full((n_productos,), 30.0)
    seed = 42
    
    # Primera generación
    set_global_particle_seed(seed)
    reset_seed_counter()
    particle1 = generate_particle(n_productos, n_tiendas, precios_base, variation_factor=0.1)
    
    # Segunda generación con el mismo seed
    set_global_particle_seed(seed)
    reset_seed_counter()
    particle2 = generate_particle(n_productos, n_tiendas, precios_base, variation_factor=0.1)
    
    # Verificación
    diferencia = np.abs(particle1 - particle2).max()
    print(f"   Diferencia máxima entre partículas: {diferencia}")
    
    if diferencia < 1e-10:
        print("   ✅ PASS: Las partículas son idénticas con el mismo seed")
        return True
    else:
        print("   ❌ FAIL: Las partículas difieren con el mismo seed")
        print(f"   Partícula 1[0,0]: {particle1[0,0]:.6f}")
        print(f"   Partícula 2[0,0]: {particle2[0,0]:.6f}")
        return False

def test_optimizador_seed_consistency():
    """Verifica que el optimizador sea consistente con el mismo seed."""
    print("\n🧪 Test 2: Consistencia de seeds en optimizador")
    
    # Configuración básica
    precios_test = np.full((10, 2), 25.0)  # Precios fijos para test
    
    # Cargar inventario inicial de la misma manera que el código principal
    static_params = load_static_params_once("parametros", 10, 2)
    inventario_inicial = static_params["I_initial_global"]
    
    semana = 1
    
    # Primera evaluación
    set_global_eval_seed(123)
    resultado1 = calcular_resultados_optimizacion(
        precios_test, inventario_inicial, semana, "parametros", use_eval_seed=True
    )
    
    # Segunda evaluación con el mismo seed
    set_global_eval_seed(123)  
    resultado2 = calcular_resultados_optimizacion(
        precios_test, inventario_inicial, semana, "parametros", use_eval_seed=True
    )
    
    # Verificación
    utilidad1 = resultado1["utilidad_total_horizonte"]
    utilidad2 = resultado2["utilidad_total_horizonte"]
    diferencia = abs(utilidad1 - utilidad2)
    
    print(f"   Utilidad 1: {utilidad1:.6f}")
    print(f"   Utilidad 2: {utilidad2:.6f}")
    print(f"   Diferencia: {diferencia:.6f}")
    
    if diferencia < 1e-6:
        print("   ✅ PASS: El optimizador es determinístico con el mismo seed")
        return True
    else:
        print("   ❌ FAIL: El optimizador da resultados diferentes con el mismo seed")
        return False

def test_demanda_real_seed_consistency():
    """Verifica que la simulación de demanda real sea consistente."""
    print("\n🧪 Test 3: Consistencia de seeds en simulación de demanda real")
    
    # Preparar datos de test
    mu_dict = {(0, 0, 1): 100.0, (0, 1, 1): 80.0, (1, 0, 1): 120.0, (1, 1, 1): 90.0}
    sigma_dict = {(0, 0, 1): 10.0, (0, 1, 1): 8.0, (1, 0, 1): 12.0, (1, 1, 1): 9.0}
    semana = 5
    
    # Limpiar archivo previo
    ruta_demanda = "resultados/demanda_real.csv"
    if os.path.exists(ruta_demanda):
        os.remove(ruta_demanda)
    
    # Primera simulación
    set_analizar_utilidad_seed(456)
    guardar_demanda_real_simulada(semana, mu_dict, sigma_dict, n_muestras=5)
    
    # Leer resultado
    import pandas as pd
    df1 = pd.read_csv(ruta_demanda)
    os.remove(ruta_demanda)  # Limpiar para segunda prueba
    
    # Segunda simulación con el mismo seed
    set_analizar_utilidad_seed(456) 
    guardar_demanda_real_simulada(semana, mu_dict, sigma_dict, n_muestras=5)
    df2 = pd.read_csv(ruta_demanda)
    
    # Verificación
    diferencias = (df1['demanda_real'] - df2['demanda_real']).abs()
    max_diferencia = diferencias.max()
    
    print(f"   Diferencia máxima en demanda real: {max_diferencia}")
    print(f"   Demandas iguales: {(diferencias < 1e-10).all()}")
    
    # Limpiar archivo de test
    if os.path.exists(ruta_demanda):
        os.remove(ruta_demanda)
    
    if max_diferencia < 1e-10:
        print("   ✅ PASS: La simulación de demanda real es determinística")
        return True
    else:
        print("   ❌ FAIL: La simulación de demanda real no es determinística")
        return False

def test_seed_sequence_determinism():
    """Verifica que la secuencia de seeds determinísticos sea reproducible."""
    print("\n🧪 Test 4: Determinismo de secuencia de seeds")
    
    # Primera secuencia
    set_global_particle_seed(789)
    reset_seed_counter()
    seeds1 = [get_next_deterministic_seed() for _ in range(10)]
    
    # Segunda secuencia
    set_global_particle_seed(789)
    reset_seed_counter()
    seeds2 = [get_next_deterministic_seed() for _ in range(10)]
    
    print(f"   Seeds 1: {seeds1[:5]}...")
    print(f"   Seeds 2: {seeds2[:5]}...")
    
    if seeds1 == seeds2:
        print("   ✅ PASS: La secuencia de seeds es determinística")
        return True
    else:
        print("   ❌ FAIL: La secuencia de seeds no es determinística")
        return False

def main():
    """Ejecuta todos los tests de verificación de seeds."""
    print("="*70)
    print("🔍 VERIFICACIÓN DEL MANEJO DE SEEDS EN EL EXPERIMENTO")
    print("="*70)
    
    tests = [
        test_particle_seed_consistency,
        test_optimizador_seed_consistency, 
        test_demanda_real_seed_consistency,
        test_seed_sequence_determinism
    ]
    
    resultados = []
    for test in tests:
        try:
            resultado = test()
            resultados.append(resultado)
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            resultados.append(False)
    
    print("\n" + "="*70)
    print("📊 RESUMEN DE VERIFICACIÓN")
    print("="*70)
    
    tests_passed = sum(resultados)
    total_tests = len(resultados)
    
    if all(resultados):
        print(f"🎉 TODOS LOS TESTS PASARON ({tests_passed}/{total_tests})")
        print("✅ El manejo de seeds es correcto y determinístico")
        print("✅ El experimento producirá resultados reproducibles")
    else:
        print(f"⚠️  ALGUNOS TESTS FALLARON ({tests_passed}/{total_tests})")
        print("❌ Es necesario revisar el manejo de seeds antes del experimento")
        
        for i, (test, resultado) in enumerate(zip(tests, resultados)):
            status = "✅" if resultado else "❌"
            print(f"   {status} Test {i+1}: {test.__name__}")
    
    return all(resultados)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 