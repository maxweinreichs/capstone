"""
Políticas de Resampling para Filtro de Partículas
Todas las políticas escalan proporcionalmente al número inicial de partículas
"""
import numpy as np
from particle_filter import (
    generate_particle_historical, generate_particle_from_base, 
    initialize_historical_prices, get_historical_prices_matrix,
    set_global_particle_seed, reset_seed_counter
)

def sin_resampling_policy(n_particles, n_productos, n_tiendas, precios_base, evaluate_fn, 
                         semana_idx, costo_transporte=3.8):
    """
    Política Sin Resampling: Solo generación inicial con todo el presupuesto de partículas.
    """
    print(f"    === Sin Resampling para Semana {semana_idx} ===")
    
    precios_historicos_data = initialize_historical_prices()
    precios_historicos_semana_completa = get_historical_prices_matrix(
        precios_historicos_data, semana_idx, n_productos, n_tiendas
    )
    
    precios_partida_tienda0_ref = _get_reference_prices(
        precios_historicos_semana_completa, precios_base, semana_idx, n_productos
    )
    
    print(f"    🔄 Generando {n_particles} partículas sin resampling...")
    
    particles = []
    scores = []
    
    for i in range(n_particles):
        p = generate_particle_historical(n_productos, n_tiendas, precios_partida_tienda0_ref, 
                                         semana_idx, variation_factor=0.3, costo_transporte=costo_transporte)
        score = evaluate_fn(p)
        particles.append(p)
        scores.append(score)
        
        if (i + 1) % 10 == 0:
            print(f"      Evaluadas {i + 1}/{n_particles} partículas...")
    
    if not scores:
        return _fallback_particle(n_productos, n_tiendas, precios_partida_tienda0_ref, costo_transporte), -float('inf')
    
    best_idx = np.argmax(scores)
    best_particle = particles[best_idx]
    best_score = scores[best_idx]
    
    print(f"    ✅ Mejor utilidad encontrada (sin resampling): {best_score:.2f}")
    return best_particle, best_score


def actual_policy(n_particles, n_productos, n_tiendas, precios_base, evaluate_fn, 
                 semana_idx, costo_transporte=3.8):
    """
    Política Actual: Multi-etapa con porcentajes fijos escalados proporcionalmente.
    """
    print(f"    === Política Actual (Multi-Etapa) para Semana {semana_idx} ===")
    
    precios_historicos_data = initialize_historical_prices()
    precios_historicos_semana_completa = get_historical_prices_matrix(
        precios_historicos_data, semana_idx, n_productos, n_tiendas
    )
    
    precios_partida_tienda0_ref = _get_reference_prices(
        precios_historicos_semana_completa, precios_base, semana_idx, n_productos
    )
    
    # Configuración proporcional basada en n_particles
    rondas_config = [
        {"nombre": "Round 1", "porcentaje": 0.50, "mejores_porcentaje": 0.20, "variation_factor": 0.20},
        {"nombre": "Round 2", "porcentaje": 0.44, "mejores_porcentaje": 0.16, "variation_factor": 0.10},
        {"nombre": "Round 3", "porcentaje": 0.36, "mejores_porcentaje": 0.12, "variation_factor": 0.05}
    ]
    
    return _execute_multistage_resampling(
        n_particles, n_productos, n_tiendas, precios_partida_tienda0_ref,
        semana_idx, evaluate_fn, rondas_config, costo_transporte, "Actual"
    )


def agresivo_policy(n_particles, n_productos, n_tiendas, precios_base, evaluate_fn, 
                   semana_idx, costo_transporte=3.8):
    """
    Política Agresiva: Muchas rondas con convergencia rápida hacia óptimos locales.
    """
    print(f"    === Política Agresiva para Semana {semana_idx} ===")
    
    precios_historicos_data = initialize_historical_prices()
    precios_historicos_semana_completa = get_historical_prices_matrix(
        precios_historicos_data, semana_idx, n_productos, n_tiendas
    )
    
    precios_partida_tienda0_ref = _get_reference_prices(
        precios_historicos_semana_completa, precios_base, semana_idx, n_productos
    )
    
    # Configuración agresiva: más rondas, convergencia rápida
    inicial_porcentaje = 0.6  # 60% en inicial, 40% en resampling
    rondas_config = [
        {"nombre": "Round 1", "porcentaje": 0.20, "mejores_porcentaje": 0.30, "variation_factor": 0.25},
        {"nombre": "Round 2", "porcentaje": 0.15, "mejores_porcentaje": 0.20, "variation_factor": 0.15},
        {"nombre": "Round 3", "porcentaje": 0.10, "mejores_porcentaje": 0.15, "variation_factor": 0.08},
        {"nombre": "Round 4", "porcentaje": 0.08, "mejores_porcentaje": 0.10, "variation_factor": 0.04},
        {"nombre": "Round 5", "porcentaje": 0.07, "mejores_porcentaje": 0.08, "variation_factor": 0.02}
    ]
    
    n_inicial = int(n_particles * inicial_porcentaje)
    
    return _execute_multistage_resampling(
        n_inicial, n_productos, n_tiendas, precios_partida_tienda0_ref,
        semana_idx, evaluate_fn, rondas_config, costo_transporte, "Agresiva", n_particles
    )


def conservador_policy(n_particles, n_productos, n_tiendas, precios_base, evaluate_fn, 
                      semana_idx, costo_transporte=3.8):
    """
    Política Conservadora: Pocas rondas, más énfasis en exploración inicial.
    """
    print(f"    === Política Conservadora para Semana {semana_idx} ===")
    
    precios_historicos_data = initialize_historical_prices()
    precios_historicos_semana_completa = get_historical_prices_matrix(
        precios_historicos_data, semana_idx, n_productos, n_tiendas
    )
    
    precios_partida_tienda0_ref = _get_reference_prices(
        precios_historicos_semana_completa, precios_base, semana_idx, n_productos
    )
    
    # Configuración conservadora: menos rondas, más exploración
    inicial_porcentaje = 0.70  # 70% en inicial, 30% en resampling
    rondas_config = [
        {"nombre": "Round 1", "porcentaje": 0.20, "mejores_porcentaje": 0.40, "variation_factor": 0.15},
        {"nombre": "Round 2", "porcentaje": 0.10, "mejores_porcentaje": 0.25, "variation_factor": 0.08}
    ]
    
    n_inicial = int(n_particles * inicial_porcentaje)
    
    return _execute_multistage_resampling(
        n_inicial, n_productos, n_tiendas, precios_partida_tienda0_ref,
        semana_idx, evaluate_fn, rondas_config, costo_transporte, "Conservadora", n_particles
    )


def adaptativo_policy(n_particles, n_productos, n_tiendas, precios_base, evaluate_fn, 
                     semana_idx, costo_transporte=3.8):
    """
    Política Adaptativa: Ajusta parámetros basándose en la mejora obtenida.
    """
    print(f"    === Política Adaptativa para Semana {semana_idx} ===")
    
    precios_historicos_data = initialize_historical_prices()
    precios_historicos_semana_completa = get_historical_prices_matrix(
        precios_historicos_data, semana_idx, n_productos, n_tiendas
    )
    
    precios_partida_tienda0_ref = _get_reference_prices(
        precios_historicos_semana_completa, precios_base, semana_idx, n_productos
    )
    
    # Etapa inicial (40% del presupuesto)
    n_inicial = int(n_particles * 0.4)
    print(f"    🔄 Etapa Inicial: Generando {n_inicial} partículas con ±30% variación...")
    
    initial_particles = []
    initial_scores = []
    
    for i in range(n_inicial):
        p = generate_particle_historical(n_productos, n_tiendas, precios_partida_tienda0_ref, 
                                         semana_idx, variation_factor=0.3, costo_transporte=costo_transporte)
        score = evaluate_fn(p)
        initial_particles.append(p)
        initial_scores.append(score)
        if (i + 1) % 10 == 0:
            print(f"      Evaluadas {i + 1}/{n_inicial} partículas iniciales...")
    
    if not initial_scores:
        return _fallback_particle(n_productos, n_tiendas, precios_partida_tienda0_ref, costo_transporte), -float('inf')
    
    all_particles = list(initial_particles)
    all_scores = list(initial_scores)
    presupuesto_restante = n_particles - n_inicial
    
    # Rondas adaptativas
    ronda_num = 1
    mejor_inicial = max(initial_scores)
    
    while presupuesto_restante > 0 and ronda_num <= 3:
        mejor_actual = max(all_scores)
        mejora_porcentual = ((mejor_actual - mejor_inicial) / mejor_inicial) * 100 if mejor_inicial > 0 else 0
        
        # Adaptar configuración basándose en mejora
        if mejora_porcentual > 15:  # Buena convergencia
            n_ronda = min(int(presupuesto_restante * 0.4), presupuesto_restante)
            n_bases = max(3, int(len(all_particles) * 0.10))
            variation = max(0.02, 0.15 - (ronda_num * 0.05))
        elif mejora_porcentual > 5:  # Convergencia media
            n_ronda = min(int(presupuesto_restante * 0.5), presupuesto_restante)
            n_bases = max(5, int(len(all_particles) * 0.15))
            variation = max(0.03, 0.18 - (ronda_num * 0.04))
        else:  # Convergencia lenta - más exploración
            n_ronda = min(int(presupuesto_restante * 0.6), presupuesto_restante)
            n_bases = max(8, int(len(all_particles) * 0.25))
            variation = max(0.05, 0.25 - (ronda_num * 0.05))
        
        print(f"    🔄 Ronda Adaptativa {ronda_num}: Mejora={mejora_porcentual:.1f}%, {n_ronda} partículas, ±{variation*100:.1f}% variación...")
        
        # Ejecutar ronda
        sorted_indices = np.argsort(all_scores)[::-1]
        best_indices = sorted_indices[:n_bases]
        
        for i in range(n_ronda):
            base_idx = best_indices[i % len(best_indices)]
            base_particle = all_particles[base_idx]
            
            new_particle = generate_particle_from_base(base_particle, 
                                                      variation_factor=variation,
                                                      costo_transporte=costo_transporte)
            score = evaluate_fn(new_particle)
            all_particles.append(new_particle)
            all_scores.append(score)
        
        presupuesto_restante -= n_ronda
        ronda_num += 1
    
    if not all_scores:
        return _fallback_particle(n_productos, n_tiendas, precios_partida_tienda0_ref, costo_transporte), -float('inf')
    
    best_idx = np.argmax(all_scores)
    best_particle = all_particles[best_idx]
    best_score = all_scores[best_idx]
    
    print(f"    ✅ Mejor utilidad encontrada (adaptativa): {best_score:.2f} tras {len(all_particles)} evaluaciones")
    return best_particle, best_score


# ===== FUNCIONES AUXILIARES =====

def _get_reference_prices(precios_historicos_semana_completa, precios_base, semana_idx, n_productos):
    """Obtiene precios de referencia, priorizando históricos."""
    if precios_historicos_semana_completa is not None:
        print(f"    📈 Usando precios históricos de tienda 0 como punto de partida para semana {semana_idx}")
        return precios_historicos_semana_completa[:, 0]
    else:
        print(f"    ⚠️  Fallback: Usando precios base de tienda 0 para semana {semana_idx}")
        if precios_base.ndim == 2:
            return precios_base[:, 0]
        elif precios_base.ndim == 1 and len(precios_base) == n_productos:
            return precios_base
        else:
            print(f"      Fallback adicional: precios_base con forma inesperada ({precios_base.shape}), usando default 30.0 para tienda 0.")
            return np.full((n_productos,), 30.0)


def _fallback_particle(n_productos, n_tiendas, precios_partida_tienda0_ref, costo_transporte):
    """Genera partícula de fallback cuando no se evalúan partículas."""
    from particle_filter import generate_particle
    fallback_particle_tienda0 = precios_partida_tienda0_ref if precios_partida_tienda0_ref is not None else np.full((n_productos,), 30.0)
    return generate_particle(n_productos, n_tiendas, fallback_particle_tienda0, 0.0, costo_transporte)


def _execute_multistage_resampling(n_inicial, n_productos, n_tiendas, precios_partida_tienda0_ref,
                                  semana_idx, evaluate_fn, rondas_config, costo_transporte, 
                                  nombre_politica, n_total=None):
    """Ejecuta resampling multi-etapa con configuración dada."""
    if n_total is None:
        n_total = n_inicial
    
    print(f"    🔄 Etapa Inicial: Generando {n_inicial} partículas con ±30% variación...")
    initial_particles = []
    initial_scores = []
    
    for i in range(n_inicial):
        p = generate_particle_historical(n_productos, n_tiendas, precios_partida_tienda0_ref, 
                                         semana_idx, variation_factor=0.3, costo_transporte=costo_transporte)
        score = evaluate_fn(p)
        initial_particles.append(p)
        initial_scores.append(score)
        if (i + 1) % 10 == 0:
            print(f"      Evaluadas {i + 1}/{n_inicial} partículas iniciales...")
    
    if not initial_scores:
        return _fallback_particle(n_productos, n_tiendas, precios_partida_tienda0_ref, costo_transporte), -float('inf')
    
    all_particles = list(initial_particles)
    all_scores = list(initial_scores)
    
    # Ejecutar rondas de resampling
    for ronda in rondas_config:
        n_particulas_ronda = int(n_total * ronda["porcentaje"])
        n_mejores_base = max(1, int(len(all_particles) * ronda["mejores_porcentaje"]))
        
        if n_particulas_ronda == 0:
            continue
            
        print(f"    🔄 {ronda['nombre']}: {n_particulas_ronda} partículas, top {n_mejores_base} bases, ±{ronda['variation_factor']*100}% variación...")
        
        sorted_indices = np.argsort(all_scores)[::-1]
        best_indices = sorted_indices[:n_mejores_base]
        
        current_round_particles = []
        current_round_scores = []
        
        for i in range(n_particulas_ronda):
            base_idx = best_indices[i % len(best_indices)]
            base_particle = all_particles[base_idx]
            
            p = generate_particle_from_base(base_particle, 
                                          variation_factor=ronda['variation_factor'],
                                          costo_transporte=costo_transporte)
            score = evaluate_fn(p)
            current_round_particles.append(p)
            current_round_scores.append(score)
        
        all_particles.extend(current_round_particles)
        all_scores.extend(current_round_scores)
    
    print(f"    📊 Total de partículas evaluadas ({nombre_politica}): {len(all_particles)}")
    
    if not all_scores:
        return _fallback_particle(n_productos, n_tiendas, precios_partida_tienda0_ref, costo_transporte), -float('inf')
    
    best_idx = np.argmax(all_scores)
    best_particle = all_particles[best_idx]
    best_score = all_scores[best_idx]
    
    print(f"    ✅ Mejor utilidad encontrada ({nombre_politica}): {best_score:.2f}")
    return best_particle, best_score


# ===== MAPEO DE POLÍTICAS =====
RESAMPLING_POLICIES = {
    "sin_resampling": sin_resampling_policy,
    "actual": actual_policy,
    "agresivo": agresivo_policy,
    "conservador": conservador_policy,
    "adaptativo": adaptativo_policy
}

def get_policy_function(policy_name):
    """Obtiene la función de política por nombre."""
    if policy_name not in RESAMPLING_POLICIES:
        raise ValueError(f"Política '{policy_name}' no reconocida. Disponibles: {list(RESAMPLING_POLICIES.keys())}")
    return RESAMPLING_POLICIES[policy_name] 