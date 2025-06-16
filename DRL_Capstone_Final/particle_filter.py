import numpy as np
# Asumimos que historical_prices.py está en el mismo directorio o es accesible
from historical_prices import load_historical_prices, get_historical_prices_matrix

_precios_historicos_global = None
DEFAULT_COSTO_TRANSPORTE = 3.8 # Definido como un default aquí

def initialize_historical_prices():
    global _precios_historicos_global
    if _precios_historicos_global is None:
        print("🔄 Cargando datos históricos por primera vez...")
        _precios_historicos_global = load_historical_prices()
    return _precios_historicos_global

def _generate_prices_for_store0(base_prices_store0, variation_factor, n_productos):
    """Helper para generar precios de la tienda 0."""
    if base_prices_store0 is None: # Fallback si no hay precios base
        base_prices_store0 = np.full((n_productos,), 30.0) # Vector para tienda 0
        print("      ⚠️ Usando precios default (30.0) para tienda 0 en _generate_prices_for_store0")

    variation_range_t0 = 1.0 + variation_factor
    variation_t0 = np.random.uniform(1.0 - variation_factor, variation_range_t0, size=(n_productos,))
    return base_prices_store0 * variation_t0

def _apply_arbitrage_constraint(particle_matrix, costo_transporte_param):
    """Aplica la restricción de arbitraje a las tiendas > 0 basadas en la tienda 0."""
    n_productos, n_tiendas = particle_matrix.shape
    if n_tiendas > 1:
        precios_referencia_tienda0 = particle_matrix[:, 0]
        for tienda_idx in range(1, n_tiendas):
            for prod_idx in range(n_productos):
                precio_ref = precios_referencia_tienda0[prod_idx]
                limite_inferior_arbitraje = precio_ref - costo_transporte_param
                limite_superior_arbitraje = precio_ref + costo_transporte_param
                
                # Muestrear uniformemente dentro del rango permitido, asegurando no negatividad
                precio_tienda_actual = np.random.uniform(
                    max(0.01, limite_inferior_arbitraje), 
                    max(0.011, limite_superior_arbitraje) # Asegurar que sup > inf
                )
                # Si el rango es inválido (lim_sup < lim_inf), podría quedarse en el lim_inf.
                # Esto puede pasar si costo_transporte es muy pequeño o negativo.
                # Una mejor forma sería clippear el precio original de la tienda (si se generó)
                # o generar dentro del rango.
                # La opción actual es generar dentro del rango.
                if limite_superior_arbitraje < limite_inferior_arbitraje: # Rango inválido
                     precio_tienda_actual = max(0.01, precio_ref) # Fallback a precio_ref o mínimo
                
                particle_matrix[prod_idx, tienda_idx] = precio_tienda_actual
    return np.maximum(particle_matrix, 0.01)


def generate_particle(n_productos, n_tiendas, precios_base_matrix, # Ahora espera matriz (n_prod, n_tiendas) o (n_prod,)
                      variation_factor=0.1, costo_transporte=DEFAULT_COSTO_TRANSPORTE):
    particle = np.zeros((n_productos, n_tiendas))
    
    # Determinar precios base para la tienda 0
    if precios_base_matrix.ndim == 2:
        precios_base_tienda0_actual = precios_base_matrix[:, 0]
    elif precios_base_matrix.ndim == 1 and len(precios_base_matrix) == n_productos:
        precios_base_tienda0_actual = precios_base_matrix
    else:
        # Fallback si la forma no es la esperada
        print(f"      ⚠️ Forma de precios_base_matrix inesperada ({precios_base_matrix.shape}) en generate_particle. Usando default.")
        precios_base_tienda0_actual = np.full((n_productos,), 30.0)

    particle[:, 0] = _generate_prices_for_store0(precios_base_tienda0_actual, variation_factor, n_productos)
    
    # Aplicar restricción de arbitraje para otras tiendas
    particle = _apply_arbitrage_constraint(particle, costo_transporte)
    return particle


def generate_particle_historical(n_productos, n_tiendas, precios_historicos_matrix, # Matriz (n_prod, n_tiendas) o None
                                 semana_objetivo, variation_factor=0.1, costo_transporte=DEFAULT_COSTO_TRANSPORTE):
    particle = np.zeros((n_productos, n_tiendas))
    
    precios_base_tienda0_actual = None
    if precios_historicos_matrix is None:
        print(f"      ⚠️ No hay precios históricos para semana {semana_objetivo}, usando default para tienda 0.")
        precios_base_tienda0_actual = np.full((n_productos,), 30.0)
    elif precios_historicos_matrix.ndim == 2 and precios_historicos_matrix.shape[0] == n_productos:
        precios_base_tienda0_actual = precios_historicos_matrix[:, 0] # Usar primera tienda
    elif precios_historicos_matrix.ndim == 1 and len(precios_historicos_matrix) == n_productos: # Ya es para tienda 0
        precios_base_tienda0_actual = precios_historicos_matrix
    else:
        print(f"      ⚠️ Forma de precios_historicos_matrix inesperada ({precios_historicos_matrix.shape}). Usando default para tienda 0.")
        precios_base_tienda0_actual = np.full((n_productos,), 30.0)

    particle[:, 0] = _generate_prices_for_store0(precios_base_tienda0_actual, variation_factor, n_productos)
    
    # Aplicar restricción de arbitraje para otras tiendas
    particle = _apply_arbitrage_constraint(particle, costo_transporte)
    return particle

# particle_filter_optimization no se usa en tu main_particle.py, pero lo actualizo por consistencia
def particle_filter_optimization(n_particles_arg, n_productos, n_tiendas, precios_base_matrix, evaluate_fn, costo_transporte=DEFAULT_COSTO_TRANSPORTE):
    particles = []
    scores = []
    print(f"    Generando y evaluando {n_particles_arg} partículas (versión simple)...")
    
    for i in range(n_particles_arg):
        p = generate_particle(n_productos, n_tiendas, precios_base_matrix, costo_transporte=costo_transporte)
        score = evaluate_fn(p)
        particles.append(p)
        scores.append(score)
        if (i + 1) % 5 == 0:
            print(f"      Evaluadas {i + 1}/{n_particles_arg} partículas...")
    
    if not scores:
        print("    ⚠️ No se evaluaron partículas. Devolviendo None.")
        return None, -float('inf')

    best_idx = np.argmax(scores)
    best_particle = particles[best_idx]
    best_score = scores[best_idx]
    print(f"    Mejor utilidad encontrada (simple): {best_score:.2f}")
    return best_particle, best_score

# MODIFICADO: Firma para coincidir con tu llamada, y uso de costo_transporte
def particle_filter_optimization_multi_resample(
    n_particles, # Para coincidir con tu llamada
    n_productos, 
    n_tiendas, 
    precios_base, # Este es precios_base_matrix (n_prod, n_tiendas) de tu main
    evaluate_fn, 
    semana_idx, 
    costo_transporte=DEFAULT_COSTO_TRANSPORTE # Añadido con default
):
    print(f"    === Filtro de Partículas Multi-Resampling para Semana {semana_idx} (Costo Transp: {costo_transporte}) ===")
    
    precios_historicos_data = initialize_historical_prices()
    precios_historicos_semana_completa = get_historical_prices_matrix(
        precios_historicos_data, semana_idx, n_productos, n_tiendas
    )
    
    precios_partida_tienda0_ref = None
    if precios_historicos_semana_completa is not None:
        print(f"    📈 Usando precios históricos de tienda 0 como punto de partida para semana {semana_idx}")
        precios_partida_tienda0_ref = precios_historicos_semana_completa[:, 0]
    else:
        print(f"    ⚠️  Fallback: Usando precios base de tienda 0 para semana {semana_idx}")
        if precios_base.ndim == 2:
            precios_partida_tienda0_ref = precios_base[:, 0]
        elif precios_base.ndim == 1 and len(precios_base) == n_productos : # Ya es para tienda 0
             precios_partida_tienda0_ref = precios_base
        else:
            print(f"      Fallback adicional: precios_base con forma inesperada ({precios_base.shape}), usando default 30.0 para tienda 0.")
            precios_partida_tienda0_ref = np.full((n_productos,), 30.0)


    print(f"    🔄 Etapa Inicial: Generando {n_particles} partículas con ±30% variación...")
    initial_particles = []
    initial_scores = []
    
    for i in range(n_particles): # Usa n_particles de la firma
        p = generate_particle_historical(n_productos, n_tiendas, precios_partida_tienda0_ref, 
                                         semana_idx, variation_factor=0.3, costo_transporte=costo_transporte)
        score = evaluate_fn(p)
        initial_particles.append(p)
        initial_scores.append(score)
        if (i + 1) % 10 == 0:
            print(f"      Evaluadas {i + 1}/{n_particles} partículas iniciales...")
    
    if not initial_scores:
        print("    ⚠️ No se generaron partículas iniciales. Devolviendo precios base (o default) y utilidad negativa.")
        # Generar una partícula base sin variación para devolver algo con la forma correcta
        fallback_particle_tienda0 = precios_partida_tienda0_ref if precios_partida_tienda0_ref is not None else np.full((n_productos,), 30.0)
        return generate_particle(n_productos, n_tiendas, fallback_particle_tienda0, 0.0, costo_transporte), -float('inf')

    # ... (resto de la lógica de resampling como la tenías, pero asegurándose que
    #      generate_particle_from_base también usa costo_transporte) ...
    
    sorted_indices = np.argsort(initial_scores)[::-1]
    
    rondas_config = [
        {"nombre": "Round 1", "n_particulas_ronda": 25, "n_mejores_base": 10, "variation_factor": 0.20},
        {"nombre": "Round 2", "n_particulas_ronda": 22, "n_mejores_base": 8,  "variation_factor": 0.10},
        {"nombre": "Round 3", "n_particulas_ronda": 18, "n_mejores_base": 6,  "variation_factor": 0.05}
    ]

    all_particles_acum = list(initial_particles) 
    all_scores_acum = list(initial_scores)     

    for ronda in rondas_config:
        print(f"    🔄 {ronda['nombre']}: Resampling con ±{ronda['variation_factor']*100}% variación, {ronda['n_particulas_ronda']} partículas...")
        
        current_round_particles = []
        current_round_scores = []
        
        sorted_indices_acum = np.argsort(all_scores_acum)[::-1]
        n_bases_disponibles = min(ronda['n_mejores_base'], len(all_particles_acum))

        if n_bases_disponibles == 0:
            print(f"      No hay partículas base disponibles para {ronda['nombre']}, saltando ronda.")
            continue

        best_indices_base = sorted_indices_acum[:n_bases_disponibles]
        
        for i in range(ronda['n_particulas_ronda']):
            base_idx_actual = best_indices_base[i % n_bases_disponibles] 
            base_particle_actual = all_particles_acum[base_idx_actual] # Esta es (n_prod, n_tiendas)
            
            p = generate_particle_from_base(base_particle_actual, 
                                            variation_factor=ronda['variation_factor'],
                                            costo_transporte=costo_transporte) # Pasar costo_transporte
            score = evaluate_fn(p)
            current_round_particles.append(p)
            current_round_scores.append(score)

        all_particles_acum.extend(current_round_particles)
        all_scores_acum.extend(current_round_scores)
            
    print(f"    📊 Total de partículas evaluadas en todas las rondas: {len(all_particles_acum)}")
    
    if not all_scores_acum: # Doble chequeo por si acaso
        print("    ⚠️ No se evaluaron partículas en total (después de rondas). Devolviendo precios base (o default) y utilidad negativa.")
        fallback_particle_tienda0 = precios_partida_tienda0_ref if precios_partida_tienda0_ref is not None else np.full((n_productos,), 30.0)
        return generate_particle(n_productos, n_tiendas, fallback_particle_tienda0, 0.0, costo_transporte), -float('inf')

    best_idx_final = np.argmax(all_scores_acum)
    best_particle_final = all_particles_acum[best_idx_final]
    best_score_final = all_scores_acum[best_idx_final]
    
    print(f"    ✅ Mejor utilidad encontrada tras resampling múltiple: {best_score_final:.2f}")
    
    return best_particle_final, best_score_final

# generate_particle_from_base ya fue modificado para usar costo_transporte
def generate_particle_from_base(base_particle_matrix, variation_factor=0.1, costo_transporte=DEFAULT_COSTO_TRANSPORTE):
    n_productos, n_tiendas = base_particle_matrix.shape
    new_particle = np.zeros_like(base_particle_matrix)

    # Generar precios para la Tienda 0 con variación sobre la base
    new_particle[:, 0] = _generate_prices_for_store0(base_particle_matrix[:, 0], variation_factor, n_productos)
    
    # Aplicar restricción de arbitraje para otras tiendas
    new_particle = _apply_arbitrage_constraint(new_particle, costo_transporte)
    return new_particle