import numpy as np
from historical_prices import load_historical_prices, get_historical_prices_matrix

# Variable global para almacenar los datos históricos (se carga una sola vez)
_precios_historicos_global = None

def initialize_historical_prices():
    """
    Inicializa los datos históricos una sola vez al inicio del programa.
    """
    global _precios_historicos_global
    if _precios_historicos_global is None:
        print("🔄 Cargando datos históricos por primera vez...")
        _precios_historicos_global = load_historical_prices()
    return _precios_historicos_global

def generate_particle(n_productos, n_tiendas, precios_base, variation_factor=0.1):
    """
    Generate a single particle representing price configurations.
    Each particle is a matrix of prices with dimensions (n_productos, n_tiendas).
    We'll use random variations around base prices.
    
    Args:
        n_productos: Number of products
        n_tiendas: Number of stores  
        precios_base: Base prices matrix (n_productos, n_tiendas)
        variation_factor: Factor for price variation (default 10%)
        
    Returns:
        Particle matrix with price variations
    """
    # Random variation around base prices
    variation_range = 1.0 + variation_factor
    variation = np.random.uniform(1.0 - variation_factor, variation_range, size=(n_productos, n_tiendas))
    return precios_base * variation

def generate_particle_historical(n_productos, n_tiendas, precios_historicos, semana_objetivo, variation_factor=0.1):
    """
    Generate a single particle using historical prices as the starting point.
    
    Args:
        n_productos: Number of products
        n_tiendas: Number of stores  
        precios_historicos: Historical prices matrix (n_productos, n_tiendas)
        semana_objetivo: Week number for logging purposes
        variation_factor: Factor for price variation (default 10%)
        
    Returns:
        Particle matrix with price variations around historical prices
    """
    if precios_historicos is None:
        print(f"⚠️  No hay precios históricos disponibles para semana {semana_objetivo}, usando precios por defecto")
        # Fallback a precios por defecto si no hay datos históricos
        precios_default = np.full((n_productos, n_tiendas), 30.0)
        return generate_particle(n_productos, n_tiendas, precios_default, variation_factor)
    
    # Random variation around historical prices
    variation_range = 1.0 + variation_factor
    variation = np.random.uniform(1.0 - variation_factor, variation_range, size=(n_productos, n_tiendas))
    return precios_historicos * variation

def particle_filter_optimization(n_particles, n_productos, n_tiendas, precios_base, evaluate_fn):
    """
    Basic particle filter for price optimization (version simple).
    
    Args:
        n_particles: Number of particles to generate
        n_productos: Number of products
        n_tiendas: Number of stores
        precios_base: Base prices matrix (n_productos, n_tiendas)
        evaluate_fn: Function to evaluate particle utility
        
    Returns:
        best_particle: Matrix with best price configuration
        best_score: Utility score of the best particle
    """
    particles = []
    scores = []
    
    print(f"    Generando y evaluando {n_particles} partículas...")
    
    # Generate and evaluate particles
    for i in range(n_particles):
        # Generate particle (price configuration)
        p = generate_particle(n_productos, n_tiendas, precios_base)
        
        # Evaluate particle using provided evaluation function
        score = evaluate_fn(p)
        
        particles.append(p)
        scores.append(score)
        
        if (i + 1) % 5 == 0:
            print(f"      Evaluadas {i + 1}/{n_particles} partículas...")
    
    # Find best particle
    best_idx = np.argmax(scores)
    best_particle = particles[best_idx]
    best_score = scores[best_idx]
    
    print(f"    Mejor utilidad encontrada: {best_score:.2f}")
    
    return best_particle, best_score

def particle_filter_optimization_multi_resample(n_particles, n_productos, n_tiendas, precios_base, evaluate_fn, semana_idx):
    """
    Advanced particle filter with multi-stage resampling policy using historical prices.
    
    Implementa la política de resampling:
    - Inicial: ±30% variación, 50 partículas (usando precios históricos)
    - Round 1: ±20% variación, 25 partículas  
    - Round 2: ±10% variación, 22 partículas
    - Round 3: ±5% variación, 18 partículas
    - Selección: 95 partículas
    
    Args:
        n_particles: Number of initial particles
        n_productos: Number of products
        n_tiendas: Number of stores
        precios_base: Base prices matrix (n_productos, n_tiendas) - FALLBACK ONLY
        evaluate_fn: Function to evaluate particle utility
        semana_idx: Week index for historical price lookup
        
    Returns:
        best_particle: Matrix with best price configuration
        best_score: Utility score of the best particle
    """
    
    print(f"    === Filtro de Partículas Multi-Resampling para Semana {semana_idx} ===")
    
    # Inicializar datos históricos si no están cargados
    precios_historicos_data = initialize_historical_prices()
    
    # Obtener precios históricos para la semana específica
    precios_historicos_matrix = get_historical_prices_matrix(
        precios_historicos_data, semana_idx, n_productos, n_tiendas
    )
    
    if precios_historicos_matrix is not None:
        print(f"    📈 Usando precios históricos como punto de partida para semana {semana_idx}")
        precios_punto_partida = precios_historicos_matrix
    else:
        print(f"    ⚠️  Fallback: Usando precios base como punto de partida para semana {semana_idx}")
        precios_punto_partida = precios_base
    
    # Etapa Inicial: ±30% variación usando precios históricos
    print(f"    🔄 Etapa Inicial: Generando {n_particles} partículas con ±30% variación sobre precios históricos...")
    initial_particles = []
    initial_scores = []
    
    for i in range(n_particles):
        p = generate_particle_historical(n_productos, n_tiendas, precios_punto_partida, semana_idx, variation_factor=0.3)
        score = evaluate_fn(p)
        initial_particles.append(p)
        initial_scores.append(score)
        
        if (i + 1) % 10 == 0:
            print(f"      Evaluadas {i + 1}/{n_particles} partículas iniciales...")
    
    # Seleccionar mejores partículas para resampling
    sorted_indices = np.argsort(initial_scores)[::-1]  # Ordenar descendente
    
    # Round 1: ±20% variación, 25 partículas
    print(f"    🔄 Round 1: Resampling con ±20% variación, 25 partículas...")
    round1_particles = []
    round1_scores = []
    
    # Usar las mejores 10 partículas como base para Round 1
    best_10_indices = sorted_indices[:10]
    for i in range(25):
        base_idx = best_10_indices[i % 10]  # Rotar entre las mejores 10
        base_particle = initial_particles[base_idx]
        p = generate_particle_from_base(base_particle, variation_factor=0.2)  # ±20%
        score = evaluate_fn(p)
        round1_particles.append(p)
        round1_scores.append(score)
    
    # Combinar con partículas iniciales y seleccionar mejores
    all_particles_r1 = initial_particles + round1_particles
    all_scores_r1 = initial_scores + round1_scores
    sorted_indices_r1 = np.argsort(all_scores_r1)[::-1]
    
    # Round 2: ±10% variación, 22 partículas  
    print(f"    🔄 Round 2: Resampling con ±10% variación, 22 partículas...")
    round2_particles = []
    round2_scores = []
    
    # Usar las mejores 8 partículas como base para Round 2
    best_8_indices = sorted_indices_r1[:8]
    for i in range(22):
        base_idx = best_8_indices[i % 8]  # Rotar entre las mejores 8
        base_particle = all_particles_r1[base_idx]
        p = generate_particle_from_base(base_particle, variation_factor=0.1)  # ±10%
        score = evaluate_fn(p)
        round2_particles.append(p)
        round2_scores.append(score)
    
    # Combinar todas las partículas
    all_particles_r2 = all_particles_r1 + round2_particles
    all_scores_r2 = all_scores_r1 + round2_scores
    sorted_indices_r2 = np.argsort(all_scores_r2)[::-1]
    
    # Round 3: ±5% variación, 18 partículas
    print(f"    🔄 Round 3: Resampling con ±5% variación, 18 partículas...")
    round3_particles = []
    round3_scores = []
    
    # Usar las mejores 6 partículas como base para Round 3
    best_6_indices = sorted_indices_r2[:6]
    for i in range(18):
        base_idx = best_6_indices[i % 6]  # Rotar entre las mejores 6
        base_particle = all_particles_r2[base_idx]
        p = generate_particle_from_base(base_particle, variation_factor=0.05)  # ±5%
        score = evaluate_fn(p)
        round3_particles.append(p)
        round3_scores.append(score)
    
    # Selección final: Combinar todas y seleccionar la mejor
    all_particles_final = all_particles_r2 + round3_particles
    all_scores_final = all_scores_r2 + round3_scores
    
    print(f"    📊 Total de partículas evaluadas: {len(all_particles_final)}")
    
    # Encontrar la mejor partícula
    best_idx = np.argmax(all_scores_final)
    best_particle = all_particles_final[best_idx]
    best_score = all_scores_final[best_idx]
    
    print(f"    ✅ Mejor utilidad encontrada: {best_score:.2f}")
    print(f"    🏆 Partícula óptima encontrada tras resampling múltiple")
    
    return best_particle, best_score

def generate_particle_from_base(base_particle, variation_factor=0.1):
    """
    Generate a new particle based on an existing particle with variation.
    
    Args:
        base_particle: Base particle matrix
        variation_factor: Factor for variation around base particle
        
    Returns:
        New particle with variations around the base
    """
    variation_range = 1.0 + variation_factor
    variation = np.random.uniform(1.0 - variation_factor, variation_range, size=base_particle.shape)
    return base_particle * variation 