import numpy as np

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
    Advanced particle filter with multi-stage resampling policy.
    
    Implementa la política de resampling proporcional:
    - Inicial: ±100% variación, n_particles
    - Round 1: ±20% variación, ~50% de n_particles (cada seleccionada se usa 2 veces)
    - Round 2: ±10% variación, ~40% de n_particles (cada seleccionada se usa 2 veces)
    - Round 3: ±5% variación, ~30% de n_particles (cada seleccionada se usa 2 veces)
    
    Args:
        n_particles: Number of initial particles
        n_productos: Number of products
        n_tiendas: Number of stores
        precios_base: Base prices matrix (n_productos, n_tiendas)
        evaluate_fn: Function to evaluate particle utility
        semana_idx: Week index for logging
        
    Returns:
        best_particle: Matrix with best price configuration
        best_score: Utility score of the best particle
    """
    
    print(f"    === Filtro de Partículas Multi-Resampling para Semana {semana_idx} ===")
    
    # Calcular cantidades proporcionales
    round1_particles_count = int(n_particles * 0.5)  # 50% del inicial
    round2_particles_count = int(n_particles * 0.4)  # 40% del inicial  
    round3_particles_count = int(n_particles * 0.3)  # 30% del inicial
    
    # Calcular cuántas partículas base seleccionar (cada una se usa 2 veces)
    round1_base_count = round1_particles_count // 2
    round2_base_count = round2_particles_count // 2
    round3_base_count = round3_particles_count // 2
    
    print(f"    📊 Configuración de rounds:")
    print(f"      Round 1: {round1_particles_count} partículas ({round1_base_count} bases × 2)")
    print(f"      Round 2: {round2_particles_count} partículas ({round2_base_count} bases × 2)")
    print(f"      Round 3: {round3_particles_count} partículas ({round3_base_count} bases × 2)")
    
    # Etapa Inicial: ±100% variación
    print(f"    🔄 Etapa Inicial: Generando {n_particles} partículas con ±100% variación...")
    initial_particles = []
    initial_scores = []
    
    for i in range(n_particles):
        p = generate_particle(n_productos, n_tiendas, precios_base, variation_factor=1.0)  # ±100%
        score = evaluate_fn(p)
        initial_particles.append(p)
        initial_scores.append(score)
        
        if (i + 1) % 10 == 0:
            print(f"      Evaluadas {i + 1}/{n_particles} partículas iniciales...")
    
    # Seleccionar mejores partículas para resampling
    sorted_indices = np.argsort(initial_scores)[::-1]  # Ordenar descendente
    
    # Round 1: ±20% variación
    print(f"    🔄 Round 1: Resampling con ±20% variación, {round1_particles_count} partículas...")
    round1_particles = []
    round1_scores = []
    
    # Seleccionar las mejores partículas base y usar cada una exactamente 2 veces
    best_round1_indices = sorted_indices[:round1_base_count]
    for base_idx in best_round1_indices:
        base_particle = initial_particles[base_idx]
        # Generar 2 variaciones de cada partícula base
        for _ in range(2):
            p = generate_particle_from_base(base_particle, variation_factor=0.2)  # ±20%
            score = evaluate_fn(p)
            round1_particles.append(p)
            round1_scores.append(score)
    
    # Combinar con partículas iniciales y seleccionar mejores
    all_particles_r1 = initial_particles + round1_particles
    all_scores_r1 = initial_scores + round1_scores
    sorted_indices_r1 = np.argsort(all_scores_r1)[::-1]
    
    # Round 2: ±10% variación
    print(f"    🔄 Round 2: Resampling con ±10% variación, {round2_particles_count} partículas...")
    round2_particles = []
    round2_scores = []
    
    # Seleccionar las mejores partículas base y usar cada una exactamente 2 veces
    best_round2_indices = sorted_indices_r1[:round2_base_count]
    for base_idx in best_round2_indices:
        base_particle = all_particles_r1[base_idx]
        # Generar 2 variaciones de cada partícula base
        for _ in range(2):
            p = generate_particle_from_base(base_particle, variation_factor=0.1)  # ±10%
            score = evaluate_fn(p)
            round2_particles.append(p)
            round2_scores.append(score)
    
    # Combinar todas las partículas
    all_particles_r2 = all_particles_r1 + round2_particles
    all_scores_r2 = all_scores_r1 + round2_scores
    sorted_indices_r2 = np.argsort(all_scores_r2)[::-1]
    
    # Round 3: ±5% variación
    print(f"    🔄 Round 3: Resampling con ±5% variación, {round3_particles_count} partículas...")
    round3_particles = []
    round3_scores = []
    
    # Seleccionar las mejores partículas base y usar cada una exactamente 2 veces
    best_round3_indices = sorted_indices_r2[:round3_base_count]
    for base_idx in best_round3_indices:
        base_particle = all_particles_r2[base_idx]
        # Generar 2 variaciones de cada partícula base
        for _ in range(2):
            p = generate_particle_from_base(base_particle, variation_factor=0.05)  # ±5%
            score = evaluate_fn(p)
            round3_particles.append(p)
            round3_scores.append(score)
    
    # Selección final: Combinar todas y seleccionar la mejor
    all_particles_final = all_particles_r2 + round3_particles
    all_scores_final = all_scores_r2 + round3_scores
    
    total_evaluations = len(all_particles_final)
    print(f"    📊 Total de partículas evaluadas: {total_evaluations}")
    print(f"      Inicial: {n_particles}, R1: {len(round1_particles)}, R2: {len(round2_particles)}, R3: {len(round3_particles)}")
    
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