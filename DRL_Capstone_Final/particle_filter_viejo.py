import numpy as np

def generate_particle(n_productos, n_tiendas, precios_base):
    """
    Generate a single particle representing price configurations.
    Each particle is a matrix of prices with dimensions (n_productos, n_tiendas).
    We'll use random variations around base prices.
    """
    # Random variation between -10% and +10% of base prices
    variation = np.random.uniform(0.9, 1.1, size=(n_productos, n_tiendas))
    return precios_base * variation

def particle_filter_optimization(n_particles, n_productos, n_tiendas, precios_base, evaluate_fn):
    """
    Implement particle filter for price optimization.
    
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
    
    # Generate and evaluate particles
    for _ in range(n_particles):
        # Generate particle (price configuration)
        p = generate_particle(n_productos, n_tiendas, precios_base)
        
        # Evaluate particle using provided evaluation function
        score = evaluate_fn(p)
        
        particles.append(p)
        scores.append(score)
    
    # Find best particle
    best_idx = np.argmax(scores)
    best_particle = particles[best_idx]
    best_score = scores[best_idx]
    
    return best_particle, best_score 