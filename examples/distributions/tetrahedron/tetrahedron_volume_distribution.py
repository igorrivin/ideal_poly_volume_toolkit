import numpy as np
import matplotlib.pyplot as plt
from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay
import time

def sample_uniform_sphere(n_samples, seed=None):
    """
    Sample points uniformly on the unit sphere using Marsaglia's method
    """
    if seed is not None:
        np.random.seed(seed)
    
    points = []
    while len(points) < n_samples:
        # Generate random point in [-1, 1] x [-1, 1]
        x1, x2 = 2 * np.random.random(2) - 1
        s = x1**2 + x2**2
        
        if s < 1:  # Accept if inside unit circle
            x = 2 * x1 * np.sqrt(1 - s)
            y = 2 * x2 * np.sqrt(1 - s)
            z = 1 - 2 * s
            points.append((x, y, z))
    
    return np.array(points[:n_samples])

def stereographic_project(sphere_point):
    """
    Stereographic projection from sphere to complex plane
    North pole (0,0,1) maps to infinity
    South pole (0,0,-1) maps to origin
    """
    x, y, z = sphere_point
    if abs(z - 1) < 1e-10:  # North pole
        return np.inf
    return complex(x / (1 - z), y / (1 - z))

def compute_tetrahedron_volumes(n_samples=10000, seed=42):
    """
    Compute distribution of ideal tetrahedron volumes
    with vertices at 0, 1, infinity, and a random fourth vertex
    """
    print(f"Computing {n_samples} tetrahedron volumes...")
    
    # Sample points on sphere
    sphere_points = sample_uniform_sphere(n_samples, seed)
    
    volumes = []
    failed = 0
    
    start_time = time.time()
    
    for i, sphere_pt in enumerate(sphere_points):
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            eta = (n_samples - i) / rate
            print(f"  Progress: {i}/{n_samples} ({i/n_samples*100:.1f}%), "
                  f"Rate: {rate:.1f} samples/s, ETA: {eta:.1f}s")
        
        # Project to complex plane
        z = stereographic_project(sphere_pt)
        
        # Skip if projection is at infinity or coincides with fixed vertices
        if not np.isfinite(z) or abs(z) < 1e-10 or abs(z - 1) < 1e-10:
            failed += 1
            continue
        
        # Compute volume of tetrahedron with vertices at 0, 1, infinity, z
        # Since infinity is already one vertex, we just need 0, 1, z
        vertices = np.array([0+0j, 1+0j, z])
        
        try:
            volume = ideal_poly_volume_via_delaunay(vertices, mode='fast', series_terms=96)
            if np.isfinite(volume) and volume > 0:
                volumes.append(volume)
            else:
                failed += 1
        except:
            failed += 1
    
    print(f"\nCompleted in {time.time() - start_time:.1f}s")
    print(f"Successful: {len(volumes)}, Failed: {failed}")
    
    return np.array(volumes), sphere_points

def analyze_distribution(volumes):
    """Analyze the volume distribution"""
    print("\nVolume Distribution Statistics:")
    print(f"  Count: {len(volumes)}")
    print(f"  Mean: {np.mean(volumes):.6f}")
    print(f"  Median: {np.median(volumes):.6f}")
    print(f"  Std Dev: {np.std(volumes):.6f}")
    print(f"  Min: {np.min(volumes):.6f}")
    print(f"  Max: {np.max(volumes):.6f}")
    print(f"  25th percentile: {np.percentile(volumes, 25):.6f}")
    print(f"  75th percentile: {np.percentile(volumes, 75):.6f}")
    
    # Check for known special value
    regular_tet_vol = 1.01494160640965
    print(f"\nRegular tetrahedron volume: {regular_tet_vol:.6f}")
    print(f"Fraction below regular: {np.sum(volumes < regular_tet_vol) / len(volumes):.4f}")
    
    return {
        'mean': np.mean(volumes),
        'median': np.median(volumes),
        'std': np.std(volumes),
        'min': np.min(volumes),
        'max': np.max(volumes)
    }

def plot_distribution(volumes, bins=100):
    """Create visualization of volume distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram
    ax = axes[0, 0]
    counts, bins, patches = ax.hist(volumes, bins=bins, density=True, 
                                   alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(1.01494, color='red', linestyle='--', linewidth=2, 
               label='Regular tetrahedron')
    ax.axvline(np.mean(volumes), color='green', linestyle='--', linewidth=2,
               label=f'Mean = {np.mean(volumes):.3f}')
    ax.set_xlabel('Volume')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Ideal Tetrahedron Volumes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log-scale histogram
    ax = axes[0, 1]
    ax.hist(volumes, bins=bins, density=True, alpha=0.7, color='blue', 
            edgecolor='black', log=True)
    ax.axvline(1.01494, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Volume')
    ax.set_ylabel('Log Density')
    ax.set_title('Log-Scale Distribution')
    ax.grid(True, alpha=0.3)
    
    # Cumulative distribution
    ax = axes[1, 0]
    sorted_volumes = np.sort(volumes)
    cumulative = np.arange(1, len(volumes) + 1) / len(volumes)
    ax.plot(sorted_volumes, cumulative, 'b-', linewidth=2)
    ax.axvline(1.01494, color='red', linestyle='--', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Volume')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(2, np.percentile(volumes, 99)))
    
    # Q-Q plot against exponential
    ax = axes[1, 1]
    # Transform volumes to check against exponential
    sorted_volumes = np.sort(volumes)
    n = len(volumes)
    theoretical_quantiles = -np.log(1 - (np.arange(1, n+1) - 0.5) / n)
    ax.scatter(theoretical_quantiles[:1000], sorted_volumes[:1000], 
               alpha=0.5, s=10)
    ax.plot([0, 5], [0, 5], 'r--', linewidth=2)
    ax.set_xlabel('Exponential Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title('Q-Q Plot vs Exponential')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tetrahedron_volume_distribution.png', dpi=150)
    plt.show()

def main():
    # Run simulation
    n_samples = 50000  # Adjust based on computational resources
    volumes, sphere_points = compute_tetrahedron_volumes(n_samples, seed=42)
    
    # Analyze
    stats = analyze_distribution(volumes)
    
    # Visualize
    plot_distribution(volumes, bins=100)
    
    # Check for patterns in extreme cases
    print("\nAnalyzing extreme cases:")
    
    # Find configurations with very small/large volumes
    idx_min = np.argmin(volumes)
    idx_max = np.argmax(volumes)
    
    print(f"\nMinimum volume configuration:")
    print(f"  Volume: {volumes[idx_min]:.6f}")
    z_min = stereographic_project(sphere_points[idx_min])
    print(f"  Fourth vertex at: {z_min}")
    
    print(f"\nMaximum volume configuration:")
    print(f"  Volume: {volumes[idx_max]:.6f}")
    z_max = stereographic_project(sphere_points[idx_max])
    print(f"  Fourth vertex at: {z_max}")
    
    # Save results
    np.savez('tetrahedron_volume_distribution.npz', 
             volumes=volumes, 
             sphere_points=sphere_points,
             stats=stats)
    print("\nResults saved to tetrahedron_volume_distribution.npz")

if __name__ == '__main__':
    main()