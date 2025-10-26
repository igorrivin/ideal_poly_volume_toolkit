import numpy as np
import json
from scipy.spatial import Delaunay
from ideal_poly_volume_toolkit.rivin_holonomy import Triangulation, generators_from_triangulation

def compute_holonomy_traces(vertices_complex, triangles):
    """
    Compute holonomy generators and their traces for an ideal polyhedron.
    
    Returns traces and checks if they're close to integers.
    """
    # Convert complex vertices to 2D points for triangulation
    vertices_2d = np.array([[z['real'], z['imag']] for z in vertices_complex if isinstance(z, dict)])
    
    # Build adjacency structure from Delaunay triangulation
    F = len(triangles)  # number of triangles
    adjacency = {}
    edge_id_map = {}
    edge_id = 0
    
    # Build adjacency between triangles
    for i, tri_i in enumerate(triangles):
        for side_i in range(3):
            v1_i, v2_i = tri_i[side_i], tri_i[(side_i + 1) % 3]
            edge = tuple(sorted([v1_i, v2_i]))
            
            # Find matching triangle
            for j, tri_j in enumerate(triangles):
                if i == j:
                    continue
                for side_j in range(3):
                    v1_j, v2_j = tri_j[side_j], tri_j[(side_j + 1) % 3]
                    if set([v1_j, v2_j]) == set([v1_i, v2_i]):
                        # Found matching edge
                        if (i, side_i) not in adjacency:
                            if edge not in edge_id_map:
                                edge_id_map[edge] = edge_id
                                edge_id += 1
                            adjacency[(i, side_i)] = (j, side_j, edge_id_map[edge])
    
    # Define order (counterclockwise) for each triangle
    order = {t: [0, 1, 2] for t in range(F)}
    
    # Define orientations for edges
    orientation = {}
    for edge, eid in edge_id_map.items():
        # Find triangles containing this edge
        for (t, s), (u, su, e) in adjacency.items():
            if e == eid:
                orientation[eid] = ((t, s), (u, su))
                break
    
    # Create triangulation object
    T = Triangulation(F, adjacency, order, orientation)
    
    # Zero shears (ideal polyhedra have no shearing)
    Z = {eid: 0.0 for eid in range(edge_id)}
    
    # Compute generators
    gens = generators_from_triangulation(T, Z, root=0)
    
    # Extract traces
    traces = []
    trace_analysis = []
    
    print(f"\nHolonomy analysis for {F} triangles:")
    print("-" * 70)
    print(f"Number of generators: {len(gens)}")
    print("\nTraces of holonomy generators:")
    
    for i, (u, v, tokens, M) in enumerate(gens):
        trace = M[0][0] + M[1][1]
        traces.append(trace)
        
        # Check proximity to integers
        nearest_int = round(trace)
        distance = abs(trace - nearest_int)
        is_close = distance < 0.01
        
        trace_analysis.append({
            'generator': i,
            'edge': (u, v),
            'trace': trace,
            'nearest_integer': nearest_int,
            'distance': distance,
            'possibly_integral': is_close
        })
        
        print(f"  Generator {i} (edge {u}-{v}): trace = {trace:.6f}")
        print(f"    Nearest integer: {nearest_int}, distance: {distance:.6f}")
        if is_close:
            print(f"    *** CLOSE TO INTEGER! ***")
    
    return traces, trace_analysis

# Load optimal configurations
with open('optimal_configurations.json', 'r') as f:
    data = json.load(f)

print("="*70)
print("ARITHMETICITY CHECK FOR MAXIMAL IDEAL POLYHEDRA")
print("="*70)

# Check the maximal 8-vertex configuration
print("\n1. MAXIMAL 8-VERTEX CONFIGURATION (volume = 6.488)")
print("-"*70)

config = data['discovered_configurations']['maximal_8vertex']
vertices = []
for k, v in config['vertices_complex'].items():
    if isinstance(v, dict):
        vertices.append(v)
triangles = config['delaunay_triangles']

traces1, analysis1 = compute_holonomy_traces(vertices, triangles)

# Check for arithmetic patterns
integral_count = sum(1 for a in analysis1 if a['possibly_integral'])
print(f"\nSummary: {integral_count}/{len(analysis1)} traces are close to integers")

# Check the golden ratio configuration
print("\n\n2. GOLDEN RATIO 8-VERTEX CONFIGURATION (volume = 6.002)")
print("-"*70)

config = data['discovered_configurations']['golden_ratio_8vertex']
vertices = []
for k, v in config['vertices_complex'].items():
    if isinstance(v, dict):
        vertices.append(v)
triangles = config['delaunay_triangles']

traces2, analysis2 = compute_holonomy_traces(vertices, triangles)

integral_count = sum(1 for a in analysis2 if a['possibly_integral'])
print(f"\nSummary: {integral_count}/{len(analysis2)} traces are close to integers")

# Check trace field generation
print("\n\n3. TRACE FIELD ANALYSIS")
print("-"*70)

def analyze_trace_field(traces):
    """Check if traces generate an algebraic number field close to integers."""
    # Look for algebraic relations
    products = []
    sums = []
    
    for i in range(len(traces)):
        for j in range(i, len(traces)):
            products.append(traces[i] * traces[j])
            sums.append(traces[i] + traces[j])
    
    all_values = traces + products + sums
    integral_values = []
    
    for val in all_values:
        nearest = round(val)
        if abs(val - nearest) < 0.01:
            integral_values.append((val, nearest))
    
    return integral_values

print("\nMaximal configuration trace field:")
integral_vals1 = analyze_trace_field(traces1)
print(f"Found {len(integral_vals1)} values close to integers in trace field")

print("\nGolden ratio configuration trace field:")
integral_vals2 = analyze_trace_field(traces2)
print(f"Found {len(integral_vals2)} values close to integers in trace field")

print("\n\nCONCLUSION:")
print("-"*70)
print("If many traces are close to integers, the polyhedron may be arithmetic.")
print("Arithmetic hyperbolic 3-manifolds have holonomies in PSL(2,O_K) where")
print("O_K is the ring of integers in a number field K.")
print("\nFor ideal polyhedra, arithmeticity would mean the configuration")
print("has deep number-theoretic significance!")