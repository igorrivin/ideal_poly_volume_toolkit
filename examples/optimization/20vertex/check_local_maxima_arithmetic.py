#!/usr/bin/env python3
"""
Check arithmeticity of local maxima found during 20-vertex optimization.
Tests the conjecture that local maxima are more likely to be arithmetic.
"""

import numpy as np
import json
import os
from scipy.spatial import Delaunay
from ideal_poly_volume_toolkit.rivin_holonomy import Triangulation, generators_from_triangulation

def build_triangulation_from_config(vertices_dict):
    """Convert vertex configuration to triangulation for holonomy computation."""
    # Extract finite vertices (excluding infinity)
    vertices = []
    for k, v in sorted(vertices_dict.items()):
        if k != 'z_inf':
            vertices.append(complex(v['real'], v['imag']))
    
    # Convert to 2D points
    points_2d = np.array([[v.real, v.imag] for v in vertices])
    
    # Get Delaunay triangulation
    tri = Delaunay(points_2d)
    triangles = tri.simplices
    
    # Build adjacency structure
    F = len(triangles)
    adjacency = {}
    edge_id_map = {}
    edge_id = 0
    
    for i, tri_i in enumerate(triangles):
        for side_i in range(3):
            v1_i, v2_i = tri_i[side_i], tri_i[(side_i + 1) % 3]
            edge = tuple(sorted([v1_i, v2_i]))
            
            for j, tri_j in enumerate(triangles):
                if i == j:
                    continue
                for side_j in range(3):
                    v1_j, v2_j = tri_j[side_j], tri_j[(side_j + 1) % 3]
                    if set([v1_j, v2_j]) == set([v1_i, v2_i]):
                        if (i, side_i) not in adjacency:
                            if edge not in edge_id_map:
                                edge_id_map[edge] = edge_id
                                edge_id += 1
                            adjacency[(i, side_i)] = (j, side_j, edge_id_map[edge])
    
    # Define order and orientation
    order = {t: [0, 1, 2] for t in range(F)}
    orientation = {}
    for edge, eid in edge_id_map.items():
        for (t, s), (u, su, e) in adjacency.items():
            if e == eid:
                orientation[eid] = ((t, s), (u, su))
                break
    
    return Triangulation(F, adjacency, order, orientation), triangles

def check_arithmeticity(config):
    """Check if a configuration has arithmetic holonomy."""
    try:
        T, triangles = build_triangulation_from_config(config['vertices'])
        
        # Zero shears for ideal polyhedra
        Z = {eid: 0.0 for eid in range(len(T.Ori))}
        
        # Compute generators
        gens = generators_from_triangulation(T, Z, root=0)
        
        # Extract traces
        traces = []
        integral_traces = 0
        close_to_algebraic = 0
        
        for i, (u, v, tokens, M) in enumerate(gens):
            trace = M[0][0] + M[1][1]
            traces.append(trace)
            
            # Check proximity to integers
            nearest_int = round(trace)
            if abs(trace - nearest_int) < 0.001:
                integral_traces += 1
            
            # Check proximity to simple algebraic numbers
            # (e.g., ±√2, ±√3, golden ratio, etc.)
            algebraic_values = [
                np.sqrt(2), -np.sqrt(2), np.sqrt(3), -np.sqrt(3),
                (1 + np.sqrt(5))/2, (1 - np.sqrt(5))/2,  # golden ratio
                np.sqrt(5), -np.sqrt(5)
            ]
            
            for alg in algebraic_values:
                if abs(trace - alg) < 0.001:
                    close_to_algebraic += 1
                    break
        
        # Analyze trace field
        all_integral = (integral_traces == len(traces))
        mostly_integral = (integral_traces >= len(traces) * 0.8)
        has_algebraic = (close_to_algebraic > 0)
        
        return {
            'traces': traces,
            'num_generators': len(traces),
            'integral_traces': integral_traces,
            'all_integral': all_integral,
            'mostly_integral': mostly_integral,
            'has_algebraic': has_algebraic,
            'likely_arithmetic': all_integral or (mostly_integral and has_algebraic)
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'likely_arithmetic': False
        }

def analyze_all_local_maxima():
    """Analyze all saved local maxima for arithmeticity."""
    
    # Check if local maxima file exists
    if not os.path.exists('20vertex_local_maxima.json'):
        print("No local maxima file found yet. Waiting for optimization to save some...")
        return
    
    with open('20vertex_local_maxima.json', 'r') as f:
        data = json.load(f)
    
    print(f"\n{'='*70}")
    print("ARITHMETICITY ANALYSIS OF LOCAL MAXIMA")
    print(f"{'='*70}")
    print(f"Found {data['local_maxima_count']} local maxima to analyze\n")
    
    arithmetic_count = 0
    results = []
    
    for i, config in enumerate(data['configurations']):
        print(f"\nConfiguration {i+1}:")
        print(f"  Volume: {config['volume']:.8f}")
        print(f"  Ratio to dodecahedron: {config['volume']/data['dodecahedron_volume']:.6f}")
        
        arith_result = check_arithmeticity(config)
        
        if 'error' in arith_result:
            print(f"  Error in analysis: {arith_result['error']}")
        else:
            print(f"  Generators: {arith_result['num_generators']}")
            print(f"  Integral traces: {arith_result['integral_traces']}/{arith_result['num_generators']}")
            print(f"  Traces: {[f'{t:.6f}' for t in arith_result['traces']]}")
            
            if arith_result['likely_arithmetic']:
                print(f"  *** LIKELY ARITHMETIC! ***")
                arithmetic_count += 1
        
        results.append({
            'volume': config['volume'],
            'arithmetic': arith_result.get('likely_arithmetic', False),
            'traces': arith_result.get('traces', [])
        })
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY:")
    print(f"{'='*70}")
    print(f"Total local maxima analyzed: {len(data['configurations'])}")
    print(f"Likely arithmetic: {arithmetic_count} ({arithmetic_count/len(data['configurations'])*100:.1f}%)")
    
    # Save results
    output = {
        'analysis_date': str(datetime.now()),
        'local_maxima_count': len(data['configurations']),
        'arithmetic_count': arithmetic_count,
        'results': results,
        'conjecture_support': arithmetic_count > len(data['configurations']) * 0.5
    }
    
    with open('20vertex_arithmetic_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to 20vertex_arithmetic_analysis.json")
    
    if output['conjecture_support']:
        print("\n*** CONJECTURE SUPPORTED: Majority of local maxima appear arithmetic! ***")
    else:
        print("\n*** More data needed to support conjecture ***")

if __name__ == "__main__":
    from datetime import datetime
    analyze_all_local_maxima()