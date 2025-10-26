#!/usr/bin/env python3
"""Check if the current best 20-vertex configuration is arithmetic."""

import json
import sys
# Import from local module
sys.path.insert(0, '.')
from check_local_maxima_arithmetic import check_arithmeticity

# Load the current best configuration
with open('20vertex_maximal_intermediate.json', 'r') as f:
    config = json.load(f)

print(f"\nChecking arithmeticity of best configuration:")
print(f"Volume: {config['volume']:.8f}")
print(f"Ratio to dodecahedron: {config['ratio_to_dodecahedron']:.6f}")

result = check_arithmeticity(config)

if 'error' not in result:
    print(f"\nHolonomy analysis:")
    print(f"Number of generators: {result['num_generators']}")
    print(f"Integral traces: {result['integral_traces']}/{result['num_generators']}")
    print(f"Traces: {result['traces']}")
    
    if result['all_integral']:
        print("\n*** ALL TRACES ARE INTEGRAL - DEFINITELY ARITHMETIC! ***")
    elif result['mostly_integral']:
        print("\n*** Most traces integral - likely arithmetic ***")
    else:
        print("\n*** Not obviously arithmetic ***")
else:
    print(f"Error: {result['error']}")