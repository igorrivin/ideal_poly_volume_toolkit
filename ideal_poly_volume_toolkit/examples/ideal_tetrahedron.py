import numpy as np
from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay

Z = np.array([1+0j, 0+0j, np.exp(1j*np.pi/3)], dtype=np.complex128)
print('[Ideal Tetrahedron via Delaunay]')
print('  volume (fast): ', ideal_poly_volume_via_delaunay(Z, mode='fast', series_terms=64))
print('  volume (exact):', ideal_poly_volume_via_delaunay(Z, mode='eval_only', dps=200))
