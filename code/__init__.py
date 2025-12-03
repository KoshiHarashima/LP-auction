"""
Rochet-Choné型オークション機構設計と最適輸送問題の双対問題
"""

from .utils import beta_pdf, make_tensor_grid_2d, make_tensor_grid_3d, make_tensor_grid_7d, product_beta_density
from .primal import solve_mechanism, solve_mechanism_iterative
from .primal_3goods import solve_mechanism_3goods_4synergy, solve_mechanism_3goods_4synergy_iterative
from .primal_2agents import solve_mechanism_2agents, solve_mechanism_2agents_iterative
from .dual import solve_dual, cost_snew, build_cost_matrix, discretize_signed_measure
from .plot import classify_region, plot_polyhedral_regions

__all__ = [
    'beta_pdf',
    'make_tensor_grid_2d',
    'make_tensor_grid_3d',
    'make_tensor_grid_7d',
    'product_beta_density',
    'solve_mechanism',
    'solve_mechanism_iterative',
    'solve_mechanism_3goods_4synergy',
    'solve_mechanism_3goods_4synergy_iterative',
    'solve_mechanism_2agents',
    'solve_mechanism_2agents_iterative',
    'solve_dual',
    'cost_snew',
    'build_cost_matrix',
    'discretize_signed_measure',
    'classify_region',
    'plot_polyhedral_regions',
]

