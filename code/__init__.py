"""
Rochet-Choné型オークション機構設計と最適輸送問題の双対問題
"""

from .utils import beta_pdf, make_tensor_grid_2d, make_tensor_grid_3d, make_tensor_grid_7d, product_beta_density
from .primal import (
    solve_mechanism, 
    solve_mechanism_iterative,
    save_results,
    load_results
)
from .primal_3goods import (
    solve_mechanism_3goods_4synergy, 
    solve_mechanism_3goods_4synergy_iterative,
    save_results_3goods,
    load_results_3goods
)
from .primal_2agents import (
    solve_mechanism_2agents, 
    solve_mechanism_2agents_iterative,
    save_results_2agents,
    load_results_2agents
)
from .primal_multi_agent import (
    solve_mechanism_multi_agent,
    solve_mechanism_2agents as solve_mechanism_2agents_multi,
    solve_mechanism_multi_agent_iterative,
    solve_mechanism_2agents_iterative as solve_mechanism_2agents_iterative_multi,
    save_results_multi_agent,
    save_results_2agents as save_results_2agents_multi,
    load_results_multi_agent,
    load_results_2agents as load_results_2agents_multi
)
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
    'save_results',
    'load_results',
    'solve_mechanism_3goods_4synergy',
    'solve_mechanism_3goods_4synergy_iterative',
    'save_results_3goods',
    'load_results_3goods',
    'solve_mechanism_2agents',
    'solve_mechanism_2agents_iterative',
    'save_results_2agents',
    'load_results_2agents',
    'solve_mechanism_multi_agent',
    'solve_mechanism_2agents_multi',
    'solve_mechanism_multi_agent_iterative',
    'solve_mechanism_2agents_iterative_multi',
    'save_results_multi_agent',
    'save_results_2agents_multi',
    'load_results_multi_agent',
    'load_results_2agents_multi',
    'solve_dual',
    'cost_snew',
    'build_cost_matrix',
    'discretize_signed_measure',
    'classify_region',
    'plot_polyhedral_regions',
]

