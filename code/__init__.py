"""
Rochet-Choné型オークション機構設計と最適輸送問題の双対問題
"""

from .utils import (
    beta_pdf, 
    make_tensor_grid_2d, 
    make_tensor_grid_3d, 
    make_tensor_grid_7d, 
    product_beta_density,
    mixture_beta_pdf
)
from .primal_1agent_2goods_with_synergy import (
    solve_mechanism, 
    solve_mechanism_iterative,
    save_results,
    load_results
)
from .primal_1agent_3goods_with_synergy import (
    solve_mechanism_3goods_4synergy, 
    solve_mechanism_3goods_4synergy_iterative,
    save_results_3goods,
    load_results_3goods
)
from .primal_2agents_2goods_with_synergy import (
    solve_mechanism_2agents, 
    solve_mechanism_2agents_iterative,
    save_results_2agents,
    load_results_2agents
)

from .primal_2agents_2goods_no_synergy import (
    solve_mechanism_multi_agent,
    solve_mechanism_2agents as solve_mechanism_2agents_multi,
    solve_mechanism_multi_agent_iterative,
    solve_mechanism_2agents_iterative as solve_mechanism_2agents_iterative_multi,
    save_results_multi_agent,
    save_results_2agents as save_results_2agents_multi,
    load_results_multi_agent,
    load_results_2agents as load_results_2agents_multi
)
from .primal_1agent_3goods_no_synergy import (
    solve_mechanism_single_agent,
    solve_mechanism_single_agent_iterative,
    save_results_single_agent,
    load_results_single_agent
)
from .primal_2agents_2goods_symmetric_no_synergy import (
    solve_mechanism_symmetry_2agents,
    solve_mechanism_symmetry_2agents_iterative,
    save_results_symmetry_2agents,
    load_results_symmetry_2agents
)
from .primal_2agents_2goods_symmetric_with_synergy import (
    solve_mechanism_symmetry_2agents_with_synergy,
    solve_mechanism_symmetry_2agents_with_synergy_iterative,
    save_results_symmetry_2agents_with_synergy,
    load_results_symmetry_2agents_with_synergy
)
from .dual import solve_dual, cost_snew, build_cost_matrix, discretize_signed_measure
from .plot import classify_region, plot_polyhedral_regions

__all__ = [
    'beta_pdf',
    'make_tensor_grid_2d',
    'make_tensor_grid_3d',
    'make_tensor_grid_7d',
    'product_beta_density',
    'mixture_beta_pdf',
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
    'solve_mechanism_single_agent',
    'solve_mechanism_single_agent_iterative',
    'save_results_single_agent',
    'load_results_single_agent',
    'solve_mechanism_symmetry_2agents',
    'solve_mechanism_symmetry_2agents_iterative',
    'save_results_symmetry_2agents',
    'load_results_symmetry_2agents',
    'solve_mechanism_symmetry_2agents_with_synergy',
    'solve_mechanism_symmetry_2agents_with_synergy_iterative',
    'save_results_symmetry_2agents_with_synergy',
    'load_results_symmetry_2agents_with_synergy',
    'solve_dual',
    'cost_snew',
    'build_cost_matrix',
    'discretize_signed_measure',
    'classify_region',
    'plot_polyhedral_regions',
]

