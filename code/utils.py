"""
共通ユーティリティ関数
- Beta分布の密度関数
- グリッド生成（Beta分布・一様分布対応）
- Product-Beta密度計算
"""

from math import gamma
import numpy as np
try:
    from scipy.special import gamma as gamma_func, beta as beta_func
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def beta_pdf(x, a, b):
    """
    Beta(a,b) の密度 f(x) を計算（0<x<1）。
    スカラーまたはNumPy配列に対応。
    """
    is_scalar = not isinstance(x, np.ndarray)
    if is_scalar:
        x = np.array([x])
    
    if HAS_SCIPY:
        # scipy.specialを使った高速なベクトル化実装
        B = beta_func(a, b)
        result = (x ** (a - 1)) * ((1 - x) ** (b - 1)) / B
    else:
        # math.gammaを使った実装（スカラーのみ）
        B = gamma(a) * gamma(b) / gamma(a + b)
        result = (x ** (a - 1)) * ((1 - x) ** (b - 1)) / B
    
    if is_scalar:
        return float(result[0])
    return result


def _normalize_dist_param(param, dim_index=0):
    """
    分布パラメータを正規化する。
    
    後方互換性のため、以下の形式をサポート：
    - (a, b): ベータ分布、範囲[0, 1]
    - {'type': 'beta', 'params': (a, b), 'range': (min, max)}
    - {'type': 'uniform', 'range': (min, max)}
    
    戻り値:
        (dist_type, params, range_min, range_max)
    """
    if isinstance(param, tuple) and len(param) == 2:
        # 従来の形式: (a, b) -> ベータ分布、範囲[0, 1]
        return 'beta', param, 0.0, 1.0
    elif isinstance(param, dict):
        dist_type = param.get('type', 'beta')
        if dist_type == 'beta':
            params = param.get('params', (1.0, 1.0))
            range_min, range_max = param.get('range', (0.0, 1.0))
            return 'beta', params, range_min, range_max
        elif dist_type == 'uniform':
            range_min, range_max = param.get('range', (0.0, 1.0))
            return 'uniform', None, range_min, range_max
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
    else:
        raise ValueError(f"Invalid parameter format at dimension {dim_index}: {param}")


def _generate_coords(n, dist_type, params, range_min, range_max, return_array=False):
    """
    指定された分布に基づいて座標を生成する。
    
    パラメータ:
        return_array: Trueの場合、NumPy配列を返す（ベクトル化用）
    
    戻り値:
        coords: list of float または np.ndarray, グリッド座標
        cell_size: float, セルサイズ
    """
    cell_size = (range_max - range_min) / n
    range_diff = range_max - range_min
    
    if dist_type == 'beta':
        # ベータ分布: [0,1]で生成してから範囲にマッピング
        # (i + 0.5) / n の形式で生成
        u = (np.arange(n) + 0.5) / n
        coords = range_min + u * range_diff
    elif dist_type == 'uniform':
        # 一様分布: 範囲内で等間隔
        coords = range_min + (np.arange(n) + 0.5) * cell_size
    else:
        coords = np.array([])
    
    if not return_array:
        coords = coords.tolist()
    
    return coords, cell_size


def _compute_weight(coord, dist_type, params, range_min, range_max):
    """
    指定された座標での密度を計算する（スカラーまたは配列対応）。
    
    パラメータ:
        coord: float または np.ndarray, 座標値
        dist_type: str, 分布タイプ
        params: tuple, 分布パラメータ
        range_min: float, 範囲の最小値
        range_max: float, 範囲の最大値
    
    戻り値:
        density: float または np.ndarray, 密度値
    """
    is_scalar = not isinstance(coord, np.ndarray)
    if is_scalar:
        coord = np.array([coord])
    
    range_diff = range_max - range_min
    if range_diff == 0:
        range_diff = 1.0
    
    if dist_type == 'beta':
        a, b = params
        # [0,1]に正規化
        u = (coord - range_min) / range_diff
        # 範囲外を0に
        mask = (u >= 0) & (u <= 1)
        density = np.zeros_like(coord)
        if np.any(mask):
            # ベクトル化されたbeta_pdf計算
            u_valid = u[mask]
            # Beta分布の密度を計算（範囲のスケーリングを考慮）
            B = gamma(a) * gamma(b) / gamma(a + b)
            density_valid = ((u_valid ** (a - 1)) * ((1 - u_valid) ** (b - 1))) / B / range_diff
            density[mask] = density_valid
    elif dist_type == 'uniform':
        # 一様分布の密度
        mask = (coord >= range_min) & (coord <= range_max)
        density = np.zeros_like(coord)
        density[mask] = 1.0 / range_diff
    else:
        density = np.zeros_like(coord)
    
    if is_scalar:
        return float(density[0])
    return density


def make_tensor_grid_2d(nx, ny, beta_params=None):
    """
    [0,1]^2 を nx x ny の等間隔グリッドで離散化する。
    
    beta_paramsが指定された場合、Product-Beta分布の重みも計算して返す。
    指定されない場合、グリッド点とセル体積のみを返す。
    
    パラメータ形式（後方互換性あり）:
    - 従来形式: [(a1, b1), (a2, b2)] -> ベータ分布、範囲[0,1]
    - 拡張形式: [
        {'type': 'beta', 'params': (a, b), 'range': (min, max)},
        {'type': 'uniform', 'range': (min, max)},
      ]
    
    戻り値:
        points: list of (x1,x2)
        weights: list of w_j (beta_params指定時) または cell_volume (指定なし)
    """
    if beta_params is None:
        # 重みなし：ベクトル化された座標生成
        xs = (np.arange(nx) + 0.5) / nx
        ys = (np.arange(ny) + 0.5) / ny
        dx, dy = 1.0 / nx, 1.0 / ny
        cell_vol = dx * dy
        
        # np.meshgridで直積を生成し、reshapeで平坦化
        grid = np.stack(np.meshgrid(xs, ys, indexing='ij'), axis=-1)
        points = grid.reshape(-1, 2).tolist()
        # タプルのリストに変換（後方互換性のため）
        points = [tuple(p) for p in points]
        
        return points, cell_vol
    else:
        # 重みあり：分布パラメータを正規化
        assert len(beta_params) == 2, "beta_params は2次元分必要"
        
        dist_configs = [_normalize_dist_param(p, i) for i, p in enumerate(beta_params)]
        
        # 各次元の座標を生成（NumPy配列として）
        xs_arr, dx = _generate_coords(nx, *dist_configs[0], return_array=True)
        ys_arr, dy = _generate_coords(ny, *dist_configs[1], return_array=True)
        
        cell_vol = dx * dy
        
        # np.meshgridで直積を生成
        grid = np.stack(np.meshgrid(xs_arr, ys_arr, indexing='ij'), axis=-1)
        points_arr = grid.reshape(-1, 2)
        
        # ベクトル化された重み計算
        # 各軸の密度を事前計算
        f1_arr = _compute_weight(xs_arr, *dist_configs[0])
        f2_arr = _compute_weight(ys_arr, *dist_configs[1])
        
        # 外積で全組み合わせの重みを計算
        # f1_arr (nx,) × f2_arr (ny,) -> (nx, ny)
        weights_2d = np.outer(f1_arr, f2_arr)
        weights = (weights_2d * cell_vol).flatten()
        
        # NumPyリダクションで正規化
        total_w = weights.sum()
        if total_w > 0:
            weights = weights / total_w
        else:
            # 一様分布の場合は等重み
            weights = np.full(len(points_arr), 1.0 / len(points_arr))
        
        # 後方互換性のためリストに変換（タプルの要素もfloatに変換）
        points = [tuple(float(x) for x in p) for p in points_arr]
        weights = weights.tolist()
        
        return points, weights


def make_tensor_grid_3d(nx, ny, nz, beta_params=None):
    """
    [0,1]^3 を nx x ny x nz の等間隔グリッドで離散化する。
    
    beta_paramsが指定された場合、Product-Beta分布の重みも計算して返す。
    指定されない場合、グリッド点とセル体積のみを返す。
    
    パラメータ形式（後方互換性あり）:
    - 従来形式: [(a1, b1), (a2, b2), (a3, b3)] -> ベータ分布、範囲[0,1]
    - 拡張形式: [
        {'type': 'beta', 'params': (a, b), 'range': (min, max)},
        {'type': 'uniform', 'range': (min, max)},
        ...
      ]
    
    戻り値:
        points: list of (x1,x2,x3)
        weights: list of w_j (beta_params指定時) または cell_volume (指定なし)
    """
    if beta_params is None:
        # 重みなし：ベクトル化された座標生成
        xs = (np.arange(nx) + 0.5) / nx
        ys = (np.arange(ny) + 0.5) / ny
        zs = (np.arange(nz) + 0.5) / nz
        dx, dy, dz = 1.0 / nx, 1.0 / ny, 1.0 / nz
        cell_vol = dx * dy * dz
        
        # np.meshgridで直積を生成し、reshapeで平坦化
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1)
        points = grid.reshape(-1, 3).tolist()
        # タプルのリストに変換（後方互換性のため）
        points = [tuple(p) for p in points]
        
        return points, cell_vol
    else:
        # 重みあり：分布パラメータを正規化
        assert len(beta_params) == 3, "beta_params は3次元分必要"
        
        dist_configs = [_normalize_dist_param(p, i) for i, p in enumerate(beta_params)]
        
        # 各次元の座標を生成（NumPy配列として）
        xs_arr, dx = _generate_coords(nx, *dist_configs[0], return_array=True)
        ys_arr, dy = _generate_coords(ny, *dist_configs[1], return_array=True)
        zs_arr, dz = _generate_coords(nz, *dist_configs[2], return_array=True)
        
        cell_vol = dx * dy * dz
        
        # np.meshgridで直積を生成
        grid = np.stack(np.meshgrid(xs_arr, ys_arr, zs_arr, indexing='ij'), axis=-1)
        points_arr = grid.reshape(-1, 3)
        
        # ベクトル化された重み計算
        # 各軸の密度を事前計算
        f1_arr = _compute_weight(xs_arr, *dist_configs[0])
        f2_arr = _compute_weight(ys_arr, *dist_configs[1])
        f3_arr = _compute_weight(zs_arr, *dist_configs[2])
        
        # 外積で全組み合わせの重みを計算
        # f1_arr (nx,) × f2_arr (ny,) × f3_arr (nz,) -> (nx, ny, nz)
        weights_3d = np.multiply.outer(np.multiply.outer(f1_arr, f2_arr), f3_arr)
        weights = (weights_3d * cell_vol).flatten()
        
        # NumPyリダクションで正規化
        total_w = weights.sum()
        if total_w > 0:
            weights = weights / total_w
        else:
            # 一様分布の場合は等重み
            weights = np.full(len(points_arr), 1.0 / len(points_arr))
        
        # 後方互換性のためリストに変換（タプルの要素もfloatに変換）
        points = [tuple(float(x) for x in p) for p in points_arr]
        weights = weights.tolist()
        
        return points, weights


def make_tensor_grid_7d(n, beta_params=None):
    """
    [0,1]^7 を n^7 の等間隔グリッドで離散化する。
    
    beta_paramsが指定された場合、Product-Beta分布の重みも計算して返す。
    指定されない場合、グリッド点とセル体積のみを返す。
    
    パラメータ形式（後方互換性あり）:
    - 従来形式: [(a1, b1), ..., (a7, b7)] -> ベータ分布、範囲[0,1]
    - 拡張形式: [
        {'type': 'beta', 'params': (a, b), 'range': (min, max)},
        {'type': 'uniform', 'range': (min, max)},
        ...
      ]
    
    戻り値:
        points: list of (x1,x2,x3,x4,x5,x6,x7)
        weights: list of w_j (beta_params指定時) または cell_volume (指定なし)
    """
    if beta_params is None:
        # 重みなし：ベクトル化された座標生成
        coords_1d = (np.arange(n) + 0.5) / n
        cell_vol = (1.0 / n) ** 7
        
        # np.meshgridで7次元の直積を生成
        grid = np.stack(np.meshgrid(*([coords_1d] * 7), indexing='ij'), axis=-1)
        points_arr = grid.reshape(-1, 7)
        
        # 後方互換性のためタプルのリストに変換（タプルの要素もfloatに変換）
        points = [tuple(float(x) for x in p) for p in points_arr]
        
        return points, cell_vol
    else:
        # 重みあり：分布パラメータを正規化
        assert len(beta_params) == 7, "beta_params は7次元分必要"
        
        dist_configs = [_normalize_dist_param(p, i) for i, p in enumerate(beta_params)]
        
        # 各次元の座標を生成（NumPy配列として）
        coords_arr_list = []
        cell_sizes = []
        for dist_config in dist_configs:
            coords_arr, cell_size = _generate_coords(n, *dist_config, return_array=True)
            coords_arr_list.append(coords_arr)
            cell_sizes.append(cell_size)
        
        cell_vol = np.prod(cell_sizes)
        
        # np.meshgridで7次元の直積を生成
        grid = np.stack(np.meshgrid(*coords_arr_list, indexing='ij'), axis=-1)
        points_arr = grid.reshape(-1, 7)
        
        # ベクトル化された重み計算
        # 各軸の密度を事前計算
        density_arrs = [_compute_weight(coords_arr, *dist_config) 
                       for coords_arr, dist_config in zip(coords_arr_list, dist_configs)]
        
        # 外積で全組み合わせの重みを計算
        # 7次元の外積を段階的に計算
        weights_nd = density_arrs[0]
        for density_arr in density_arrs[1:]:
            weights_nd = np.multiply.outer(weights_nd, density_arr)
        
        weights = (weights_nd * cell_vol).flatten()
        
        # NumPyリダクションで正規化
        total_w = weights.sum()
        if total_w > 0:
            weights = weights / total_w
        else:
            # 一様分布の場合は等重み
            weights = np.full(len(points_arr), 1.0 / len(points_arr))
        
        # 後方互換性のためリストに変換（タプルの要素もfloatに変換）
        points = [tuple(float(x) for x in p) for p in points_arr]
        weights = weights.tolist()
        
        return points, weights


def product_beta_density(x, beta_params):
    """任意次元の product-Beta 密度を返す。"""
    assert len(beta_params) == len(x)
    val = 1.0
    for coord, (a, b) in zip(x, beta_params):
        val *= beta_pdf(coord, a, b)
    return val

