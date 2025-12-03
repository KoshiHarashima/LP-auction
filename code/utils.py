"""
共通ユーティリティ関数
- Beta分布の密度関数
- グリッド生成（Beta分布・一様分布対応）
- Product-Beta密度計算
"""

from math import gamma


def beta_pdf(x, a, b):
    """Beta(a,b) の密度 f(x) を計算（0<x<1）。"""
    B = gamma(a) * gamma(b) / gamma(a + b)
    return (x ** (a - 1)) * ((1 - x) ** (b - 1)) / B


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


def _generate_coords(n, dist_type, params, range_min, range_max):
    """
    指定された分布に基づいて座標を生成する。
    
    戻り値:
        coords: list of float, グリッド座標
        cell_size: float, セルサイズ
    """
    cell_size = (range_max - range_min) / n
    coords = []
    
    if dist_type == 'beta':
        # ベータ分布: [0,1]で生成してから範囲にマッピング
        a, b = params
        for i in range(n):
            # [0,1]での中央点
            u = (i + 0.5) / n
            # 範囲にマッピング
            x = range_min + u * (range_max - range_min)
            coords.append(x)
    elif dist_type == 'uniform':
        # 一様分布: 範囲内で等間隔
        for i in range(n):
            x = range_min + (i + 0.5) * cell_size
            coords.append(x)
    
    return coords, cell_size


def _compute_weight(coord, dist_type, params, range_min, range_max):
    """
    指定された座標での密度を計算する。
    """
    if dist_type == 'beta':
        a, b = params
        # [0,1]に正規化
        u = (coord - range_min) / (range_max - range_min) if range_max != range_min else 0.5
        if u < 0 or u > 1:
            return 0.0
        # ベータ分布の密度を計算（範囲のスケーリングを考慮）
        density = beta_pdf(u, a, b) / (range_max - range_min) if range_max != range_min else beta_pdf(u, a, b)
        return density
    elif dist_type == 'uniform':
        # 一様分布の密度
        if coord < range_min or coord > range_max:
            return 0.0
        return 1.0 / (range_max - range_min) if range_max != range_min else 1.0
    return 0.0


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
        # 重みなし：従来通り[0,1]で生成
        xs = [(i + 0.5) / nx for i in range(nx)]
        ys = [(j + 0.5) / ny for j in range(ny)]
        dx, dy = 1.0 / nx, 1.0 / ny
        cell_vol = dx * dy
        points = []
        for x1 in xs:
            for x2 in ys:
                points.append((x1, x2))
        return points, cell_vol
    else:
        # 重みあり：分布パラメータを正規化
        assert len(beta_params) == 2, "beta_params は2次元分必要"
        
        dist_configs = [_normalize_dist_param(p, i) for i, p in enumerate(beta_params)]
        
        # 各次元の座標を生成
        xs, dx = _generate_coords(nx, *dist_configs[0])
        ys, dy = _generate_coords(ny, *dist_configs[1])
        
        cell_vol = dx * dy
        points = []
        weights = []
        
        for x1 in xs:
            f1 = _compute_weight(x1, *dist_configs[0])
            for x2 in ys:
                f2 = _compute_weight(x2, *dist_configs[1])
                points.append((x1, x2))
                w = f1 * f2 * cell_vol
                weights.append(w)
        
        # 数値誤差を補正して重みが1になるよう正規化
        total_w = sum(weights)
        if total_w > 0:
            weights = [w / total_w for w in weights]
        else:
            # 一様分布の場合は等重み
            weights = [1.0 / len(points)] * len(points)
        
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
        # 重みなし：従来通り[0,1]で生成
        xs = [(i + 0.5) / nx for i in range(nx)]
        ys = [(j + 0.5) / ny for j in range(ny)]
        zs = [(k + 0.5) / nz for k in range(nz)]
        dx, dy, dz = 1.0 / nx, 1.0 / ny, 1.0 / nz
        cell_vol = dx * dy * dz
        points = []
        for x1 in xs:
            for x2 in ys:
                for x3 in zs:
                    points.append((x1, x2, x3))
        return points, cell_vol
    else:
        # 重みあり：分布パラメータを正規化
        assert len(beta_params) == 3, "beta_params は3次元分必要"
        
        dist_configs = [_normalize_dist_param(p, i) for i, p in enumerate(beta_params)]
        
        # 各次元の座標を生成
        xs, dx = _generate_coords(nx, *dist_configs[0])
        ys, dy = _generate_coords(ny, *dist_configs[1])
        zs, dz = _generate_coords(nz, *dist_configs[2])
        
        cell_vol = dx * dy * dz
        points = []
        weights = []
        
        for x1 in xs:
            f1 = _compute_weight(x1, *dist_configs[0])
            for x2 in ys:
                f2 = _compute_weight(x2, *dist_configs[1])
                for x3 in zs:
                    f3 = _compute_weight(x3, *dist_configs[2])
                    points.append((x1, x2, x3))
                    w = f1 * f2 * f3 * cell_vol
                    weights.append(w)
        
        # 数値誤差を補正して重みが1になるよう正規化
        total_w = sum(weights)
        if total_w > 0:
            weights = [w / total_w for w in weights]
        else:
            # 一様分布の場合は等重み
            weights = [1.0 / len(points)] * len(points)
        
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
        # 重みなし：従来通り[0,1]で生成
        coords = [[(i + 0.5) / n for i in range(n)] for _ in range(7)]
        cell_vol = (1.0 / n) ** 7
        points = []
        from itertools import product
        for point in product(*coords):
            points.append(point)
        return points, cell_vol
    else:
        # 重みあり：分布パラメータを正規化
        assert len(beta_params) == 7, "beta_params は7次元分必要"
        
        dist_configs = [_normalize_dist_param(p, i) for i, p in enumerate(beta_params)]
        
        # 各次元の座標を生成
        coords_list = []
        cell_sizes = []
        for dist_config in dist_configs:
            coords, cell_size = _generate_coords(n, *dist_config)
            coords_list.append(coords)
            cell_sizes.append(cell_size)
        
        cell_vol = 1.0
        for cs in cell_sizes:
            cell_vol *= cs
        
        points = []
        weights = []
        from itertools import product
        
        for point in product(*coords_list):
            points.append(point)
            w = cell_vol
            for coord, dist_config in zip(point, dist_configs):
                w *= _compute_weight(coord, *dist_config)
            weights.append(w)
        
        # 数値誤差を補正して重みが1になるよう正規化
        total_w = sum(weights)
        if total_w > 0:
            weights = [w / total_w for w in weights]
        else:
            # 一様分布の場合は等重み
            weights = [1.0 / len(points)] * len(points)
        
        return points, weights


def product_beta_density(x, beta_params):
    """任意次元の product-Beta 密度を返す。"""
    assert len(beta_params) == len(x)
    val = 1.0
    for coord, (a, b) in zip(x, beta_params):
        val *= beta_pdf(coord, a, b)
    return val

