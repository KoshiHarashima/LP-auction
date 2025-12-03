"""
Primal問題: Rochet-Choné型LP
2財1シナジー（バンドル{0,1}）の最適オークション機構設計

Cursor.mdに基づく実装:
1人2財1シナジーの問題を、1人3財の問題+シナジーの分配の制約と捉え直す。
2財はa, b、シナジーはαとする。
"""

import pulp
import numpy as np
import os
from datetime import datetime


def solve_mechanism(points, weights, solver=None):
    """
    Rochet–Choné 型の LP を構築して解く（2財1シナジー版）。
    
    Cursor.mdの仕様に基づく:
    - 1人2財1シナジーを、1人3財の問題+シナジーの分配の制約と捉え直す
    - 2財はa, b、シナジーはα

    型数: J = len(points)
    財数: 2財1シナジー（pointsの次元は3である必要がある）

    変数:
        u[j]       : 各グリッド点でのinterim utility (>=0)
        p[l,j]     : 配分確率のベクトル（uの勾配）
                     l=0: 財a, l=1: 財b, l=2: シナジーα (0<=p<=1)

    目的関数:
        max Σ_{1,2,3} Σ_{j} w ((p(j)・x(j)) - u(j))
        = max Σ_j w_j (p[0,j]*x[j,0] + p[1,j]*x[j,1] + p[2,j]*x[j,2] - u_j)
        ※ p(j)・x(j) は3次元ベクトルの内積

    制約:
        - 非負性: u[j] >= 0 (全てのxで)
        - convex: u_i >= u_j + p_j (x_i - x_j) for all i,j
        - 1-Lipshitz: 0 <= p[l,j] <= 1 (l財で点jの傾き)
        - シナジーの配分制約:
            p[2,j] <= p[0,j]  (u_α(x) ≤ u_a(x))
            p[2,j] <= p[1,j]  (u_α(x) ≤ u_b(x))
            p[2,j] >= p[0,j] + p[1,j] - 1  (u_α(x) ≥ u_a(x) + u_b(x) - 1)

    パラメータ:
        points: list of tuples, 各tupleは型空間の点 (x_a, x_b, x_α) - 3次元
        weights: list of floats, 各点の重み w
        solver: PuLPソルバー（必須: Gurobi）

    戻り値:
        (status_string, objective_value, u, p)
        - u: list of floats, 各型での効用
        - p: list of lists, p[l][j] = 財lを型jに割り当てる確率 (l=0,1,2)
    """
    J = len(points)
    assert J == len(weights)
    assert len(points[0]) == 3, "pointsは3次元である必要があります (財a, 財b, シナジーα)"

    # 問題設定
    prob = pulp.LpProblem("RC_2goods_1synergy", pulp.LpMaximize)

    # ========== 変数の定義 ==========
    
    # 変数 u_j (効用、連続変数)
    u = {
        j: pulp.LpVariable(f"u_{j}", lowBound=0.0, cat=pulp.LpContinuous)
        for j in range(J)
    }

    # 変数 p_{l,j} (配分確率のベクトル: l=0,1,2)
    # l=0: 財a, l=1: 財b, l=2: シナジーα
    # 連続変数として定義（[0,1]の任意の値を取れる）
    p = {
        (l, j): pulp.LpVariable(f"p_{l}_{j}", lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
        for l in range(3)
        for j in range(J)
    }

    # ========== 目的関数 ==========
    # Cursor.md: max Σ_{1,2,3} Σ_{j} w ((p(j)・x(j)) - u(j))
    # p(j)・x(j) は3次元ベクトルの内積
    objective = pulp.lpSum(
        weights[j] * (
            p[(0, j)] * points[j][0]  # 財aの価値
            + p[(1, j)] * points[j][1]  # 財bの価値
            + p[(2, j)] * points[j][2]  # シナジーαの価値
            - u[j]
        )
        for j in range(J)
    )
    prob += objective

    # ========== 制約 ==========
    
    # 1. 非負性: u(x) ≥ 0 (全てのxで)
    # 変数の定義で既に lowBound=0.0 として設定済み
    
    # 2. 1-Lipshitz制約: 0 ≤ p_{l}(j) ≤ 1 (l財で点jの傾き)
    # 変数の定義で既に lowBound=0.0, upBound=1.0 として設定済み
    
    # 3. 凸性制約: u_i ≥ u_j + p_j (x_i - x_j) for all i,j
    # Cursor.md: 全てのタイプの組み(i, j)に関して
    for i in range(J):
        x_i = points[i]
        for j in range(J):
            x_j = points[j]
            prob += u[i] >= u[j] + (
                p[(0, j)] * (x_i[0] - x_j[0])  # 財aの項
                + p[(1, j)] * (x_i[1] - x_j[1])  # 財bの項
                + p[(2, j)] * (x_i[2] - x_j[2])  # シナジーαの項
            ), f"convex_{i}_{j}"
    
    # 4. シナジーの配分制約
    # Cursor.md:
    #   u_α(x) ≤ u_a(x)
    #   u_α(x) ≤ u_b(x)
    #   u_α(x) ≥ u_a(x) + u_b(x) - 1
    for j in range(J):
        # 4.1 上界制約: p[2,j] <= p[0,j], p[2,j] <= p[1,j]
        prob += p[(2, j)] <= p[(0, j)], f"synergy_item0_type_{j}_upper"
        prob += p[(2, j)] <= p[(1, j)], f"synergy_item1_type_{j}_upper"
        
        # 4.2 下界制約（inclusion-exclusion）: p[2,j] >= p[0,j] + p[1,j] - 1
        prob += p[(2, j)] >= p[(0, j)] + p[(1, j)] - 1, f"synergy_type_{j}_lower"

    # ========== 解く ==========
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)

    # ========== 結果 ==========
    # NumPy配列に直接変換
    u_sol = np.array([u[j].varValue for j in range(J)], dtype=np.float64)
    p_sol = np.array([[p[(l, j)].varValue for j in range(J)] for l in range(3)], dtype=np.float64)

    return status, obj_val, u_sol, p_sol


def solve_mechanism_iterative(points, weights, grid_sizes, solver=None, max_iter=100, tol=1e-6):
    """
    反復的制約追加アルゴリズムによるRochet-Choné型LPの求解（2財1シナジー版）
    
    アルゴリズム:
    1. 局所的なIC制約（ic-local）のみで初期問題を定義
    2. 最適解を求める
    3. 違反している大局的なIC制約を検出
    4. 違反制約を追加して問題を更新
    5. 違反がなくなるまで反復
    
    パラメータ:
        points: list of tuples, 各tupleは型空間の点 (x_a, x_b, x_α) - 3次元
        weights: list of floats, 各点の重み
        grid_sizes: tuple, 各次元のグリッドサイズ (nx, ny, nz)
        solver: PuLPソルバー（必須: Gurobi）
        max_iter: 最大反復回数
        tol: 違反判定の許容誤差
    
    戻り値:
        (status_string, objective_value, u, p, n_iterations)
    """
    J = len(points)
    assert J == len(weights)
    assert len(points[0]) == 3, "pointsは3次元である必要があります"
    assert len(grid_sizes) == 3, "grid_sizesは3次元である必要があります"
    
    nx, ny, nz = grid_sizes
    assert nx * ny * nz == J, f"grid_sizesの積({nx * ny * nz})がpointsの数({J})と一致しません"
    
    # pointsをNumPy配列に変換（一度だけ、高速化のため）
    points_arr = np.array(points, dtype=np.float64)  # (J, 3)
    
    # グリッドインデックスを計算する関数
    def get_grid_indices(point_idx):
        """点のインデックスからグリッド座標（各次元のインデックス）を取得"""
        k = point_idx % nz
        j = (point_idx // nz) % ny
        i = point_idx // (ny * nz)
        return (i, j, k)
    
    def get_point_idx(grid_indices):
        """グリッド座標から点のインデックスを取得"""
        i, j, k = grid_indices
        return i * (ny * nz) + j * nz + k
    
    def get_neighbors(point_idx):
        """局所的なIC制約に使用する隣接点のインデックスを取得"""
        i, j, k = get_grid_indices(point_idx)
        neighbors = []
        # 各次元で±1の隣接点
        for di in [-1, 1]:
            if 0 <= i + di < nx:
                neighbor_idx = get_point_idx((i + di, j, k))
                neighbors.append(neighbor_idx)
        for dj in [-1, 1]:
            if 0 <= j + dj < ny:
                neighbor_idx = get_point_idx((i, j + dj, k))
                neighbors.append(neighbor_idx)
        for dk in [-1, 1]:
            if 0 <= k + dk < nz:
                neighbor_idx = get_point_idx((i, j, k + dk))
                neighbors.append(neighbor_idx)
        return neighbors
    
    # ========== 問題の初期化 ==========
    prob = pulp.LpProblem("RC_iterative_2goods_1synergy", pulp.LpMaximize)
    
    # 変数の定義
    u = {j: pulp.LpVariable(f"u_{j}", lowBound=0.0, cat=pulp.LpContinuous) for j in range(J)}
    # 配分確率は連続変数（[0,1]の任意の値を取れる）
    p = {
        (l, j): pulp.LpVariable(f"p_{l}_{j}", lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
        for l in range(3)
        for j in range(J)
    }
    
    # 目的関数
    objective = pulp.lpSum(
        weights[j] * (
            p[(0, j)] * points[j][0]
            + p[(1, j)] * points[j][1]
            + p[(2, j)] * points[j][2]
            - u[j]
        )
        for j in range(J)
    )
    prob += objective
    
    # シナジーの配分制約
    for j in range(J):
        prob += p[(2, j)] <= p[(0, j)], f"synergy_item0_type_{j}_upper"
        prob += p[(2, j)] <= p[(1, j)], f"synergy_item1_type_{j}_upper"
        prob += p[(2, j)] >= p[(0, j)] + p[(1, j)] - 1, f"synergy_type_{j}_lower"
    
    # 局所的なIC制約（隣接点のみ）
    local_constraints = set()
    for i in range(J):
        neighbors = get_neighbors(i)
        x_i = points_arr[i]  # NumPy配列から直接取得
        for j in neighbors:
            x_j = points_arr[j]  # NumPy配列から直接取得
            prob += u[i] >= u[j] + (
                p[(0, j)] * (x_i[0] - x_j[0])
                + p[(1, j)] * (x_i[1] - x_j[1])
                + p[(2, j)] * (x_i[2] - x_j[2])
            ), f"convex_local_{i}_{j}"
            local_constraints.add((i, j))
    
    # 追加された制約を記録（重複追加を防ぐ）
    added_constraints = local_constraints.copy()
    
    # ========== 反復ループ ==========
    for iteration in range(max_iter):
        # 最適解を求める
        prob.solve(solver)
        
        if prob.status != pulp.LpStatusOptimal:
            status = pulp.LpStatus[prob.status]
            return status, None, None, None, iteration
        
        # 解を取得（NumPy配列に直接変換、高速化）
        u_sol = np.array([u[j].varValue for j in range(J)], dtype=np.float64)
        p_sol = np.array([[p[(l, j)].varValue for j in range(J)] for l in range(3)], dtype=np.float64)  # (3, J)
        # points_arrは既に定義済み（最初に一度だけ変換）
        
        # 違反している制約を検出（ベクトル化で高速化）
        violations = []
        
        # 既に追加済みの制約をマスクとして作成
        # ベクトル演算のためにNumPy配列として作成（高速化）
        constraint_mask = np.zeros((J, J), dtype=bool)
        if added_constraints:  # 空でない場合のみ処理
            # リストに変換してからNumPy配列でインデックス設定（高速化）
            constraint_pairs = np.array(list(added_constraints), dtype=np.int32)
            if len(constraint_pairs) > 0:
                constraint_mask[constraint_pairs[:, 0], constraint_pairs[:, 1]] = True
        
        # 全ての(i,j)の組み合わせに対してベクトル演算
        # u_diff[i,j] = u_sol[i] - u_sol[j]
        u_diff = u_sol[:, np.newaxis] - u_sol[np.newaxis, :]  # (J, J)
        
        # points_diff[i,j] = points[i] - points[j] (形状: (J, J, 3))
        points_diff = points_arr[:, np.newaxis, :] - points_arr[np.newaxis, :, :]  # (J, J, 3)
        
        # inner_product[i,j] = p_sol[0,j] * (x_i[0] - x_j[0]) + p_sol[1,j] * (x_i[1] - x_j[1]) + p_sol[2,j] * (x_i[2] - x_j[2])
        # p_solの形状は(3, J)、points_diffの形状は(J, J, 3)
        # p_sol[l, j]を(J, J)にブロードキャスト: p_sol[l, :][np.newaxis, :] は (1, J) -> (J, J)にブロードキャスト
        inner_product = (p_sol[0, :][np.newaxis, :] * points_diff[:, :, 0] +
                         p_sol[1, :][np.newaxis, :] * points_diff[:, :, 1] +
                         p_sol[2, :][np.newaxis, :] * points_diff[:, :, 2])  # 結果の形状: (J, J)
        
        # 違反チェック: u_i - u_j < p_j・(x_i - x_j) - tol
        violation_mask = (u_diff < inner_product - tol) & (~constraint_mask)
        
        # 違反している(i,j)のペアを取得
        violation_indices = np.where(violation_mask)
        violations = list(zip(violation_indices[0], violation_indices[1]))
        
        # 違反がなければ終了
        if not violations:
            status = pulp.LpStatus[prob.status]
            obj_val = pulp.value(prob.objective)
            return status, obj_val, u_sol, p_sol, iteration + 1
        
        # 違反している制約を追加（バッチ処理で高速化）
        # NumPy配列から直接取得して高速化
        for i, j in violations:
            x_i = points_arr[i]  # NumPy配列から直接取得
            x_j = points_arr[j]  # NumPy配列から直接取得
            constraint = u[i] >= u[j] + (
                p[(0, j)] * (x_i[0] - x_j[0])
                + p[(1, j)] * (x_i[1] - x_j[1])
                + p[(2, j)] * (x_i[2] - x_j[2])
            )
            prob += constraint, f"convex_{i}_{j}_iter{iteration}"
            added_constraints.add((i, j))
        
        print(f"Iteration {iteration + 1}: {len(violations)} violations found, added {len(violations)} constraints")
    
    # 最大反復回数に達した場合
    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)
    u_sol = np.array([u[j].varValue for j in range(J)], dtype=np.float64)
    p_sol = np.array([[p[(l, j)].varValue for j in range(J)] for l in range(3)], dtype=np.float64)
    return status, obj_val, u_sol, p_sol, max_iter


def save_results(points, weights, u_sol, p_sol, obj_val, status,
                 grid_sizes=None, n_iter=None, filename=None, data_dir="data"):
    """
    2財1シナジーの結果をNumPy形式で保存する。
    
    パラメータ:
        points: 型空間の点 (list or np.ndarray)
        weights: 各点の重み (list or np.ndarray)
        u_sol: 効用 (np.ndarray, shape: (J,))
        p_sol: 配分確率 (np.ndarray, shape: (3, J))
        obj_val: 目的関数値 (float)
        status: LPステータス (str)
        grid_sizes: グリッドサイズ (tuple, optional)
        n_iter: 反復回数 (int, optional)
        filename: 保存ファイル名 (str, optional)
        data_dir: データ保存ディレクトリ (str, default: "data")
    
    戻り値:
        filepath: 保存されたファイルのパス (str)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_2goods_1synergy_{timestamp}.npz"
    
    filepath = os.path.join(data_dir, filename)
    
    points_arr = np.array(points, dtype=np.float64)
    weights_arr = np.array(weights, dtype=np.float64)
    u_sol_arr = np.array(u_sol, dtype=np.float64)
    p_sol_arr = np.array(p_sol, dtype=np.float64)
    
    save_dict = {
        'points': points_arr,
        'weights': weights_arr,
        'u_sol': u_sol_arr,
        'p_sol': p_sol_arr,
        'obj_val': np.array([obj_val], dtype=np.float64) if obj_val is not None else np.array([np.nan], dtype=np.float64),
        'J': np.array([len(points)], dtype=np.int32),
    }
    
    if grid_sizes is not None:
        save_dict['grid_sizes'] = np.array(grid_sizes, dtype=np.int32)
    if n_iter is not None:
        save_dict['n_iter'] = np.array([n_iter], dtype=np.int32)
    
    metadata_filepath = filepath.replace('.npz', '_metadata.txt')
    with open(metadata_filepath, 'w') as f:
        f.write(f"status: {status}\n")
        f.write(f"obj_val: {obj_val}\n")
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
        if grid_sizes is not None:
            f.write(f"grid_sizes: {grid_sizes}\n")
        if n_iter is not None:
            f.write(f"n_iter: {n_iter}\n")
    
    np.savez_compressed(filepath, **save_dict)
    print(f"Results saved to: {filepath}")
    return filepath


def load_results(filepath):
    """
    保存された結果を読み込む。
    
    パラメータ:
        filepath: 保存されたファイルのパス (str)
    
    戻り値:
        dict: 読み込んだデータの辞書
    """
    data = np.load(filepath, allow_pickle=True)
    
    result = {
        'points': data['points'],
        'weights': data['weights'],
        'u_sol': data['u_sol'],
        'p_sol': data['p_sol'],
        'obj_val': float(data['obj_val'][0]) if not np.isnan(data['obj_val'][0]) else None,
        'J': int(data['J'][0]),
    }
    
    if 'grid_sizes' in data:
        result['grid_sizes'] = tuple(data['grid_sizes'])
    if 'n_iter' in data:
        result['n_iter'] = int(data['n_iter'][0])
    
    metadata_filepath = filepath.replace('.npz', '_metadata.txt')
    if os.path.exists(metadata_filepath):
        with open(metadata_filepath, 'r') as f:
            for line in f:
                if line.startswith('status:'):
                    result['status'] = line.split(':', 1)[1].strip()
                elif line.startswith('timestamp:'):
                    result['timestamp'] = line.split(':', 1)[1].strip()
    
    return result
