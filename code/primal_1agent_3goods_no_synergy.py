"""
Primal問題: Rochet-Choné型LP
1人3財の最適オークション機構設計（シナジーなし、Implementability制約なし）

仕様:
- 参加者1人: 型 (x_a, x_b, x_c) ∈ [0,1]³
- シナジーなし（3財は独立）
- Implementability制約なし
"""

import pulp
import numpy as np
import os
from datetime import datetime


def solve_mechanism_single_agent(points, weights, solver=None):
    """
    Rochet–Choné 型の LP を構築して解く（1人3財版、シナジーなし）。
    
    仕様:
    - 参加者1人: 型 (x_a, x_b, x_c) ∈ [0,1]³
    - シナジーなし（3財は独立）

    型数: 
        J = len(points) (参加者の型数)
    財数: 3財（pointsの次元は3である必要がある）

    変数:
        u[j]   : 参加者が型jのときのinterim utility (>=0)
        p[l, j]: 参加者への配分確率のベクトル（uの勾配）
                 l=0: 財a, l=1: 財b, l=2: 財c (0<=p<=1)

    目的関数:
        max Σ_j w[j] * (p(j)・x(j) - u(j))
        = max Σ_j w[j] * (
            p[0,j]*x[j,0] + p[1,j]*x[j,1] + p[2,j]*x[j,2] - u[j]
        )

    制約:
        - 非負性: u[j] >= 0 (全てのjで)
        - IC: u[i] >= u[j] + p(j)・(x(i) - x(j)) for all i, j
        - 1-Lipschitz: 0 <= p[l,j] <= 1
        - IR: u(x) >= 0 for all x
              （実装では、変数の定義時に lowBound=0.0 として設定し、IR制約は明示的に追加しない）

    パラメータ:
        points: list of tuples, 参加者の型空間の点 [(x_a, x_b, x_c), ...] - 3次元
        weights: list of floats, 参加者の各点の重み [w, ...]
        solver: PuLPソルバー（必須: Gurobi）

    戻り値:
        (status_string, objective_value, u, p)
        - u: np.ndarray, shape (J,), u[j] = 参加者が型jのときの効用
        - p: np.ndarray, shape (3, J), p[l, j] = 参加者への財lの配分確率 (l=0,1,2)
    """
    J = len(points)
    
    # pointsとweightsをNumPy配列に変換（一度だけ、高速化のため）
    points_arr = np.asarray(points, dtype=np.float64)  # (J, 3)
    weights_arr = np.asarray(weights, dtype=np.float64)  # (J,)
    
    # 差分行列を一括計算（ループ外で一度だけ）
    # points_diff[i,j] = points[i] - points[j] (形状: (J, J, 3))
    points_diff = points_arr[:, None, :] - points_arr[None, :, :]  # (J, J, 3)
    
    # 問題設定
    prob = pulp.LpProblem("RC_single_agent_3goods", pulp.LpMaximize)

    # ========== 変数の定義 ==========
    # 変数 u[j] (参加者の効用、連続変数)
    u = {
        j: pulp.LpVariable(f"u_{j}", lowBound=0.0, cat=pulp.LpContinuous)
        for j in range(J)
    }
    
    # 変数 p[l, j] (参加者の配分確率のベクトル: l=0,1,2)
    # l=0: 財a, l=1: 財b, l=2: 財c
    p = {
        (l, j): pulp.LpVariable(f"p_{l}_{j}", 
                               lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
        for l in range(3)
        for j in range(J)
    }

    # ========== 目的関数 ==========
    # max Σ_j w[j] * (p(j)・x(j) - u(j))
    # 配列アクセスを使用して高速化
    objective = pulp.lpSum(
        weights_arr[j] * (
            p[(0, j)] * points_arr[j, 0]  # 財aの価値
            + p[(1, j)] * points_arr[j, 1]  # 財bの価値
            + p[(2, j)] * points_arr[j, 2]  # 財cの価値
            - u[j]
        )
        for j in range(J)
    )
    prob += objective

    # ========== 制約 ==========
    
    # 1. 非負性: u[j] >= 0 (全てのjで)
    # 変数の定義で既に lowBound=0.0 として設定済み
    
    # 2. 1-Lipschitz制約: 0 ≤ p[l,j] ≤ 1
    # 変数の定義で既に lowBound=0.0, upBound=1.0 として設定済み
    
    # 3. IC制約
    # 差分行列をキャッシュから参照して高速化
    for i in range(J):
        for j in range(J):
            # IC制約: u[i] >= u[j] + p(j)・(x(i) - x(j))
            # 差分行列から直接参照（points_diff[i, j, :]）
            prob += u[i] >= u[j] + (
                p[(0, j)] * points_diff[i, j, 0]  # 財aの項
                + p[(1, j)] * points_diff[i, j, 1]  # 財bの項
                + p[(2, j)] * points_diff[i, j, 2]  # 財cの項
            ), f"ic_{i}_{j}"

    # ========== 解く ==========
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)

    # ========== 結果 ==========
    # NumPy配列に変換（np.fromiterを使用してPythonループを削減）
    u_sol = np.fromiter((u[j].varValue for j in range(J)), dtype=np.float64, count=J)
    p_sol = np.array([
        np.fromiter((p[(l, j)].varValue for j in range(J)), dtype=np.float64, count=J)
        for l in range(3)
    ], dtype=np.float64)

    return status, obj_val, u_sol, p_sol


def solve_mechanism_single_agent_iterative(points, weights, grid_sizes, 
                                           solver=None, max_iter=100, tol=1e-6):
    """
    反復的制約追加アルゴリズムによるRochet-Choné型LPの求解（1人3財版、シナジーなし）。
    
    アルゴリズム:
    1. 局所的なIC制約（ic-local）のみで初期問題を定義
    2. 最適解を求める
    3. 違反している大局的なIC制約を検出
    4. 違反制約を追加して問題を更新
    5. 違反がなくなるまで反復
    
    パラメータ:
        points: list of tuples, 参加者の型空間の点 [(x_a, x_b, x_c), ...] - 3次元
        weights: list of floats, 参加者の各点の重み
        grid_sizes: tuple, 各次元のグリッドサイズ (nx, ny, nz)
        solver: PuLPソルバー（必須: Gurobi）
        max_iter: 最大反復回数
        tol: 違反判定の許容誤差
    
    戻り値:
        (status_string, objective_value, u, p, n_iterations)
    """
    J = len(points)
    nx, ny, nz = grid_sizes
    
    # pointsとweightsをNumPy配列に変換（一度だけ、高速化のため）
    points_arr = np.asarray(points, dtype=np.float64)  # (J, 3)
    weights_arr = np.asarray(weights, dtype=np.float64)  # (J,)
    
    # 差分行列をループ外で一度だけ計算（反復中は不変）
    # points_diff[i,j] = points[i] - points[j] (形状: (J, J, 3))
    points_diff = points_arr[:, None, :] - points_arr[None, :, :]  # (J, J, 3)
    
    # グリッドインデックスを計算する関数
    def get_grid_indices(point_idx):
        """点のインデックスからグリッド座標を取得"""
        k = point_idx % nz
        j = (point_idx // nz) % ny
        i = point_idx // (ny * nz)
        return (i, j, k)
    
    def get_point_idx(grid_indices):
        """グリッド座標から点のインデックスを取得"""
        i, j, k = grid_indices
        return i * ny * nz + j * nz + k
    
    def get_neighbors(point_idx):
        """局所的なIC制約に使用する隣接点のインデックスを取得"""
        i, j, k = get_grid_indices(point_idx)
        neighbors = []
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
    prob = pulp.LpProblem("RC_iterative_single_agent_3goods", pulp.LpMaximize)
    
    # 変数の定義
    u = {
        j: pulp.LpVariable(f"u_{j}", lowBound=0.0, cat=pulp.LpContinuous)
        for j in range(J)
    }
    
    p = {
        (l, j): pulp.LpVariable(f"p_{l}_{j}", 
                               lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
        for l in range(3)
        for j in range(J)
    }
    
    # 目的関数（配列アクセスを使用して高速化）
    objective = pulp.lpSum(
        weights_arr[j] * (
            p[(0, j)] * points_arr[j, 0]
            + p[(1, j)] * points_arr[j, 1]
            + p[(2, j)] * points_arr[j, 2]
            - u[j]
        )
        for j in range(J)
    )
    prob += objective
    
    # 局所的なIC制約
    local_constraints = set()
    for i in range(J):
        neighbors_i = get_neighbors(i)
        for j in neighbors_i:
            # 差分行列から直接参照（points_diff[i, j, :]）
            prob += u[i] >= u[j] + (
                p[(0, j)] * points_diff[i, j, 0]
                + p[(1, j)] * points_diff[i, j, 1]
                + p[(2, j)] * points_diff[i, j, 2]
            ), f"ic_local_{i}_{j}"
            local_constraints.add((i, j))
    
    # 追加された制約を記録（重複追加を防ぐ）
    added_constraints = local_constraints.copy()
    
    # 制約マスクをループ外で初期化（違反分だけ更新する方式）
    constraint_mask = np.zeros((J, J), dtype=bool)
    if added_constraints:
        constraint_pairs = np.array(list(added_constraints), dtype=np.int32)
        if len(constraint_pairs) > 0:
            constraint_mask[constraint_pairs[:, 0], constraint_pairs[:, 1]] = True
    
    # ========== 反復ループ ==========
    for iteration in range(max_iter):
        # 最適解を求める
        prob.solve(solver)
        
        if prob.status != pulp.LpStatusOptimal:
            status = pulp.LpStatus[prob.status]
            return status, None, None, None, iteration
        
        # 解を取得（NumPy配列に直接変換、高速化）
        # np.fromiterを使用してPythonループを削減
        u_sol = np.fromiter((u[j].varValue for j in range(J)), dtype=np.float64, count=J)
        p_sol = np.array([
            np.fromiter((p[(l, j)].varValue for j in range(J)), dtype=np.float64, count=J)
            for l in range(3)
        ], dtype=np.float64)
        
        # 違反している制約を検出（ベクトル化で高速化）
        # 全ての(i,j)の組み合わせに対してベクトル演算
        # u_diff[i,j] = u_sol[i] - u_sol[j]
        u_diff = u_sol[:, np.newaxis] - u_sol[np.newaxis, :]  # (J, J)
        
        # points_diffは既にループ外で計算済み（再計算不要）
        
        # inner_productをeinsumで計算（高速化）
        # inner_product[i,j] = Σ_l p_sol[l,j] * points_diff[i,j,l]
        inner_product = np.einsum('lj,ijl->ij', p_sol, points_diff)  # 結果の形状: (J, J)
        
        # 違反チェック: u_i - u_j < p_j・(x_i - x_j) - tol
        violation_mask = (u_diff < inner_product - tol) & (~constraint_mask)
        
        # 違反している(i,j)のペアを取得（np.argwhereで直接取得）
        violation_pairs = np.argwhere(violation_mask)  # (N_violations, 2)
        violations = [(int(i), int(j)) for i, j in violation_pairs]
        
        # 違反している制約を追加（バッチ処理で高速化）
        # 差分行列から直接参照して高速化
        for i, j in violations:
            # 差分行列から直接参照（points_diff[i, j, :]）
            constraint = u[i] >= u[j] + (
                p[(0, j)] * points_diff[i, j, 0]
                + p[(1, j)] * points_diff[i, j, 1]
                + p[(2, j)] * points_diff[i, j, 2]
            )
            prob += constraint, f"ic_{i}_{j}_iter{iteration}"
            added_constraints.add((i, j))
            # 制約マスクを更新（違反分だけTrueに設定）
            constraint_mask[i, j] = True
        
        # 違反がなければ終了
        if len(violations) == 0:
            status = pulp.LpStatus[prob.status]
            obj_val = pulp.value(prob.objective)
            return status, obj_val, u_sol, p_sol, iteration + 1
        
        print(f"Iteration {iteration + 1}: {len(violations)} violations found, added {len(violations)} constraints")
    
    # 最大反復回数に達した場合
    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)
    # np.fromiterを使用してPythonループを削減
    u_sol = np.fromiter((u[j].varValue for j in range(J)), dtype=np.float64, count=J)
    p_sol = np.array([
        np.fromiter((p[(l, j)].varValue for j in range(J)), dtype=np.float64, count=J)
        for l in range(3)
    ], dtype=np.float64)
    return status, obj_val, u_sol, p_sol, max_iter


def save_results_single_agent(
    points, weights,
    u, p,
    obj_val, status,
    grid_sizes=None, n_iter=None,
    filename=None, data_dir="data"
):
    """
    1人3財の結果をNumPy形式で保存する。
    
    パラメータ:
        points: 参加者の型空間の点 (list or np.ndarray)
        weights: 参加者の各点の重み (list or np.ndarray)
        u: 参加者の効用 (np.ndarray, shape: (J,))
        p: 参加者の配分確率 (np.ndarray, shape: (3, J))
        obj_val: 目的関数値 (float)
        status: LPステータス (str)
        grid_sizes: グリッドサイズ (tuple, optional)
        n_iter: 反復回数 (int, optional)
        filename: 保存ファイル名 (str, optional, 指定しない場合は自動生成)
        data_dir: データ保存ディレクトリ (str, default: "data")
    
    戻り値:
        filepath: 保存されたファイルのパス (str)
    """
    # データディレクトリを作成
    os.makedirs(data_dir, exist_ok=True)
    
    # ファイル名を生成
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_single_agent_3goods_{timestamp}.npz"
    
    filepath = os.path.join(data_dir, filename)
    
    # NumPy配列に変換
    points_arr = np.array(points, dtype=np.float64)
    weights_arr = np.array(weights, dtype=np.float64)
    u_arr = np.array(u, dtype=np.float64)
    p_arr = np.array(p, dtype=np.float64)
    
    # NumPy形式で保存（メタデータも含める）
    save_dict = {
        'obj_val': np.array([obj_val], dtype=np.float64) if obj_val is not None else np.array([np.nan], dtype=np.float64),
        'points': points_arr,
        'weights': weights_arr,
        'u': u_arr,
        'p': p_arr,
        'J': np.array([len(points)], dtype=np.int32),
    }
    
    if grid_sizes is not None:
        save_dict['grid_sizes'] = np.array(grid_sizes, dtype=np.int32)
    if n_iter is not None:
        save_dict['n_iter'] = np.array([n_iter], dtype=np.int32)
    
    # メタデータファイルも保存（文字列情報用）
    metadata_filepath = filepath.replace('.npz', '_metadata.txt')
    with open(metadata_filepath, 'w') as f:
        f.write(f"status: {status}\n")
        f.write(f"obj_val: {obj_val}\n")
        f.write(f"J: {len(points)}\n")
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
        if grid_sizes is not None:
            f.write(f"grid_sizes: {grid_sizes}\n")
        if n_iter is not None:
            f.write(f"n_iter: {n_iter}\n")
    
    np.savez_compressed(filepath, **save_dict)
    
    print(f"Results saved to: {filepath}")
    return filepath


def load_results_single_agent(filepath):
    """
    保存された結果を読み込む。
    
    パラメータ:
        filepath: 保存されたファイルのパス (str)
    
    戻り値:
        dict: 読み込んだデータの辞書
    """
    data = np.load(filepath, allow_pickle=True)
    
    result = {
        'obj_val': float(data['obj_val'][0]) if not np.isnan(data['obj_val'][0]) else None,
        'points': data['points'],
        'weights': data['weights'],
        'u': data['u'],
        'p': data['p'],
        'J': int(data['J'][0]),
    }
    
    if 'grid_sizes' in data:
        result['grid_sizes'] = tuple(data['grid_sizes'])
    if 'n_iter' in data:
        result['n_iter'] = int(data['n_iter'][0])
    
    # メタデータファイルから文字列情報を読み込む
    metadata_filepath = filepath.replace('.npz', '_metadata.txt')
    if os.path.exists(metadata_filepath):
        with open(metadata_filepath, 'r') as f:
            for line in f:
                if line.startswith('status:'):
                    result['status'] = line.split(':', 1)[1].strip()
                elif line.startswith('timestamp:'):
                    result['timestamp'] = line.split(':', 1)[1].strip()
    
    return result

