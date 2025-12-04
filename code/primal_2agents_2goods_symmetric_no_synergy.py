"""
Primal問題: Rochet-Choné型LP
複数エージェント2財の最適オークション機構設計（シナジーなし、対称性制約あり）

primal_2agents_2goods_no_synergy.pyを改良:
- シナジーを削除（2財のみ）
- 対称性により、u1のみの設計で完結（u2は不要）
- 目的関数: 2 Σ_j w_j (p(j)・x_j - u(j))
- IC制約: u(i) >= u(j) + p(j)・(x(i) - x(j)) ∀i, j
- Feasibility制約: 0 ≤ 2p[l,j] ≤ 1 → 0 ≤ p[l,j] ≤ 0.5

対称性制約:
- 対称性により、参加者1の設計のみで完結（J^4からJ^2にパラメータ削減）
- u1のみを学習し、u2は不要
"""

import pulp
import numpy as np
import os
from datetime import datetime
from itertools import product


def solve_mechanism_symmetry_2agents(points1, weights1, solver=None):
    """
    Rochet–Choné 型の LP を構築して解く（2人2財版、シナジーなし、対称性制約あり）。
    
    仕様:
    - 参加者1: 型 (x₁_a, x₁_b) ∈ [0,1]²
    - 対称性により、u1のみの設計で完結（u2は不要）

    型数: 
        J1 = len(points1)
    財数: 2財（pointsの次元は2である必要がある）

    変数:
        u[j]   : 参加者1が型jのときの効用
        p[l, j]: 参加者1が型jのときの財lの配分確率
                 l=0: 財a, l=1: 財b

    目的関数:
        max 2 Σ_j w_j (p(j)・x_j - u(j))

    制約:
        - 非負性: u[j] >= 0
        - IC: u(i) >= u(j) + p(j)・(x(i) - x(j)) ∀i, j
        - Feasibility制約: 0 ≤ 2p[l,j] ≤ 1 → 0 ≤ p[l,j] ≤ 0.5

    パラメータ:
        points1: list of tuples, 参加者1の型空間の点 [(x₁_a, x₁_b), ...] - 2次元
        weights1: list of floats, 参加者1の各点の重み w₁
        solver: PuLPソルバー（必須: Gurobi）
    
    戻り値:
        (status_string, objective_value, u1, u2, p1, p2)
        - u1: np.ndarray, shape (J1,), u1[j] = 参加者1が型jのときの効用
        - u2: np.ndarray, shape (J1,), u2 = u1（後方互換性のため）
        - p1: np.ndarray, shape (2, J1), p1[l, j] = 参加者1が型jのときの財lの配分確率 (l=0,1)
        - p2: np.ndarray, shape (2, J1), p2 = p1（後方互換性のため）
    """
    
    J1 = len(points1)
    
    # pointsとweightsをNumPy配列に変換
    points1_arr = np.asarray(points1, dtype=np.float64)  # (J1, 2)
    weights1_arr = np.asarray(weights1, dtype=np.float64)  # (J1,)

    # 差分行列を一括計算（ループ外で一度だけ）
    points1_diff = points1_arr[:, None, :] - points1_arr[None, :, :]  # (J1, J1, 2)
    
    # 問題設定
    prob = pulp.LpProblem("RC_symmetry_2agents_2goods", pulp.LpMaximize)

    # ========== 変数の定義 ==========
    # 対称性により、u1のみを学習（u2は不要）
    # 変数 u[j] (参加者1が型jのときの効用)
    u = {
        j: pulp.LpVariable(f"u_{j}", 
                          lowBound=0.0, cat=pulp.LpContinuous)
        for j in range(J1)
    }
    
    # 変数 p[l, j] (参加者1が型jのときの財lの配分確率)
    # l=0: 財a, l=1: 財b
    # Feasibility制約により、0 ≤ 2p[l, j] ≤ 1、つまり 0 ≤ p[l, j] ≤ 0.5
    p = {
        (l, j): pulp.LpVariable(f"p_{l}_{j}", 
                                lowBound=0.0, upBound=0.5, cat=pulp.LpContinuous)
        for l in range(2)
        for j in range(J1)
    }

    # ========== 目的関数 ==========
    # max 2 Σ_j w_j (p(j)・x_j - u(j))
    objective = pulp.lpSum(
        2.0 * weights1_arr[j] * (
            p[(0, j)] * points1_arr[j, 0]  # 財a
            + p[(1, j)] * points1_arr[j, 1]  # 財b
            - u[j]
        )
        for j in range(J1)
    )
    prob += objective

    # ========== 制約 ==========
    
    # 1. 非負性: u[j] >= 0
    # 変数の定義で既に lowBound=0.0 として設定済み
    
    # 2. Feasibility制約（itemのallocation）: 0 ≤ 2p[l,j] ≤ 1
    # つまり、0 ≤ p[l,j] ≤ 0.5
    # 変数の定義で既に lowBound=0.0, upBound=0.5 として設定済み
    
    # 3. IC制約: u(i) >= u(j) + p(j)・(x(i) - x(j)) ∀i, j
    # 差分行列をキャッシュから参照して高速化
    for i in range(J1):
        for j in range(J1):
            # IC制約: u(i) >= u(j) + p(j)・(x(i) - x(j))
            # 差分行列から直接参照（points1_diff[i, j, :]）
            prob += u[i] >= u[j] + (
                p[(0, j)] * points1_diff[i, j, 0]  # 財aの項
                + p[(1, j)] * points1_diff[i, j, 1]  # 財bの項
            ), f"ic_{i}_{j}"

    # ========== 解く ==========
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)

    # ========== 結果 ==========
    # NumPy配列に変換
    u1_sol = np.fromiter((u[j].varValue for j in range(J1)), 
                        dtype=np.float64, count=J1)
    
    # pの形状: (2, J1)
    p1_sol = np.zeros((2, J1), dtype=np.float64)
    for l in range(2):
        p1_sol[l] = np.fromiter((p[(l, j)].varValue for j in range(J1)),
                                dtype=np.float64, count=J1)
    
    # 戻り値のインターフェースを維持（後方互換性のため）
    u2_sol = u1_sol.copy()  # 対称性により同じ
    p2_sol = p1_sol.copy()  # 対称性により同じ

    return status, obj_val, u1_sol, u2_sol, p1_sol, p2_sol


def solve_mechanism_symmetry_2agents_iterative(points1, weights1, grid_sizes1,
                                                solver=None, max_iter=100, tol=1e-6):
    """
    反復的制約追加アルゴリズムによるRochet-Choné型LPの求解（2人2財版、シナジーなし、対称性制約あり）。
    
    アルゴリズム:
    1. 局所的なIC制約（ic-local）のみで初期問題を定義
    2. 最適解を求める
    3. 違反している大局的なIC制約を検出
    4. 違反制約を追加して問題を更新
    5. 違反がなくなるまで反復
    
    パラメータ:
        points1: list of tuples, 参加者1の型空間の点 (x₁_a, x₁_b) - 2次元
        weights1: list of floats, 参加者1の各点の重み
        grid_sizes1: tuple, 参加者1の各次元のグリッドサイズ (nx1, ny1)
        solver: PuLPソルバー（必須: Gurobi）
        max_iter: 最大反復回数
        tol: 違反判定の許容誤差
    
    戻り値:
        (status_string, objective_value, u1, u2, p1, p2, n_iterations)
    """
    J1 = len(points1)
    nx1, ny1 = grid_sizes1
    
    # pointsをNumPy配列に変換
    points1_arr = np.asarray(points1, dtype=np.float64)  # (J1, 2)
    weights1_arr = np.asarray(weights1, dtype=np.float64)  # (J1,)
    
    # グリッドインデックスを計算する関数
    def get_grid_indices_1(point_idx):
        """参加者1の点のインデックスからグリッド座標を取得"""
        j = point_idx % ny1
        i_coord = point_idx // ny1
        return (i_coord, j)
    
    def get_point_idx_1(grid_indices):
        """参加者1のグリッド座標から点のインデックスを取得"""
        i_coord, j = grid_indices
        return i_coord * ny1 + j
    
    def get_neighbors_1(point_idx):
        """参加者1の局所的なIC制約に使用する隣接点のインデックスを取得"""
        i_coord, j = get_grid_indices_1(point_idx)
        neighbors = []
        for di in [-1, 1]:
            if 0 <= i_coord + di < nx1:
                neighbor_idx = get_point_idx_1((i_coord + di, j))
                neighbors.append(neighbor_idx)
        for dj in [-1, 1]:
            if 0 <= j + dj < ny1:
                neighbor_idx = get_point_idx_1((i_coord, j + dj))
                neighbors.append(neighbor_idx)
        return neighbors
    
    # ========== 問題の初期化 ==========
    prob = pulp.LpProblem("RC_iterative_symmetry_2agents_2goods", pulp.LpMaximize)
    
    # 変数の定義（対称性によりu1のみ）
    u = {
        j: pulp.LpVariable(f"u_{j}", 
                           lowBound=0.0, cat=pulp.LpContinuous)
        for j in range(J1)
    }
    
    # Feasibility制約により、0 ≤ 2p[l, j] ≤ 1、つまり 0 ≤ p[l, j] ≤ 0.5
    p = {
        (l, j): pulp.LpVariable(f"p_{l}_{j}", 
                               lowBound=0.0, upBound=0.5, cat=pulp.LpContinuous)
        for l in range(2)
        for j in range(J1)
    }
    
    # 目的関数: 2 Σ_j w_j (p(j)・x_j - u(j))
    objective = pulp.lpSum(
        2.0 * weights1_arr[j] * (
            p[(0, j)] * points1_arr[j, 0]
            + p[(1, j)] * points1_arr[j, 1]
            - u[j]
        )
        for j in range(J1)
    )
    prob += objective
    
    # 差分行列をループ外で一度だけ計算（反復中は不変）
    points1_diff = points1_arr[:, None, :] - points1_arr[None, :, :]  # (J1, J1, 2)
    
    # 局所的なIC制約
    local_constraints = set()
    for i in range(J1):
        neighbors = get_neighbors_1(i)
        for k in neighbors:
            # 差分行列から直接参照（points1_diff[i, k, :]）
            prob += u[i] >= u[k] + (
                p[(0, k)] * points1_diff[i, k, 0]
                + p[(1, k)] * points1_diff[i, k, 1]
            ), f"ic_local_{i}_{k}"
            local_constraints.add((i, k))
    
    # 追加された制約を記録
    added_constraints = local_constraints.copy()
    
    # 制約マスクをループ外で初期化（違反分だけ更新する方式）
    constraint_mask = np.zeros((J1, J1), dtype=bool)
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
            return status, None, None, None, None, None, iteration
        
        # 解を取得
        u_sol = np.fromiter((u[j].varValue for j in range(J1)),
                           dtype=np.float64, count=J1)
        
        p_sol = np.zeros((2, J1), dtype=np.float64)
        for l in range(2):
            p_sol[l] = np.fromiter((p[(l, j)].varValue for j in range(J1)),
                                  dtype=np.float64, count=J1)
        
        # 違反している制約を検出
        # ベクトル化で高速化
        violations = []
        
        # 全ての(i, k)の組み合わせに対してベクトル演算
        # u_diff[i, k] = u_sol[i] - u_sol[k]
        u_diff = u_sol[:, None] - u_sol[None, :]  # (J1, J1)
        
        # points1_diffは既にループ外で計算済み（再計算不要）
        
        # inner_productをeinsumで計算（高速化）
        # inner_product[i, k] = Σ_l p_sol[l, k] * points1_diff[i, k, l]
        inner_product = np.einsum('lk,ikl->ik', p_sol, points1_diff)  # 結果の形状: (J1, J1)
        
        # 違反チェック: u_i - u_k < p_k・(x_i - x_k) - tol
        violation_mask = (u_diff < inner_product - tol) & (~constraint_mask)
        
        # 違反している(i, k)のペアを取得（np.argwhereで直接取得）
        violation_pairs = np.argwhere(violation_mask)  # (N_violations, 2)
        violations = [(int(i), int(k)) for i, k in violation_pairs]
        
        # 違反している制約を追加（バッチ処理で高速化）
        # 差分行列から直接参照して高速化
        for i, k in violations:
            # 差分行列から直接参照（points1_diff[i, k, :]）
            constraint = u[i] >= u[k] + (
                p[(0, k)] * points1_diff[i, k, 0]
                + p[(1, k)] * points1_diff[i, k, 1]
            )
            prob += constraint, f"ic_{i}_{k}_iter{iteration}"
            added_constraints.add((i, k))
            # 制約マスクを更新（違反分だけTrueに設定）
            constraint_mask[i, k] = True
        
        total_violations = len(violations)
        
        # 違反がなければ終了
        if total_violations == 0:
            status = pulp.LpStatus[prob.status]
            obj_val = pulp.value(prob.objective)
            
            # 戻り値のインターフェースを維持（後方互換性のため）
            u1_sol = u_sol
            u2_sol = u_sol.copy()  # 対称性により同じ
            p1_sol = p_sol
            p2_sol = p_sol.copy()  # 対称性により同じ
            
            return status, obj_val, u1_sol, u2_sol, p1_sol, p2_sol, iteration + 1
        
        print(f"Iteration {iteration + 1}: {total_violations} violations found, added {total_violations} constraints")
    
    # 最大反復回数に達した場合
    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)
    
    # 戻り値のインターフェースを維持（後方互換性のため）
    u1_sol = u_sol
    u2_sol = u_sol.copy()  # 対称性により同じ
    p1_sol = p_sol
    p2_sol = p_sol.copy()  # 対称性により同じ
    
    return status, obj_val, u1_sol, u2_sol, p1_sol, p2_sol, max_iter


def save_results_symmetry_2agents(
    points1, weights1, u1_sol, u2_sol, p1_sol, p2_sol,
    obj_val, status,
    points2=None, weights2=None,
    grid_sizes1=None, grid_sizes2=None, n_iter=None,
    filename=None, data_dir="data"
):
    """
    2人2財の結果をNumPy形式で保存する（対称性制約あり）。
    
    パラメータ:
        points1: 参加者1の型空間の点 (list or np.ndarray)
        weights1: 参加者1の各点の重み (list or np.ndarray)
        u1_sol: 参加者1の効用 (np.ndarray, shape: (J1, J2))
        u2_sol: 参加者2の効用 (np.ndarray, shape: (J1, J2))
        p1_sol: 参加者1の配分確率 (np.ndarray, shape: (2, J1, J2))
        p2_sol: 参加者2の配分確率 (np.ndarray, shape: (2, J1, J2))
        obj_val: 目的関数値 (float)
        status: LPステータス (str)
        points2: 参加者2の型空間の点 (list or np.ndarray, optional, 対称性によりpoints1と同じ)
        weights2: 参加者2の各点の重み (list or np.ndarray, optional, 対称性によりweights1と同じ)
        grid_sizes1: 参加者1のグリッドサイズ (tuple, optional)
        grid_sizes2: 参加者2のグリッドサイズ (tuple, optional, 対称性によりgrid_sizes1と同じ)
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
        filename = f"results_symmetry_2agents_{timestamp}.npz"
    
    filepath = os.path.join(data_dir, filename)
    
    # 対称性により、points2とweights2が指定されていない場合はpoints1とweights1を使用
    if points2 is None:
        points2 = points1
    if weights2 is None:
        weights2 = weights1
    if grid_sizes2 is None:
        grid_sizes2 = grid_sizes1
    
    # NumPy配列に変換
    points1_arr = np.array(points1, dtype=np.float64)
    weights1_arr = np.array(weights1, dtype=np.float64)
    points2_arr = np.array(points2, dtype=np.float64)
    weights2_arr = np.array(weights2, dtype=np.float64)
    u1_arr = np.array(u1_sol, dtype=np.float64)
    u2_arr = np.array(u2_sol, dtype=np.float64)
    p1_arr = np.array(p1_sol, dtype=np.float64)
    p2_arr = np.array(p2_sol, dtype=np.float64)
    
    # NumPy形式で保存
    save_dict = {
        'obj_val': np.array([obj_val], dtype=np.float64) if obj_val is not None else np.array([np.nan], dtype=np.float64),
        'points1': points1_arr,
        'weights1': weights1_arr,
        'points2': points2_arr,
        'weights2': weights2_arr,
        'u1_sol': u1_arr,
        'u2_sol': u2_arr,
        'p1_sol': p1_arr,
        'p2_sol': p2_arr,
        'J1': np.array([len(points1)], dtype=np.int32),
        'J2': np.array([len(points2)], dtype=np.int32),
    }
    
    if grid_sizes1 is not None:
        save_dict['grid_sizes1'] = np.array(grid_sizes1, dtype=np.int32)
    if grid_sizes2 is not None:
        save_dict['grid_sizes2'] = np.array(grid_sizes2, dtype=np.int32)
    if n_iter is not None:
        save_dict['n_iter'] = np.array([n_iter], dtype=np.int32)
    
    # メタデータファイルも保存
    metadata_filepath = filepath.replace('.npz', '_metadata.txt')
    with open(metadata_filepath, 'w') as f:
        f.write(f"status: {status}\n")
        f.write(f"obj_val: {obj_val}\n")
        f.write(f"J1: {len(points1)}, J2: {len(points2)}\n")
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
        if grid_sizes1 is not None:
            f.write(f"grid_sizes1: {grid_sizes1}\n")
        if grid_sizes2 is not None:
            f.write(f"grid_sizes2: {grid_sizes2}\n")
        if n_iter is not None:
            f.write(f"n_iter: {n_iter}\n")
    
    np.savez_compressed(filepath, **save_dict)
    
    print(f"Results saved to: {filepath}")
    return filepath


def load_results_symmetry_2agents(filepath):
    """
    保存された結果を読み込む（対称性制約あり）。
    
    パラメータ:
        filepath: 保存されたファイルのパス (str)
    
    戻り値:
        dict: 読み込んだデータの辞書
    """
    data = np.load(filepath, allow_pickle=True)
    
    result = {
        'points1': data['points1'],
        'weights1': data['weights1'],
        'points2': data['points2'],
        'weights2': data['weights2'],
        'u1_sol': data['u1_sol'],
        'u2_sol': data['u2_sol'],
        'p1_sol': data['p1_sol'],
        'p2_sol': data['p2_sol'],
        'obj_val': float(data['obj_val'][0]) if not np.isnan(data['obj_val'][0]) else None,
        'J1': int(data['J1'][0]),
        'J2': int(data['J2'][0]),
    }
    
    if 'grid_sizes1' in data:
        result['grid_sizes1'] = tuple(data['grid_sizes1'])
    if 'grid_sizes2' in data:
        result['grid_sizes2'] = tuple(data['grid_sizes2'])
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

