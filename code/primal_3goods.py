"""
Primal問題: Rochet-Choné型LP
3財4シナジーの最適オークション機構設計

Cursor.mdのAppendix仕様に基づく:
1人3財4シナジーを、1人7財の問題+シナジーの分配の制約と捉え直す。
財a, 財b, 財c, α12, α23, α31, α123
"""

import pulp
import numpy as np
import os
from datetime import datetime


def solve_mechanism_3goods_4synergy(points, weights, solver=None):
    """
    Rochet–Choné 型の LP を構築して解く（3財4シナジー版）。
    
    Cursor.mdのAppendix仕様に基づく:
    - 1人3財4シナジーを、1人7財の問題+シナジーの分配の制約と捉え直す
    - 財a, 財b, 財c, α12, α23, α31, α123

    型数: J = len(points)
    財数: 3財4シナジー（pointsの次元は7である必要がある）

    変数:
        u[j]       : 各グリッド点でのinterim utility (>=0)
        p[l,j]     : 配分確率のベクトル（uの勾配）
                     l=0: 財a, l=1: 財b, l=2: 財c
                     l=3: シナジーα12, l=4: シナジーα23, l=5: シナジーα31, l=6: シナジーα123
                     (0<=p<=1)

    目的関数:
        max Σ_j w_j (p(j)・x(j) - u_j)
        = max Σ_j w_j (Σ_{l=0}^{6} p[l,j]*x[j,l] - u_j)

    制約:
        - 非負性: u[j] >= 0 (全てのxで)
        - convex: u_i >= u_j + p_j (x_i - x_j) for all i,j
        - 1-Lipshitz: 0 <= p[l,j] <= 1 (l財で点jの傾き)
        - シナジーの配分制約（Cursor.md Appendix参照）

    パラメータ:
        points: list of tuples, 各tupleは型空間の点 (x_a, x_b, x_c, x_α12, x_α23, x_α31, x_α123) - 7次元
        weights: list of floats, 各点の重み w
        solver: PuLPソルバー（必須: Gurobi）

    戻り値:
        (status_string, objective_value, u, p)
        - u: list of floats, 各型での効用
        - p: list of lists, p[l][j] = 財lを型jに割り当てる確率 (l=0,...,6)
    """
    J = len(points)
    assert J == len(weights)
    assert len(points[0]) == 7, "pointsは7次元である必要があります (財a, 財b, 財c, α12, α23, α31, α123)"

    # 問題設定
    prob = pulp.LpProblem("RC_3goods_4synergy", pulp.LpMaximize)

    # ========== 変数の定義 ==========
    
    # 変数 u_j (効用、連続変数)
    u = {
        j: pulp.LpVariable(f"u_{j}", lowBound=0.0, cat=pulp.LpContinuous)
        for j in range(J)
    }

    # 変数 p_{l,j} (配分確率のベクトル: l=0,...,6)
    # l=0: 財a, l=1: 財b, l=2: 財c
    # l=3: シナジーα12, l=4: シナジーα23, l=5: シナジーα31, l=6: シナジーα123
    # 連続変数として定義（[0,1]の任意の値を取れる）
    p = {
        (l, j): pulp.LpVariable(f"p_{l}_{j}", lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
        for l in range(7)
        for j in range(J)
    }

    # ========== 目的関数 ==========
    # Cursor.md: max Σ_j w[j] ((p(j)・x(j)) - u(j))
    objective = pulp.lpSum(
        weights[j] * (
            sum(p[(l, j)] * points[j][l] for l in range(7))  # 7次元ベクトルの内積
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
    # pointsをNumPy配列に変換（高速化のため）
    points_arr = np.array(points, dtype=np.float64)  # (J, 7)
    for i in range(J):
        x_i = points_arr[i]  # NumPy配列から直接取得
        for j in range(J):
            x_j = points_arr[j]  # NumPy配列から直接取得
            prob += u[i] >= u[j] + sum(
                p[(l, j)] * (x_i[l] - x_j[l]) for l in range(7)
            ), f"convex_{i}_{j}"
    
    # 4. シナジーの配分制約
    # Cursor.md Appendix:
    # 2財のシナジー（α12, α23, α31）
    for j in range(J):
        # α12: p[3,j] <= p[0,j], p[3,j] <= p[1,j], p[3,j] >= p[0,j] + p[1,j] - 1
        prob += p[(3, j)] <= p[(0, j)], f"synergy_α12_item0_type_{j}_upper"
        prob += p[(3, j)] <= p[(1, j)], f"synergy_α12_item1_type_{j}_upper"
        prob += p[(3, j)] >= p[(0, j)] + p[(1, j)] - 1, f"synergy_α12_type_{j}_lower"
        
        # α23: p[4,j] <= p[1,j], p[4,j] <= p[2,j], p[4,j] >= p[1,j] + p[2,j] - 1
        prob += p[(4, j)] <= p[(1, j)], f"synergy_α23_item1_type_{j}_upper"
        prob += p[(4, j)] <= p[(2, j)], f"synergy_α23_item2_type_{j}_upper"
        prob += p[(4, j)] >= p[(1, j)] + p[(2, j)] - 1, f"synergy_α23_type_{j}_lower"
        
        # α31: p[5,j] <= p[2,j], p[5,j] <= p[0,j], p[5,j] >= p[2,j] + p[0,j] - 1
        prob += p[(5, j)] <= p[(2, j)], f"synergy_α31_item2_type_{j}_upper"
        prob += p[(5, j)] <= p[(0, j)], f"synergy_α31_item0_type_{j}_upper"
        prob += p[(5, j)] >= p[(2, j)] + p[(0, j)] - 1, f"synergy_α31_type_{j}_lower"
        
        # α123: 3財のシナジー
        # 上界: p[6,j] <= p[0,j], p[6,j] <= p[1,j], p[6,j] <= p[2,j]
        prob += p[(6, j)] <= p[(0, j)], f"synergy_α123_item0_type_{j}_upper"
        prob += p[(6, j)] <= p[(1, j)], f"synergy_α123_item1_type_{j}_upper"
        prob += p[(6, j)] <= p[(2, j)], f"synergy_α123_item2_type_{j}_upper"
        
        # 下界: 複数の制約
        prob += p[(6, j)] >= p[(3, j)] + p[(2, j)] - 1, f"synergy_α123_α12_3_type_{j}_lower"
        prob += p[(6, j)] >= p[(4, j)] + p[(0, j)] - 1, f"synergy_α123_α23_1_type_{j}_lower"
        prob += p[(6, j)] >= p[(5, j)] + p[(1, j)] - 1, f"synergy_α123_α31_2_type_{j}_lower"
        prob += p[(6, j)] >= p[(3, j)] + p[(4, j)] + p[(5, j)] - p[(0, j)] - p[(1, j)] - p[(2, j)] + 1, f"synergy_α123_full_type_{j}_lower"

    # ========== 解く ==========
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)

    # ========== 結果 ==========
    # NumPy配列に直接変換
    u_sol = np.array([u[j].varValue for j in range(J)], dtype=np.float64)
    p_sol = np.array([[p[(l, j)].varValue for j in range(J)] for l in range(7)], dtype=np.float64)

    return status, obj_val, u_sol, p_sol


def solve_mechanism_3goods_4synergy_iterative(points, weights, grid_size, solver=None, max_iter=100, tol=1e-6):
    """
    反復的制約追加アルゴリズムによるRochet-Choné型LPの求解（3財4シナジー版）
    
    アルゴリズム:
    1. 局所的なIC制約（ic-local）のみで初期問題を定義
    2. 最適解を求める
    3. 違反している大局的なIC制約を検出
    4. 違反制約を追加して問題を更新
    5. 違反がなくなるまで反復
    
    パラメータ:
        points: list of tuples, 各tupleは型空間の点 (x_a, x_b, x_c, x_α12, x_α23, x_α31, x_α123) - 7次元
        weights: list of floats, 各点の重み
        grid_size: int, 各次元のグリッドサイズ（7次元すべて同じサイズ）
        solver: PuLPソルバー（必須: Gurobi）
        max_iter: 最大反復回数
        tol: 違反判定の許容誤差
    
    戻り値:
        (status_string, objective_value, u, p, n_iterations)
    """
    J = len(points)
    assert J == len(weights)
    assert len(points[0]) == 7, "pointsは7次元である必要があります"
    
    n = grid_size
    assert n ** 7 == J, f"grid_size^7({n ** 7})がpointsの数({J})と一致しません"
    
    # pointsをNumPy配列に変換（一度だけ、高速化のため）
    points_arr = np.array(points, dtype=np.float64)  # (J, 7)
    
    # グリッドインデックスを計算する関数
    def get_grid_indices(point_idx):
        """点のインデックスからグリッド座標（各次元のインデックス）を取得"""
        indices = []
        remaining = point_idx
        for _ in range(7):
            indices.append(remaining % n)
            remaining //= n
        return list(reversed(indices))
    
    def get_point_idx(grid_indices):
        """グリッド座標から点のインデックスを取得"""
        idx = 0
        multiplier = 1
        for coord in reversed(grid_indices):
            idx += coord * multiplier
            multiplier *= n
        return idx
    
    def get_neighbors(point_idx):
        """局所的なIC制約に使用する隣接点のインデックスを取得"""
        grid_indices = get_grid_indices(point_idx)
        neighbors = []
        for d in range(7):
            # 各次元で±1の隣接点
            for offset in [-1, 1]:
                new_indices = grid_indices.copy()
                new_indices[d] += offset
                if 0 <= new_indices[d] < n:
                    neighbor_idx = get_point_idx(new_indices)
                    neighbors.append(neighbor_idx)
        return neighbors
    
    # ========== 問題の初期化 ==========
    prob = pulp.LpProblem("RC_iterative_3goods_4synergy", pulp.LpMaximize)
    
    # 変数の定義
    u = {j: pulp.LpVariable(f"u_{j}", lowBound=0.0, cat=pulp.LpContinuous) for j in range(J)}
    # 配分確率は連続変数（[0,1]の任意の値を取れる）
    p = {
        (l, j): pulp.LpVariable(f"p_{l}_{j}", lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
        for l in range(7)
        for j in range(J)
    }
    
    # 目的関数
    objective = pulp.lpSum(
        weights[j] * (
            sum(p[(l, j)] * points[j][l] for l in range(7))
            - u[j]
        )
        for j in range(J)
    )
    prob += objective
    
    # シナジーの配分制約
    for j in range(J):
        # α12
        prob += p[(3, j)] <= p[(0, j)], f"synergy_α12_item0_type_{j}_upper"
        prob += p[(3, j)] <= p[(1, j)], f"synergy_α12_item1_type_{j}_upper"
        prob += p[(3, j)] >= p[(0, j)] + p[(1, j)] - 1, f"synergy_α12_type_{j}_lower"
        # α23
        prob += p[(4, j)] <= p[(1, j)], f"synergy_α23_item1_type_{j}_upper"
        prob += p[(4, j)] <= p[(2, j)], f"synergy_α23_item2_type_{j}_upper"
        prob += p[(4, j)] >= p[(1, j)] + p[(2, j)] - 1, f"synergy_α23_type_{j}_lower"
        # α31
        prob += p[(5, j)] <= p[(2, j)], f"synergy_α31_item2_type_{j}_upper"
        prob += p[(5, j)] <= p[(0, j)], f"synergy_α31_item0_type_{j}_upper"
        prob += p[(5, j)] >= p[(2, j)] + p[(0, j)] - 1, f"synergy_α31_type_{j}_lower"
        # α123
        prob += p[(6, j)] <= p[(0, j)], f"synergy_α123_item0_type_{j}_upper"
        prob += p[(6, j)] <= p[(1, j)], f"synergy_α123_item1_type_{j}_upper"
        prob += p[(6, j)] <= p[(2, j)], f"synergy_α123_item2_type_{j}_upper"
        prob += p[(6, j)] >= p[(3, j)] + p[(2, j)] - 1, f"synergy_α123_α12_3_type_{j}_lower"
        prob += p[(6, j)] >= p[(4, j)] + p[(0, j)] - 1, f"synergy_α123_α23_1_type_{j}_lower"
        prob += p[(6, j)] >= p[(5, j)] + p[(1, j)] - 1, f"synergy_α123_α31_2_type_{j}_lower"
        prob += p[(6, j)] >= p[(3, j)] + p[(4, j)] + p[(5, j)] - p[(0, j)] - p[(1, j)] - p[(2, j)] + 1, f"synergy_α123_full_type_{j}_lower"
    
    # 局所的なIC制約（隣接点のみ）
    local_constraints = set()
    for i in range(J):
        neighbors = get_neighbors(i)
        x_i = points_arr[i]  # NumPy配列から直接取得
        for j in neighbors:
            x_j = points_arr[j]  # NumPy配列から直接取得
            prob += u[i] >= u[j] + sum(
                p[(l, j)] * (x_i[l] - x_j[l]) for l in range(7)
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
        p_sol = np.array([[p[(l, j)].varValue for j in range(J)] for l in range(7)], dtype=np.float64)  # (7, J)
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
        
        # points_diff[i,j] = points[i] - points[j] (形状: (J, J, 7))
        points_diff = points_arr[:, np.newaxis, :] - points_arr[np.newaxis, :, :]  # (J, J, 7)
        
        # inner_product[i,j] = Σ_{l=0}^{6} p_sol[l,j] * (x_i[l] - x_j[l])
        # p_solの形状は(7, J)、points_diffの形状は(J, J, 7)
        inner_product = np.zeros((J, J), dtype=np.float64)
        for l in range(7):
            inner_product += p_sol[l, :][np.newaxis, :] * points_diff[:, :, l]  # (J, J)
        
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
            constraint = u[i] >= u[j] + sum(
                p[(l, j)] * (x_i[l] - x_j[l]) for l in range(7)
            )
            prob += constraint, f"convex_{i}_{j}_iter{iteration}"
            added_constraints.add((i, j))
        
        print(f"Iteration {iteration + 1}: {len(violations)} violations found, added {len(violations)} constraints")
    
    # 最大反復回数に達した場合
    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)
    u_sol = np.array([u[j].varValue for j in range(J)], dtype=np.float64)
    p_sol = np.array([[p[(l, j)].varValue for j in range(J)] for l in range(7)], dtype=np.float64)
    return status, obj_val, u_sol, p_sol, max_iter


def save_results_3goods(points, weights, u_sol, p_sol, obj_val, status,
                        grid_size=None, n_iter=None, filename=None, data_dir="data"):
    """
    3財4シナジーの結果をNumPy形式で保存する。
    
    パラメータ:
        points: 型空間の点 (list or np.ndarray)
        weights: 各点の重み (list or np.ndarray)
        u_sol: 効用 (np.ndarray, shape: (J,))
        p_sol: 配分確率 (np.ndarray, shape: (7, J))
        obj_val: 目的関数値 (float)
        status: LPステータス (str)
        grid_size: グリッドサイズ (int, optional)
        n_iter: 反復回数 (int, optional)
        filename: 保存ファイル名 (str, optional)
        data_dir: データ保存ディレクトリ (str, default: "data")
    
    戻り値:
        filepath: 保存されたファイルのパス (str)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_3goods_4synergy_{timestamp}.npz"
    
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
    
    if grid_size is not None:
        save_dict['grid_size'] = np.array([grid_size], dtype=np.int32)
    if n_iter is not None:
        save_dict['n_iter'] = np.array([n_iter], dtype=np.int32)
    
    metadata_filepath = filepath.replace('.npz', '_metadata.txt')
    with open(metadata_filepath, 'w') as f:
        f.write(f"status: {status}\n")
        f.write(f"obj_val: {obj_val}\n")
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
        if grid_size is not None:
            f.write(f"grid_size: {grid_size}\n")
        if n_iter is not None:
            f.write(f"n_iter: {n_iter}\n")
    
    np.savez_compressed(filepath, **save_dict)
    print(f"Results saved to: {filepath}")
    return filepath


def load_results_3goods(filepath):
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
    
    if 'grid_size' in data:
        result['grid_size'] = int(data['grid_size'][0])
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

