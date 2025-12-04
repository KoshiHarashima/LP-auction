"""
Primal問題: Rochet-Choné型LP
複数エージェント2財の最適オークション機構設計（シナジーなし、Implementability制約なし）

primal_2agents_2goods_with_synergy.pyを改良:
- シナジーを削除（2財のみ）
- Implementability制約を削除
- 複数エージェントに対応可能な設計

Cursor.mdに基づく実装:
複数の参加者がそれぞれ2財を持つケースを扱う。
各参加者の型は独立に分布から抽出され、メカニズムは全参加者の型の組み合わせに依存する。
"""

import pulp
import numpy as np
import os
from datetime import datetime
from itertools import product


def solve_mechanism_multi_agent(points_list, weights_list, solver=None):
    """
    Rochet–Choné 型の LP を構築して解く（複数エージェント2財版、シナジーなし）。
    
    仕様:
    - 各参加者i: 型 (x_i_a, x_i_b) ∈ [0,1]²
    - 各参加者の型は独立に分布から抽出される
    - メカニズムは全参加者の型の組み合わせに依存する

    型数: 
        J_i = len(points_list[i]) (参加者iの型数)
    財数: 各参加者が2財（pointsの次元は2である必要がある）

    変数:
        u_i[j_1, ..., j_n]   : 参加者iが型j_iで他の参加者が型j_{-i}のときの参加者iのinterim utility (>=0)
        p_i[l, j_1, ..., j_n]: 参加者iへの配分確率のベクトル（u_iの勾配）
                               l=0: 財a, l=1: 財b (0<=p<=1)

    目的関数:
        max Σ_{j_1, ..., j_n} (Π_i w_i[j_i]) * Σ_i (p_i(j_1,...,j_n)・x_i(j_i) - u_i(j_1,...,j_n))
        = max Σ_{j_1, ..., j_n} (Π_i w_i[j_i]) * Σ_i (
            p_i[0,j_1,...,j_n]*x_i[j_i,0] + p_i[1,j_1,...,j_n]*x_i[j_i,1] - u_i[j_1,...,j_n]
        )

    制約:
        - 非負性: u_i[j_1,...,j_n] >= 0 (全てのj_1,...,j_nで)
        - IC（参加者i）: u_i(i_i, j_{-i}) >= u_i(k_i, j_{-i}) + p_i(k_i, j_{-i})・(x_i(i_i) - x_i(k_i)) 
                         for all i_i, k_i, j_{-i}
        - 1-Lipschitz: 0 <= p_i[l,j_1,...,j_n] <= 1
        - IR: u_i(x) >= 0 for all x
              （実装では、変数の定義時に lowBound=0.0 として設定し、IR制約は明示的に追加しない）

    パラメータ:
        points_list: list of lists of tuples, 各参加者の型空間の点 [(x_i_a, x_i_b), ...] - 2次元
        weights_list: list of lists of floats, 各参加者の各点の重み [w_i, ...]
        solver: PuLPソルバー（必須: Gurobi）

    戻り値:
        (status_string, objective_value, u_list, p_list)
        - u_list: list of np.ndarray, u_list[i][j_1,...,j_n] = 参加者iの効用
        - p_list: list of np.ndarray, p_list[i][l][j_1,...,j_n] = 参加者iへの財lの配分確率 (l=0,1)
    """
    n_agents = len(points_list)
    # 各参加者の型数を取得
    J_list = [len(points) for points in points_list]
    
    # pointsとweightsをNumPy配列に変換（一度だけ、高速化のため）
    points_arr_list = [np.asarray(points, dtype=np.float64) for points in points_list]
    weights_arr_list = [np.asarray(weights, dtype=np.float64) for weights in weights_list]
    
    # 各エージェントの差分行列を一括計算（ループ外で一度だけ）
    # points_diff_list[i][j_i, k_i] = points[i][j_i] - points[i][k_i] (形状: (J_i, J_i, 2))
    points_diff_list = [
        points_arr[:, None, :] - points_arr[None, :, :] 
        for points_arr in points_arr_list
    ]
    
    # 問題設定
    prob = pulp.LpProblem("RC_multi_agent_2goods", pulp.LpMaximize)

    # ========== 変数の定義 ==========
    # インデックスをタプルで管理するためのヘルパー関数
    def get_all_indices():
        """全参加者の型の組み合わせを生成"""
        from itertools import product
        return list(product(*[range(J) for J in J_list]))
    
    all_indices = get_all_indices()
    
    # 変数 u_i[j_1, ..., j_n] (各参加者の効用、連続変数)
    u_list = []
    for i in range(n_agents):
        u_i = {
            indices: pulp.LpVariable(f"u{i}_{'_'.join(map(str, indices))}", 
                                     lowBound=0.0, cat=pulp.LpContinuous)
            for indices in all_indices
        }
        u_list.append(u_i)
    
    # 変数 p_i[l, j_1, ..., j_n] (各参加者の配分確率のベクトル: l=0,1)
    # l=0: 財a, l=1: 財b
    p_list = []
    for i in range(n_agents):
        p_i = {
            (l, indices): pulp.LpVariable(f"p{i}_{l}_{'_'.join(map(str, indices))}", 
                                          lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
            for l in range(2)
            for indices in all_indices
        }
        p_list.append(p_i)

    # ========== 目的関数 ==========
    # max Σ_{j_1, ..., j_n} (Π_i w_i[j_i]) * Σ_i (p_i(j_1,...,j_n)・x_i(j_i) - u_i(j_1,...,j_n))
    objective_terms = []
    for indices in all_indices:
        # 重みの積を計算
        weight_product = 1.0
        for i in range(n_agents):
            weight_product *= weights_arr_list[i][indices[i]]
        
        # 各参加者の項を追加
        for i in range(n_agents):
            j_i = indices[i]
            term = (
                p_list[i][(0, indices)] * points_arr_list[i][j_i, 0]  # 財aの価値
                + p_list[i][(1, indices)] * points_arr_list[i][j_i, 1]  # 財bの価値
                - u_list[i][indices]
            )
            objective_terms.append(weight_product * term)
    
    objective = pulp.lpSum(objective_terms)
    prob += objective

    # ========== 制約 ==========
    
    # 1. 非負性: u_i[j_1,...,j_n] >= 0 (全てのj_1,...,j_nで)
    # 変数の定義で既に lowBound=0.0 として設定済み
    
    # 2. 1-Lipschitz制約: 0 ≤ p_i[l,j_1,...,j_n] ≤ 1
    # 変数の定義で既に lowBound=0.0, upBound=1.0 として設定済み
    
    # 3. IC制約（各参加者について）
    for i in range(n_agents):
        # 参加者iの型の組み合わせ
        for i_i in range(J_list[i]):
            for k_i in range(J_list[i]):
                # 差分行列から直接参照（points_diff_list[i][i_i, k_i, :]）
                # 他の参加者の型の組み合わせ
                other_indices = [range(J_list[j]) for j in range(n_agents) if j != i]
                for other_comb in product(*other_indices):
                    # 全参加者の型の組み合わせを構築
                    indices_i = list(other_comb[:i]) + [i_i] + list(other_comb[i:])
                    indices_k = list(other_comb[:i]) + [k_i] + list(other_comb[i:])
                    indices_i = tuple(indices_i)
                    indices_k = tuple(indices_k)
                    
                    # IC制約: u_i(i_i, j_{-i}) >= u_i(k_i, j_{-i}) + p_i(k_i, j_{-i})・(x_i(i_i) - x_i(k_i))
                    # 差分行列から直接参照（points_diff_list[i][i_i, k_i, :]）
                    prob += u_list[i][indices_i] >= u_list[i][indices_k] + (
                        p_list[i][(0, indices_k)] * points_diff_list[i][i_i, k_i, 0]  # 財aの項
                        + p_list[i][(1, indices_k)] * points_diff_list[i][i_i, k_i, 1]  # 財bの項
                    ), f"ic{i}_{i_i}_{k_i}_{'_'.join(map(str, other_comb))}"

    # ========== 解く ==========
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)

    # ========== 結果 ==========
    # NumPy配列に変換
    u_sol_list = []
    p_sol_list = []
    
    for i in range(n_agents):
        # u_iの形状: (J_1, ..., J_n)
        u_sol = np.zeros(J_list, dtype=np.float64)
        for indices in all_indices:
            u_sol[indices] = u_list[i][indices].varValue
        u_sol_list.append(u_sol)
        
        # p_iの形状: (2, J_1, ..., J_n)
        p_sol = np.zeros((2,) + tuple(J_list), dtype=np.float64)
        for l in range(2):
            for indices in all_indices:
                p_sol[(l,) + indices] = p_list[i][(l, indices)].varValue
        p_sol_list.append(p_sol)

    return status, obj_val, u_sol_list, p_sol_list


def solve_mechanism_2agents(points1, weights1, points2, weights2, solver=None):
    """
    2人2財の最適オークション機構設計（シナジーなし）。
    
    画像の数式に基づく実装:
    - 目的関数: Σ_{i=1}^{2} Σ_j w_j (u_{i,j}^T x_j - u_{ij})
    - IC制約: u_1(i) ≥ u_1(j) + p_1(j)(x_1(i) - x_1(j)) ∀i, j
              u_2(i) ≥ u_2(j) + p_2(j)(x_2(i) - x_2(j)) ∀i, j
    - Feasibility制約: 0 ≤ u_{1,a}(x) + u_{2,a}(x) ≤ 1
                       0 ≤ u_{1,b}(x) + u_{2,b}(x) ≤ 1
    
    パラメータ:
        points1: list of tuples, 参加者1の型空間の点 (x₁_a, x₁_b) - 2次元
        weights1: list of floats, 参加者1の各点の重み w₁
        points2: list of tuples, 参加者2の型空間の点 (x₂_a, x₂_b) - 2次元
        weights2: list of floats, 参加者2の各点の重み w₂
        solver: PuLPソルバー（必須: Gurobi）
    
    戻り値:
        (status_string, objective_value, u1, u2, p1, p2)
        - u1: np.ndarray, shape (J1, J2), u1[j1, j2] = 参加者1が型j1で参加者2が型j2のときの参加者1の効用
        - u2: np.ndarray, shape (J1, J2), u2[j1, j2] = 参加者1が型j1で参加者2が型j2のときの参加者2の効用
        - p1: np.ndarray, shape (2, J1, J2), p1[l, j1, j2] = 参加者1への財lの配分確率 (l=0,1)
        - p2: np.ndarray, shape (2, J1, J2), p2[l, j1, j2] = 参加者2への財lの配分確率 (l=0,1)
    """
    J1 = len(points1)
    J2 = len(points2)
    
    # pointsとweightsをNumPy配列に変換
    points1_arr = np.asarray(points1, dtype=np.float64)  # (J1, 2)
    points2_arr = np.asarray(points2, dtype=np.float64)  # (J2, 2)
    weights1_arr = np.asarray(weights1, dtype=np.float64)  # (J1,)
    weights2_arr = np.asarray(weights2, dtype=np.float64)  # (J2,)
    
    # 差分行列を一括計算（ループ外で一度だけ）
    points1_diff = points1_arr[:, None, :] - points1_arr[None, :, :]  # (J1, J1, 2)
    points2_diff = points2_arr[:, None, :] - points2_arr[None, :, :]  # (J2, J2, 2)
    
    # 問題設定
    prob = pulp.LpProblem("RC_2agents_2goods", pulp.LpMaximize)
    
    # ========== 変数の定義 ==========
    # 変数 u1[j1, j2] (参加者1の効用)
    u1 = {
        (j1, j2): pulp.LpVariable(f"u1_{j1}_{j2}", lowBound=0.0, cat=pulp.LpContinuous)
        for j1 in range(J1)
        for j2 in range(J2)
    }
    
    # 変数 u2[j1, j2] (参加者2の効用)
    u2 = {
        (j1, j2): pulp.LpVariable(f"u2_{j1}_{j2}", lowBound=0.0, cat=pulp.LpContinuous)
        for j1 in range(J1)
        for j2 in range(J2)
    }
    
    # 変数 p1[l, j1, j2] (参加者1の配分確率: l=0,1)
    # l=0: 財a, l=1: 財b
    p1 = {
        (l, j1, j2): pulp.LpVariable(f"p1_{l}_{j1}_{j2}", lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
        for l in range(2)
        for j1 in range(J1)
        for j2 in range(J2)
    }
    
    # 変数 p2[l, j1, j2] (参加者2の配分確率: l=0,1)
    p2 = {
        (l, j1, j2): pulp.LpVariable(f"p2_{l}_{j1}_{j2}", lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
        for l in range(2)
        for j1 in range(J1)
        for j2 in range(J2)
    }
    
    # ========== 目的関数 ==========
    # Σ_{i=1}^{2} Σ_j w_j (u_{i,j}^T x_j - u_{ij})
    # = Σ_{j1, j2} w1[j1] * w2[j2] * (
    #     (p1(j1,j2)・x1(j1) - u1(j1,j2)) + (p2(j1,j2)・x2(j2) - u2(j1,j2))
    # )
    objective = pulp.lpSum(
        weights1_arr[j1] * weights2_arr[j2] * (
            # 参加者1の項
            p1[(0, j1, j2)] * points1_arr[j1, 0]  # 財a
            + p1[(1, j1, j2)] * points1_arr[j1, 1]  # 財b
            - u1[(j1, j2)]
            # 参加者2の項
            + p2[(0, j1, j2)] * points2_arr[j2, 0]  # 財a
            + p2[(1, j1, j2)] * points2_arr[j2, 1]  # 財b
            - u2[(j1, j2)]
        )
        for j1 in range(J1)
        for j2 in range(J2)
    )
    prob += objective
    
    # ========== 制約 ==========
    
    # 1. 非負性: u1[j1,j2] >= 0, u2[j1,j2] >= 0
    # 変数の定義で既に lowBound=0.0 として設定済み
    
    # 2. 1-Lipschitz制約: 0 ≤ p1[l,j1,j2] ≤ 1, 0 ≤ p2[l,j1,j2] ≤ 1
    # 変数の定義で既に lowBound=0.0, upBound=1.0 として設定済み
    
    # 3. IC制約（参加者1）: u_1(i) ≥ u_1(j) + p_1(j)(x_1(i) - x_1(j)) ∀i, j
    # 画像の数式では、他のエージェントの型は固定されていないように見えるが、
    # 実際には各(j1, j2)の組み合わせに対してIC制約が必要
    for i1 in range(J1):
        for j1 in range(J1):
            for j2 in range(J2):
                # IC制約: u1(i1, j2) >= u1(j1, j2) + p1(j1, j2)・(x1(i1) - x1(j1))
                prob += u1[(i1, j2)] >= u1[(j1, j2)] + (
                    p1[(0, j1, j2)] * points1_diff[i1, j1, 0]  # 財aの項
                    + p1[(1, j1, j2)] * points1_diff[i1, j1, 1]  # 財bの項
                ), f"ic1_{i1}_{j1}_{j2}"
    
    # 4. IC制約（参加者2）: u_2(i) ≥ u_2(j) + p_2(j)(x_2(i) - x_2(j)) ∀i, j
    for i2 in range(J2):
        for j2 in range(J2):
            for j1 in range(J1):
                # IC制約: u2(j1, i2) >= u2(j1, j2) + p2(j1, j2)・(x2(i2) - x2(j2))
                prob += u2[(j1, i2)] >= u2[(j1, j2)] + (
                    p2[(0, j1, j2)] * points2_diff[i2, j2, 0]  # 財aの項
                    + p2[(1, j1, j2)] * points2_diff[i2, j2, 1]  # 財bの項
                ), f"ic2_{j1}_{i2}_{j2}"
    
    # 5. Feasibility制約: 0 ≤ u_{1,a}(x) + u_{2,a}(x) ≤ 1
    #                    0 ≤ u_{1,b}(x) + u_{2,b}(x) ≤ 1
    # 画像では u_{1,a}(x) と u_{2,a}(x) となっているが、これは配分確率を意味する
    # つまり、p1[0, j1, j2] + p2[0, j1, j2] ≤ 1
    for j1 in range(J1):
        for j2 in range(J2):
            # 財a: 0 ≤ p1[0,j1,j2] + p2[0,j1,j2] ≤ 1
            prob += p1[(0, j1, j2)] + p2[(0, j1, j2)] >= 0, f"feasibility_a_{j1}_{j2}_lower"
            prob += p1[(0, j1, j2)] + p2[(0, j1, j2)] <= 1, f"feasibility_a_{j1}_{j2}_upper"
            # 財b: 0 ≤ p1[1,j1,j2] + p2[1,j1,j2] ≤ 1
            prob += p1[(1, j1, j2)] + p2[(1, j1, j2)] >= 0, f"feasibility_b_{j1}_{j2}_lower"
            prob += p1[(1, j1, j2)] + p2[(1, j1, j2)] <= 1, f"feasibility_b_{j1}_{j2}_upper"
    
    # ========== 解く ==========
    prob.solve(solver)
    
    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)
    
    # ========== 結果 ==========
    # NumPy配列に変換
    u1_sol = np.array([[u1[(j1, j2)].varValue for j2 in range(J2)] for j1 in range(J1)], dtype=np.float64)
    u2_sol = np.array([[u2[(j1, j2)].varValue for j2 in range(J2)] for j1 in range(J1)], dtype=np.float64)
    p1_sol = np.array([[[p1[(l, j1, j2)].varValue for j2 in range(J2)] for j1 in range(J1)] for l in range(2)], dtype=np.float64)
    p2_sol = np.array([[[p2[(l, j1, j2)].varValue for j2 in range(J2)] for j1 in range(J1)] for l in range(2)], dtype=np.float64)
    
    return status, obj_val, u1_sol, u2_sol, p1_sol, p2_sol


def solve_mechanism_multi_agent_iterative(points_list, weights_list, grid_sizes_list, 
                                          solver=None, max_iter=100, tol=1e-6):
    """
    反復的制約追加アルゴリズムによるRochet-Choné型LPの求解（複数エージェント2財版、シナジーなし）。
    
    アルゴリズム:
    1. 局所的なIC制約（ic-local）のみで初期問題を定義
    2. 最適解を求める
    3. 違反している大局的なIC制約を検出（各参加者について独立に）
    4. 違反制約を追加して問題を更新
    5. 違反がなくなるまで反復
    
    パラメータ:
        points_list: list of lists of tuples, 各参加者の型空間の点 [(x_i_a, x_i_b), ...] - 2次元
        weights_list: list of lists of floats, 各参加者の各点の重み [w_i, ...]
        grid_sizes_list: list of tuples, 各参加者の各次元のグリッドサイズ [(nx_i, ny_i), ...]
        solver: PuLPソルバー（必須: Gurobi）
        max_iter: 最大反復回数
        tol: 違反判定の許容誤差
    
    戻り値:
        (status_string, objective_value, u_list, p_list, n_iterations)
    """
    n_agents = len(points_list)
    # 各参加者の型数を取得
    J_list = [len(points) for points in points_list]
    
    # pointsをNumPy配列に変換（一度だけ、高速化のため）
    points_arr_list = [np.array(points, dtype=np.float64) for points in points_list]  # 各要素は (J_i, 2)
    
    # グリッドインデックスを計算する関数（各参加者について）
    def get_grid_indices(i, point_idx):
        """参加者iの点のインデックスからグリッド座標を取得"""
        nx_i, ny_i = grid_sizes_list[i]
        j = point_idx % ny_i
        i_coord = point_idx // ny_i
        return (i_coord, j)
    
    def get_point_idx(i, grid_indices):
        """参加者iのグリッド座標から点のインデックスを取得"""
        nx_i, ny_i = grid_sizes_list[i]
        i_coord, j = grid_indices
        return i_coord * ny_i + j
    
    def get_neighbors(i, point_idx):
        """参加者iの局所的なIC制約に使用する隣接点のインデックスを取得"""
        nx_i, ny_i = grid_sizes_list[i]
        i_coord, j = get_grid_indices(i, point_idx)
        neighbors = []
        for di in [-1, 1]:
            if 0 <= i_coord + di < nx_i:
                neighbor_idx = get_point_idx(i, (i_coord + di, j))
                neighbors.append(neighbor_idx)
        for dj in [-1, 1]:
            if 0 <= j + dj < ny_i:
                neighbor_idx = get_point_idx(i, (i_coord, j + dj))
                neighbors.append(neighbor_idx)
        return neighbors
    
    # ========== 問題の初期化 ==========
    prob = pulp.LpProblem("RC_iterative_multi_agent_2goods", pulp.LpMaximize)
    
    # インデックスをタプルで管理するためのヘルパー関数
    def get_all_indices():
        """全参加者の型の組み合わせを生成"""
        from itertools import product
        return list(product(*[range(J) for J in J_list]))
    
    all_indices = get_all_indices()
    
    # 変数の定義
    u_list = []
    for i in range(n_agents):
        u_i = {
            indices: pulp.LpVariable(f"u{i}_{'_'.join(map(str, indices))}", 
                                     lowBound=0.0, cat=pulp.LpContinuous)
            for indices in all_indices
        }
        u_list.append(u_i)
    
    p_list = []
    for i in range(n_agents):
        p_i = {
            (l, indices): pulp.LpVariable(f"p{i}_{l}_{'_'.join(map(str, indices))}", 
                                         lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
            for l in range(2)
            for indices in all_indices
        }
        p_list.append(p_i)
    
    # 目的関数
    objective_terms = []
    for indices in all_indices:
        weight_product = 1.0
        for i in range(n_agents):
            weight_product *= weights_list[i][indices[i]]
        
        for i in range(n_agents):
            j_i = indices[i]
            term = (
                p_list[i][(0, indices)] * points_list[i][j_i][0]
                + p_list[i][(1, indices)] * points_list[i][j_i][1]
                - u_list[i][indices]
            )
            objective_terms.append(weight_product * term)
    
    objective = pulp.lpSum(objective_terms)
    prob += objective
    
    # 局所的なIC制約（各参加者について）
    local_constraints_list = []
    for i in range(n_agents):
        local_constraints_i = set()
        for i_i in range(J_list[i]):
            neighbors_i = get_neighbors(i, i_i)
            x_i_i = points_arr_list[i][i_i]
            
            other_indices = [range(J_list[j]) for j in range(n_agents) if j != i]
            for other_comb in product(*other_indices):
                indices_i = tuple(list(other_comb[:i]) + [i_i] + list(other_comb[i:]))
                
                for k_i in neighbors_i:
                    x_i_k = points_arr_list[i][k_i]
                    indices_k = tuple(list(other_comb[:i]) + [k_i] + list(other_comb[i:]))
                    
                    prob += u_list[i][indices_i] >= u_list[i][indices_k] + (
                        p_list[i][(0, indices_k)] * (x_i_i[0] - x_i_k[0])
                        + p_list[i][(1, indices_k)] * (x_i_i[1] - x_i_k[1])
                    ), f"ic{i}_local_{i_i}_{k_i}_{'_'.join(map(str, other_comb))}"
                    local_constraints_i.add((i_i, k_i, other_comb))
        
        local_constraints_list.append(local_constraints_i)
    
    # 追加された制約を記録（重複追加を防ぐ）
    added_constraints_list = [constraints.copy() for constraints in local_constraints_list]
    
    # ========== 反復ループ ==========
    for iteration in range(max_iter):
        # 最適解を求める
        prob.solve(solver)
        
        if prob.status != pulp.LpStatusOptimal:
            status = pulp.LpStatus[prob.status]
            return status, None, None, None, iteration
        
        # 解を取得（NumPy配列に直接変換、高速化）
        u_sol_list = []
        p_sol_list = []
        
        for i in range(n_agents):
            u_sol = np.zeros(J_list, dtype=np.float64)
            for indices in all_indices:
                u_sol[indices] = u_list[i][indices].varValue
            u_sol_list.append(u_sol)
            
            p_sol = np.zeros((2,) + tuple(J_list), dtype=np.float64)
            for l in range(2):
                for indices in all_indices:
                    p_sol[(l,) + indices] = p_list[i][(l, indices)].varValue
            p_sol_list.append(p_sol)
        
        # 違反している制約を検出（各参加者について独立に）
        total_violations = 0
        
        for i in range(n_agents):
            violations_i = []
            
            # 参加者iの違反チェック
            for i_i in range(J_list[i]):
                x_i_i = points_arr_list[i][i_i]
                for k_i in range(J_list[i]):
                    x_i_k = points_arr_list[i][k_i]
                    x_i_diff = x_i_i - x_i_k  # (2,)
                    
                    other_indices = [range(J_list[j]) for j in range(n_agents) if j != i]
                    for other_comb in product(*other_indices):
                        if (i_i, k_i, other_comb) in added_constraints_list[i]:
                            continue
                        
                        indices_i = tuple(list(other_comb[:i]) + [i_i] + list(other_comb[i:]))
                        indices_k = tuple(list(other_comb[:i]) + [k_i] + list(other_comb[i:]))
                        
                        u_i_diff = u_sol_list[i][indices_i] - u_sol_list[i][indices_k]
                        inner_product = (p_sol_list[i][(0,) + indices_k] * x_i_diff[0] +
                                         p_sol_list[i][(1,) + indices_k] * x_i_diff[1])
                        
                        if u_i_diff < inner_product - tol:
                            violations_i.append((i_i, k_i, other_comb))
            
            # 違反している制約を追加
            for i_i, k_i, other_comb in violations_i:
                x_i_i = points_arr_list[i][i_i]
                x_i_k = points_arr_list[i][k_i]
                indices_i = tuple(list(other_comb[:i]) + [i_i] + list(other_comb[i:]))
                indices_k = tuple(list(other_comb[:i]) + [k_i] + list(other_comb[i:]))
                
                constraint = u_list[i][indices_i] >= u_list[i][indices_k] + (
                    p_list[i][(0, indices_k)] * (x_i_i[0] - x_i_k[0])
                    + p_list[i][(1, indices_k)] * (x_i_i[1] - x_i_k[1])
                )
                prob += constraint, f"ic{i}_{i_i}_{k_i}_{'_'.join(map(str, other_comb))}_iter{iteration}"
                added_constraints_list[i].add((i_i, k_i, other_comb))
            
            total_violations += len(violations_i)
        
        # 違反がなければ終了
        if total_violations == 0:
            status = pulp.LpStatus[prob.status]
            obj_val = pulp.value(prob.objective)
            return status, obj_val, u_sol_list, p_sol_list, iteration + 1
        
        print(f"Iteration {iteration + 1}: {total_violations} violations found, added {total_violations} constraints")
    
    # 最大反復回数に達した場合
    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)
    return status, obj_val, u_sol_list, p_sol_list, max_iter


def solve_mechanism_2agents_iterative(points1, weights1, grid_sizes1, points2, weights2, grid_sizes2, 
                                       solver=None, max_iter=100, tol=1e-6):
    """
    反復的制約追加アルゴリズムによるRochet-Choné型LPの求解（2人2財版、シナジーなし）。
    
    solve_mechanism_multi_agent_iterativeの2人版のラッパー関数。
    
    パラメータ:
        points1: list of tuples, 参加者1の型空間の点 (x₁_a, x₁_b) - 2次元
        weights1: list of floats, 参加者1の各点の重み
        grid_sizes1: tuple, 参加者1の各次元のグリッドサイズ (nx1, ny1)
        points2: list of tuples, 参加者2の型空間の点 (x₂_a, x₂_b) - 2次元
        weights2: list of floats, 参加者2の各点の重み
        grid_sizes2: tuple, 参加者2の各次元のグリッドサイズ (nx2, ny2)
        solver: PuLPソルバー（必須: Gurobi）
        max_iter: 最大反復回数
        tol: 違反判定の許容誤差
    
    戻り値:
        (status_string, objective_value, u1, u2, p1, p2, n_iterations)
    """
    status, obj_val, u_list, p_list, n_iter = solve_mechanism_multi_agent_iterative(
        [points1, points2], [weights1, weights2], [grid_sizes1, grid_sizes2],
        solver=solver, max_iter=max_iter, tol=tol
    )
    
    if u_list is None:
        return status, obj_val, None, None, None, None, n_iter
    
    u1 = u_list[0]  # (J1, J2)
    u2 = u_list[1]  # (J1, J2)
    p1 = p_list[0]  # (2, J1, J2)
    p2 = p_list[1]  # (2, J1, J2)
    
    return status, obj_val, u1, u2, p1, p2, n_iter


def save_results_multi_agent(
    points_list, weights_list,
    u_list, p_list,
    obj_val, status,
    grid_sizes_list=None, n_iter=None,
    filename=None, data_dir="data"
):
    """
    複数エージェント2財の結果をNumPy形式で保存する。
    
    パラメータ:
        points_list: 各参加者の型空間の点 (list of lists or np.ndarray)
        weights_list: 各参加者の各点の重み (list of lists or np.ndarray)
        u_list: 各参加者の効用 (list of np.ndarray)
        p_list: 各参加者の配分確率 (list of np.ndarray, 各要素のshape: (2, J_1, ..., J_n))
        obj_val: 目的関数値 (float)
        status: LPステータス (str)
        grid_sizes_list: 各参加者のグリッドサイズ (list of tuples, optional)
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
        n_agents = len(points_list)
        filename = f"results_multi_agent_{n_agents}agents_{timestamp}.npz"
    
    filepath = os.path.join(data_dir, filename)
    
    # NumPy配列に変換
    points_arr_list = [np.array(points, dtype=np.float64) for points in points_list]
    weights_arr_list = [np.array(weights, dtype=np.float64) for weights in weights_list]
    u_arr_list = [np.array(u, dtype=np.float64) for u in u_list]
    p_arr_list = [np.array(p, dtype=np.float64) for p in p_list]
    
    # NumPy形式で保存（メタデータも含める）
    save_dict = {
        'n_agents': np.array([len(points_list)], dtype=np.int32),
        'obj_val': np.array([obj_val], dtype=np.float64) if obj_val is not None else np.array([np.nan], dtype=np.float64),
    }
    
    # 各参加者のデータを保存
    for i in range(len(points_list)):
        save_dict[f'points{i}'] = points_arr_list[i]
        save_dict[f'weights{i}'] = weights_arr_list[i]
        save_dict[f'u{i}'] = u_arr_list[i]
        save_dict[f'p{i}'] = p_arr_list[i]
        save_dict[f'J{i}'] = np.array([len(points_list[i])], dtype=np.int32)
    
    if grid_sizes_list is not None:
        for i in range(len(grid_sizes_list)):
            save_dict[f'grid_sizes{i}'] = np.array(grid_sizes_list[i], dtype=np.int32)
    if n_iter is not None:
        save_dict['n_iter'] = np.array([n_iter], dtype=np.int32)
    
    # メタデータファイルも保存（文字列情報用）
    metadata_filepath = filepath.replace('.npz', '_metadata.txt')
    with open(metadata_filepath, 'w') as f:
        f.write(f"status: {status}\n")
        f.write(f"obj_val: {obj_val}\n")
        f.write(f"n_agents: {len(points_list)}\n")
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
        if grid_sizes_list is not None:
            for i, grid_sizes in enumerate(grid_sizes_list):
                f.write(f"grid_sizes{i}: {grid_sizes}\n")
        if n_iter is not None:
            f.write(f"n_iter: {n_iter}\n")
    
    np.savez_compressed(filepath, **save_dict)
    
    print(f"Results saved to: {filepath}")
    return filepath


def save_results_2agents(
    points1, weights1, points2, weights2,
    u1_sol, u2_sol, p1_sol, p2_sol,
    obj_val, status,
    grid_sizes1=None, grid_sizes2=None, n_iter=None,
    filename=None, data_dir="data"
):
    """
    2人2財の結果をNumPy形式で保存する。
    
    save_results_multi_agentの2人版のラッパー関数。
    
    パラメータ:
        points1: 参加者1の型空間の点 (list or np.ndarray)
        weights1: 参加者1の各点の重み (list or np.ndarray)
        points2: 参加者2の型空間の点 (list or np.ndarray)
        weights2: 参加者2の各点の重み (list or np.ndarray)
        u1_sol: 参加者1の効用 (np.ndarray, shape: (J1, J2))
        u2_sol: 参加者2の効用 (np.ndarray, shape: (J1, J2))
        p1_sol: 参加者1の配分確率 (np.ndarray, shape: (2, J1, J2))
        p2_sol: 参加者2の配分確率 (np.ndarray, shape: (2, J1, J2))
        obj_val: 目的関数値 (float)
        status: LPステータス (str)
        grid_sizes1: 参加者1のグリッドサイズ (tuple, optional)
        grid_sizes2: 参加者2のグリッドサイズ (tuple, optional)
        n_iter: 反復回数 (int, optional)
        filename: 保存ファイル名 (str, optional, 指定しない場合は自動生成)
        data_dir: データ保存ディレクトリ (str, default: "data")
    
    戻り値:
        filepath: 保存されたファイルのパス (str)
    """
    return save_results_multi_agent(
        [points1, points2], [weights1, weights2],
        [u1_sol, u2_sol], [p1_sol, p2_sol],
        obj_val, status,
        grid_sizes_list=[grid_sizes1, grid_sizes2] if grid_sizes1 is not None and grid_sizes2 is not None else None,
        n_iter=n_iter,
        filename=filename,
        data_dir=data_dir
    )


def load_results_multi_agent(filepath):
    """
    保存された結果を読み込む。
    
    パラメータ:
        filepath: 保存されたファイルのパス (str)
    
    戻り値:
        dict: 読み込んだデータの辞書
    """
    data = np.load(filepath, allow_pickle=True)
    
    n_agents = int(data['n_agents'][0])
    
    result = {
        'n_agents': n_agents,
        'obj_val': float(data['obj_val'][0]) if not np.isnan(data['obj_val'][0]) else None,
        'points_list': [],
        'weights_list': [],
        'u_list': [],
        'p_list': [],
        'J_list': [],
    }
    
    for i in range(n_agents):
        result['points_list'].append(data[f'points{i}'])
        result['weights_list'].append(data[f'weights{i}'])
        result['u_list'].append(data[f'u{i}'])
        result['p_list'].append(data[f'p{i}'])
        result['J_list'].append(int(data[f'J{i}'][0]))
    
    if 'grid_sizes0' in data:
        result['grid_sizes_list'] = []
        for i in range(n_agents):
            result['grid_sizes_list'].append(tuple(data[f'grid_sizes{i}']))
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


def load_results_2agents(filepath):
    """
    保存された結果を読み込む（2人版）。
    
    load_results_multi_agentの2人版のラッパー関数。
    
    パラメータ:
        filepath: 保存されたファイルのパス (str)
    
    戻り値:
        dict: 読み込んだデータの辞書
    """
    data = load_results_multi_agent(filepath)
    
    # 2人版の互換性のための変換
    result = {
        'points1': data['points_list'][0],
        'weights1': data['weights_list'][0],
        'points2': data['points_list'][1],
        'weights2': data['weights_list'][1],
        'u1_sol': data['u_list'][0],
        'u2_sol': data['u_list'][1],
        'p1_sol': data['p_list'][0],
        'p2_sol': data['p_list'][1],
        'obj_val': data['obj_val'],
        'J1': data['J_list'][0],
        'J2': data['J_list'][1],
    }
    
    if 'grid_sizes_list' in data:
        result['grid_sizes1'] = data['grid_sizes_list'][0]
        result['grid_sizes2'] = data['grid_sizes_list'][1]
    if 'n_iter' in data:
        result['n_iter'] = data['n_iter']
    if 'status' in data:
        result['status'] = data['status']
    if 'timestamp' in data:
        result['timestamp'] = data['timestamp']
    
    return result

