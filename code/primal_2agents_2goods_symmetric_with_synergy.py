"""
Primal問題: Rochet-Choné型LP
2人2財1シナジーの最適オークション機構設計（対称性制約あり）

primal_2agents_2goods_with_synergy.pyを改良:
- 対称性制約: u1 = u2, p1 = p2（points1 = points2, weights1 = weights2を仮定）
- 参加者2のIC制約は冗長（対称性により）
- シナジーの配分制約も対称性により1つだけ

対称性制約:
- points1 = points2, weights1 = weights2のとき、u1 = u2, p1 = p2を強制
- これにより、参加者1と参加者2が対称な扱いを受ける
"""

import pulp
import numpy as np
import os
from datetime import datetime


def solve_mechanism_symmetry_2agents_with_synergy(points1, weights1, points2, weights2, solver=None):
    """
    Rochet–Choné 型の LP を構築して解く（2人2財1シナジー版、対称性制約あり）。
    
    仕様:
    - 参加者1: 型 (x₁_a, x₁_b, x₁_α) ∈ [0,1]³
    - 参加者2: 型 (x₂_a, x₂_b, x₂_α) ∈ [0,1]³
    - 対称性制約: u1 = u2, p1 = p2（points1 = points2, weights1 = weights2を仮定）

    型数: 
        J1 = len(points1), J2 = len(points2)
    財数: 各参加者が2財1シナジー（pointsの次元は3である必要がある）

    変数:
        u[j1, j2]   : 両参加者の効用（対称性により同じ）
        p[l, j1, j2]: 両参加者の配分確率（対称性により同じ）
                       l=0: 財a, l=1: 財b, l=2: シナジーα (0<=p<=1)

    目的関数:
        max Σ_{j1, j2} w1[j1] * w2[j2] * (
            p(j1,j2)・x1(j1) - u(j1,j2) + p(j1,j2)・x2(j2) - u(j1,j2)
        )

    制約:
        - 非負性: u[j1,j2] >= 0
        - IC（参加者1についてのみ）: u(i1, j2) >= u(k1, j2) + p(k1, j2)・(x1(i1) - x1(k1))
        - 1-Lipschitz: 0 <= p[l,j1,j2] <= 1
        - シナジーの配分制約: p[2,j1,j2] <= p[0,j1,j2], p[2,j1,j2] <= p[1,j1,j2], 
                            p[2,j1,j2] >= p[0,j1,j2] + p[1,j1,j2] - 1
        - 対称性: u1 = u2, p1 = p2（変数として同じものを使用）

    パラメータ:
        points1: list of tuples, 参加者1の型空間の点 [(x₁_a, x₁_b, x₁_α), ...] - 3次元
        weights1: list of floats, 参加者1の各点の重み w₁
        points2: list of tuples, 参加者2の型空間の点 [(x₂_a, x₂_b, x₂_α), ...] - 3次元
        weights2: list of floats, 参加者2の各点の重み w₂
        solver: PuLPソルバー（必須: Gurobi）
    
    戻り値:
        (status_string, objective_value, u1, u2, p1, p2)
        - u1: np.ndarray, shape (J1, J2), u1[j1, j2] = 参加者1が型j1で参加者2が型j2のときの参加者1の効用
        - u2: np.ndarray, shape (J1, J2), u2[j1, j2] = 参加者1が型j1で参加者2が型j2のときの参加者2の効用
        - p1: np.ndarray, shape (3, J1, J2), p1[l, j1, j2] = 参加者1への財lの配分確率 (l=0,1,2)
        - p2: np.ndarray, shape (3, J1, J2), p2[l, j1, j2] = 参加者2への財lの配分確率 (l=0,1,2)
    """
    J1 = len(points1)
    J2 = len(points2)
    
    # pointsとweightsをNumPy配列に変換（一度だけ、高速化のため）
    # 対称性により、points1 = points2、weights1 = weights2なので、1つだけ使用
    points1_arr = np.asarray(points1, dtype=np.float64)  # (J1, 3)
    weights1_arr = np.asarray(weights1, dtype=np.float64)  # (J1,)
    points2_arr = points1_arr  # 参照のみ（メモリ節約）
    weights2_arr = weights1_arr  # 参照のみ（メモリ節約）

    # 差分行列を一括計算（ループ外で一度だけ）
    # 対称性により、points1 = points2なので、1つの差分行列のみ計算
    points1_diff = points1_arr[:, None, :] - points1_arr[None, :, :]  # (J1, J1, 3)
    
    # 問題設定
    prob = pulp.LpProblem("RC_symmetry_2agents_2goods_1synergy", pulp.LpMaximize)

    # ========== 変数の定義 ==========
    # 対称性制約により、u1とu2は同じ変数、p1とp2も同じ変数として定義
    # 変数 u[j1, j2] (両参加者の効用、対称性により同じ)
    u = {
        (j1, j2): pulp.LpVariable(f"u_{j1}_{j2}", 
                                 lowBound=0.0, cat=pulp.LpContinuous)
        for j1 in range(J1)
        for j2 in range(J2)
    }
    
    # 変数 p[l, j1, j2] (両参加者の配分確率、対称性により同じ)
    # l=0: 財a, l=1: 財b, l=2: シナジーα
    p = {
        (l, j1, j2): pulp.LpVariable(f"p_{l}_{j1}_{j2}", 
                                     lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
        for l in range(3)
        for j1 in range(J1)
        for j2 in range(J2)
    }

    # ========== 目的関数 ==========
    # max Σ_{j1, j2} w1[j1] * w2[j2] * (
    #     p(j1,j2)・x1(j1) - u(j1,j2) + p(j1,j2)・x2(j2) - u(j1,j2)
    # )
    # = max Σ_{j1, j2} w1[j1] * w2[j2] * (
    #     p(j1,j2)・(x1(j1) + x2(j2)) - 2*u(j1,j2)
    # )
    # 対称性により、u[(j1,j2)] = u[(j2,j1)]、p[(l,j1,j2)] = p[(l,j2,j1)]なので、
    # j1 <= j2の範囲で計算し、j1 != j2の場合は2倍にする
    objective = pulp.lpSum(
        (2.0 if j1 != j2 else 1.0) * weights1_arr[j1] * weights1_arr[j2] * (
            p[(0, j1, j2)] * (points1_arr[j1, 0] + points1_arr[j2, 0])  # 財aの価値の和
            + p[(1, j1, j2)] * (points1_arr[j1, 1] + points1_arr[j2, 1])  # 財bの価値の和
            + p[(2, j1, j2)] * (points1_arr[j1, 2] + points1_arr[j2, 2])  # シナジーαの価値の和
            - 2.0 * u[(j1, j2)]  # 対称性によりu1 = u2 = u
        )
        for j1 in range(J1)
        for j2 in range(j1, J2)  # j1 <= j2の範囲のみ
    )
    prob += objective

    # ========== 制約 ==========
    
    # 1. 非負性: u[j1,j2] >= 0
    # 変数の定義で既に lowBound=0.0 として設定済み
    
    # 2. 1-Lipschitz制約: 0 ≤ p[l,j1,j2] ≤ 1
    # 変数の定義で既に lowBound=0.0, upBound=1.0 として設定済み
    
    # 3. IC制約（参加者1についてのみ）
    # 対称性により、u1 = u2、p1 = p2、points1 = points2なので、
    # 参加者2のIC制約は参加者1のIC制約と実質的に同じ（冗長）
    # 差分行列をキャッシュから参照して高速化
    for i1 in range(J1):
        for k1 in range(J1):
            for j2 in range(J2):
                # IC制約: u1(i1, j2) >= u1(k1, j2) + p1(k1, j2)・(x1(i1) - x1(k1))
                # 対称性により、これは参加者2のIC制約も満たす
                # 差分行列から直接参照（points1_diff[i1, k1, :]）
                prob += u[(i1, j2)] >= u[(k1, j2)] + (
                    p[(0, k1, j2)] * points1_diff[i1, k1, 0]  # 財aの項
                    + p[(1, k1, j2)] * points1_diff[i1, k1, 1]  # 財bの項
                    + p[(2, k1, j2)] * points1_diff[i1, k1, 2]  # シナジーαの項
                ), f"ic_{i1}_{k1}_{j2}"
    
    # 4. シナジーの配分制約（対称性により1つだけ）
    # p[2,j1,j2] <= p[0,j1,j2], p[2,j1,j2] <= p[1,j1,j2]
    # p[2,j1,j2] >= p[0,j1,j2] + p[1,j1,j2] - 1
    for j1 in range(J1):
        for j2 in range(J2):
            # 上界制約
            prob += p[(2, j1, j2)] <= p[(0, j1, j2)], f"synergy_item0_{j1}_{j2}_upper"
            prob += p[(2, j1, j2)] <= p[(1, j1, j2)], f"synergy_item1_{j1}_{j2}_upper"
            # 下界制約（inclusion-exclusion）
            prob += p[(2, j1, j2)] >= p[(0, j1, j2)] + p[(1, j1, j2)] - 1, f"synergy_{j1}_{j2}_lower"

    # ========== 解く ==========
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)

    # ========== 結果 ==========
    # NumPy配列に変換（np.fromiterを使用してPythonループを削減）
    # 対称性により、u1 = u2、p1 = p2なので、u_solとp_solのみを計算
    u_flat = np.fromiter((u[(j1, j2)].varValue for j1 in range(J1) for j2 in range(J2)), 
                        dtype=np.float64, count=J1*J2)
    u_sol = u_flat.reshape(J1, J2)
    
    # pの形状: (3, J1, J2)
    p_sol = np.zeros((3, J1, J2), dtype=np.float64)
    for l in range(3):
        p_flat = np.fromiter((p[(l, j1, j2)].varValue for j1 in range(J1) for j2 in range(J2)),
                            dtype=np.float64, count=J1*J2)
        p_sol[l] = p_flat.reshape(J1, J2)
    
    # 戻り値のインターフェースを維持（対称性により参照のみ）
    u1_sol = u_sol
    u2_sol = u_sol  # 対称性により同じ（参照のみ）
    p1_sol = p_sol
    p2_sol = p_sol  # 対称性により同じ（参照のみ）

    return status, obj_val, u1_sol, u2_sol, p1_sol, p2_sol


def solve_mechanism_symmetry_2agents_with_synergy_iterative(points1, weights1, grid_sizes1, points2, weights2, grid_sizes2,
                                                             solver=None, max_iter=100, tol=1e-6):
    """
    反復的制約追加アルゴリズムによるRochet-Choné型LPの求解（2人2財1シナジー版、対称性制約あり）。
    
    アルゴリズム:
    1. 局所的なIC制約（ic-local）のみで初期問題を定義
    2. 最適解を求める
    3. 違反している大局的なIC制約を検出
    4. 違反制約を追加して問題を更新
    5. 違反がなくなるまで反復
    
    パラメータ:
        points1: list of tuples, 参加者1の型空間の点 (x₁_a, x₁_b, x₁_α) - 3次元
        weights1: list of floats, 参加者1の各点の重み
        grid_sizes1: tuple, 参加者1の各次元のグリッドサイズ (nx1, ny1, nz1)
        points2: list of tuples, 参加者2の型空間の点 (x₂_a, x₂_b, x₂_α) - 3次元
        weights2: list of floats, 参加者2の各点の重み
        grid_sizes2: tuple, 参加者2の各次元のグリッドサイズ (nx2, ny2, nz2)
        solver: PuLPソルバー（必須: Gurobi）
        max_iter: 最大反復回数
        tol: 違反判定の許容誤差
    
    戻り値:
        (status_string, objective_value, u1, u2, p1, p2, n_iterations)
    """
    J1 = len(points1)
    J2 = len(points2)
    nx1, ny1, nz1 = grid_sizes1
    nx2, ny2, nz2 = grid_sizes2
    
    # pointsをNumPy配列に変換
    # 対称性により、points1 = points2なので、1つだけ使用
    points1_arr = np.asarray(points1, dtype=np.float64)  # (J1, 3)
    points2_arr = points1_arr  # 参照のみ（メモリ節約）
    weights1_arr = np.asarray(weights1, dtype=np.float64)  # (J1,)
    weights2_arr = weights1_arr  # 参照のみ（メモリ節約）
    
    # グリッドインデックスを計算する関数
    def get_grid_indices_1(point_idx):
        """参加者1の点のインデックスからグリッド座標を取得"""
        k = point_idx % nz1
        j = (point_idx // nz1) % ny1
        i_coord = point_idx // (ny1 * nz1)
        return (i_coord, j, k)
    
    def get_point_idx_1(grid_indices):
        """参加者1のグリッド座標から点のインデックスを取得"""
        i_coord, j, k = grid_indices
        return i_coord * (ny1 * nz1) + j * nz1 + k
    
    def get_neighbors_1(point_idx):
        """参加者1の局所的なIC制約に使用する隣接点のインデックスを取得"""
        i_coord, j, k = get_grid_indices_1(point_idx)
        neighbors = []
        for di in [-1, 1]:
            if 0 <= i_coord + di < nx1:
                neighbor_idx = get_point_idx_1((i_coord + di, j, k))
                neighbors.append(neighbor_idx)
        for dj in [-1, 1]:
            if 0 <= j + dj < ny1:
                neighbor_idx = get_point_idx_1((i_coord, j + dj, k))
                neighbors.append(neighbor_idx)
        for dk in [-1, 1]:
            if 0 <= k + dk < nz1:
                neighbor_idx = get_point_idx_1((i_coord, j, k + dk))
                neighbors.append(neighbor_idx)
        return neighbors
    
    # ========== 問題の初期化 ==========
    prob = pulp.LpProblem("RC_iterative_symmetry_2agents_2goods_1synergy", pulp.LpMaximize)
    
    # 変数の定義（対称性により同じ変数を使用）
    u = {
        (j1, j2): pulp.LpVariable(f"u_{j1}_{j2}", 
                                 lowBound=0.0, cat=pulp.LpContinuous)
        for j1 in range(J1)
        for j2 in range(J2)
    }
    
    p = {
        (l, j1, j2): pulp.LpVariable(f"p_{l}_{j1}_{j2}", 
                                     lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
        for l in range(3)
        for j1 in range(J1)
        for j2 in range(J2)
    }
    
    # 目的関数（配列アクセスを使用して高速化）
    # 対称性により、weights1 = weights2、points1 = points2
    # さらに、u[(j1,j2)] = u[(j2,j1)]、p[(l,j1,j2)] = p[(l,j2,j1)]なので、
    # j1 <= j2の範囲で計算し、j1 != j2の場合は2倍にする
    objective = pulp.lpSum(
        (2.0 if j1 != j2 else 1.0) * weights1_arr[j1] * weights1_arr[j2] * (
            p[(0, j1, j2)] * (points1_arr[j1, 0] + points1_arr[j2, 0])
            + p[(1, j1, j2)] * (points1_arr[j1, 1] + points1_arr[j2, 1])
            + p[(2, j1, j2)] * (points1_arr[j1, 2] + points1_arr[j2, 2])
            - 2.0 * u[(j1, j2)]
        )
        for j1 in range(J1)
        for j2 in range(j1, J2)  # j1 <= j2の範囲のみ
    )
    prob += objective
    
    # シナジーの配分制約（対称性により1つだけ）
    for j1 in range(J1):
        for j2 in range(J2):
            prob += p[(2, j1, j2)] <= p[(0, j1, j2)], f"synergy_item0_{j1}_{j2}_upper"
            prob += p[(2, j1, j2)] <= p[(1, j1, j2)], f"synergy_item1_{j1}_{j2}_upper"
            prob += p[(2, j1, j2)] >= p[(0, j1, j2)] + p[(1, j1, j2)] - 1, f"synergy_{j1}_{j2}_lower"
    
    # 差分行列をループ外で一度だけ計算（反復中は不変）
    # 対称性により、points1 = points2なので、1つの差分行列のみ計算
    points1_diff = points1_arr[:, None, :] - points1_arr[None, :, :]  # (J1, J1, 3)
    
    # 局所的なIC制約（参加者1についてのみ）
    # 対称性により、参加者2のIC制約は冗長
    local_constraints = set()
    for i1 in range(J1):
        neighbors_1 = get_neighbors_1(i1)
        for j2 in range(J2):
            for k1 in neighbors_1:
                # 差分行列から直接参照（points1_diff[i1, k1, :]）
                prob += u[(i1, j2)] >= u[(k1, j2)] + (
                    p[(0, k1, j2)] * points1_diff[i1, k1, 0]
                    + p[(1, k1, j2)] * points1_diff[i1, k1, 1]
                    + p[(2, k1, j2)] * points1_diff[i1, k1, 2]
                ), f"ic_local_{i1}_{k1}_{j2}"
                local_constraints.add((i1, k1, j2))
    
    # 追加された制約を記録（参加者1のみ）
    added_constraints = local_constraints.copy()
    
    # 制約マスクをループ外で初期化（違反分だけ更新する方式）
    constraint_mask = np.zeros((J1, J1, J2), dtype=bool)
    if added_constraints:
        constraint_pairs = np.array(list(added_constraints), dtype=np.int32)
        if len(constraint_pairs) > 0:
            constraint_mask[constraint_pairs[:, 0], constraint_pairs[:, 1], constraint_pairs[:, 2]] = True
    
    # ========== 反復ループ ==========
    for iteration in range(max_iter):
        # 最適解を求める
        prob.solve(solver)
        
        if prob.status != pulp.LpStatusOptimal:
            status = pulp.LpStatus[prob.status]
            return status, None, None, None, None, None, iteration
        
        # 解を取得（np.fromiterを使用してPythonループを削減）
        # 対称性により、u1 = u2、p1 = p2なので、u_solとp_solのみを計算
        u_flat = np.fromiter((u[(j1, j2)].varValue for j1 in range(J1) for j2 in range(J2)),
                            dtype=np.float64, count=J1*J2)
        u_sol = u_flat.reshape(J1, J2)
        
        p_sol = np.zeros((3, J1, J2), dtype=np.float64)
        for l in range(3):
            p_flat = np.fromiter((p[(l, j1, j2)].varValue for j1 in range(J1) for j2 in range(J2)),
                                dtype=np.float64, count=J1*J2)
            p_sol[l] = p_flat.reshape(J1, J2)
        
        # 違反している制約を検出（参加者1についてのみ）
        # 対称性により、参加者2のIC制約は冗長
        # ベクトル化で高速化
        violations = []
        
        # 全ての(i1, k1, j2)の組み合わせに対してベクトル演算
        # u_diff[i1, k1, j2] = u_sol[i1, j2] - u_sol[k1, j2]
        u_diff = u_sol[:, None, :] - u_sol[None, :, :]  # (J1, J1, J2)
        
        # points1_diffは既にループ外で計算済み（再計算不要）
        
        # inner_productをeinsumで計算（高速化）
        # inner_product[i1, k1, j2] = Σ_l p_sol[l, k1, j2] * points1_diff[i1, k1, l]
        inner_product = np.einsum('lkj,ikl->ikj', p_sol, points1_diff)  # 結果の形状: (J1, J1, J2)
        
        # 違反チェック: u_i1 - u_k1 < p_k1・(x_i1 - x_k1) - tol
        violation_mask = (u_diff < inner_product - tol) & (~constraint_mask)
        
        # 違反している(i1, k1, j2)のペアを取得（np.argwhereで直接取得）
        violation_pairs = np.argwhere(violation_mask)  # (N_violations, 3)
        violations = [(int(i1), int(k1), int(j2)) for i1, k1, j2 in violation_pairs]
        
        # 違反している制約を追加（バッチ処理で高速化）
        # 差分行列から直接参照して高速化
        for i1, k1, j2 in violations:
            # 差分行列から直接参照（points1_diff[i1, k1, :]）
            constraint = u[(i1, j2)] >= u[(k1, j2)] + (
                p[(0, k1, j2)] * points1_diff[i1, k1, 0]
                + p[(1, k1, j2)] * points1_diff[i1, k1, 1]
                + p[(2, k1, j2)] * points1_diff[i1, k1, 2]
            )
            prob += constraint, f"ic_{i1}_{k1}_{j2}_iter{iteration}"
            added_constraints.add((i1, k1, j2))
            # 制約マスクを更新（違反分だけTrueに設定）
            constraint_mask[i1, k1, j2] = True
        
        total_violations = len(violations)
        
        # 違反がなければ終了
        if total_violations == 0:
            status = pulp.LpStatus[prob.status]
            obj_val = pulp.value(prob.objective)
            
            # 戻り値のインターフェースを維持（対称性により参照のみ）
            u1_sol = u_sol
            u2_sol = u_sol  # 対称性により同じ（参照のみ）
            p1_sol = p_sol
            p2_sol = p_sol  # 対称性により同じ（参照のみ）
            
            return status, obj_val, u1_sol, u2_sol, p1_sol, p2_sol, iteration + 1
        
        print(f"Iteration {iteration + 1}: {total_violations} violations found, added {total_violations} constraints")
    
    # 最大反復回数に達した場合
    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)
    
    # 戻り値のインターフェースを維持（対称性により参照のみ）
    u1_sol = u_sol
    u2_sol = u_sol  # 対称性により同じ（参照のみ）
    p1_sol = p_sol
    p2_sol = p_sol  # 対称性により同じ（参照のみ）
    
    return status, obj_val, u1_sol, u2_sol, p1_sol, p2_sol, max_iter


def save_results_symmetry_2agents_with_synergy(
    points1, weights1, points2, weights2,
    u1_sol, u2_sol, p1_sol, p2_sol,
    obj_val, status,
    grid_sizes1=None, grid_sizes2=None, n_iter=None,
    filename=None, data_dir="data"
):
    """
    2人2財1シナジーの結果をNumPy形式で保存する（対称性制約あり）。
    
    パラメータ:
        points1: 参加者1の型空間の点 (list or np.ndarray)
        weights1: 参加者1の各点の重み (list or np.ndarray)
        points2: 参加者2の型空間の点 (list or np.ndarray)
        weights2: 参加者2の各点の重み (list or np.ndarray)
        u1_sol: 参加者1の効用 (np.ndarray, shape: (J1, J2))
        u2_sol: 参加者2の効用 (np.ndarray, shape: (J1, J2))
        p1_sol: 参加者1の配分確率 (np.ndarray, shape: (3, J1, J2))
        p2_sol: 参加者2の配分確率 (np.ndarray, shape: (3, J1, J2))
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
    # データディレクトリを作成
    os.makedirs(data_dir, exist_ok=True)
    
    # ファイル名を生成
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_symmetry_2agents_synergy_{timestamp}.npz"
    
    filepath = os.path.join(data_dir, filename)
    
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


def load_results_symmetry_2agents_with_synergy(filepath):
    """
    保存された結果を読み込む（対称性制約あり、シナジーあり）。
    
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

