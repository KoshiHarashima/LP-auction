"""
Primal問題: Rochet-Choné型LP
2人2財1シナジーの最適オークション機構設計

Cursor.mdに基づく実装:
2人の参加者（i = 1, 2）がそれぞれ2財1シナジーを持つケースを扱う。
各参加者の型は独立に分布から抽出され、メカニズムは両者の型の組み合わせに依存する。
"""

import pulp
import numpy as np
import os
from datetime import datetime


def solve_mechanism_2agents(points1, weights1, points2, weights2, solver=None):
    """
    Rochet–Choné 型の LP を構築して解く（2人2財1シナジー版）。
    
    Cursor.mdの仕様に基づく:
    - 参加者1: 型 (x₁_a, x₁_b, x₁_α) ∈ [0,1]³
    - 参加者2: 型 (x₂_a, x₂_b, x₂_α) ∈ [0,1]³
    - 各参加者の型は独立に分布から抽出される
    - メカニズムは両者の型の組み合わせ (x₁, x₂) に依存する

    型数: 
        J1 = len(points1) (参加者1の型数)
        J2 = len(points2) (参加者2の型数)
    財数: 各参加者が2財1シナジー（pointsの次元は3である必要がある）

    変数:
        u1[j1, j2]   : 参加者1が型j1で参加者2が型j2のときの参加者1のinterim utility (>=0)
        u2[j1, j2]   : 参加者1が型j1で参加者2が型j2のときの参加者2のinterim utility (>=0)
        p1[l, j1, j2]: 参加者1への配分確率のベクトル（u1の勾配）
                       l=0: 財a, l=1: 財b, l=2: シナジーα (0<=p<=1)
        p2[l, j1, j2]: 参加者2への配分確率のベクトル（u2の勾配）
                       l=0: 財a, l=1: 財b, l=2: シナジーα (0<=p<=1)

    目的関数:
        max Σ_{j1, j2} w1[j1] w2[j2] ((p1(j1,j2)・x1(j1) - u1(j1,j2)) + (p2(j1,j2)・x2(j2) - u2(j1,j2)))
        = max Σ_{j1, j2} w1[j1] w2[j2] (
            p1[0,j1,j2]*x1[j1,0] + p1[1,j1,j2]*x1[j1,1] + p1[2,j1,j2]*x1[j1,2] - u1[j1,j2]
            + p2[0,j1,j2]*x2[j2,0] + p2[1,j1,j2]*x2[j2,1] + p2[2,j1,j2]*x2[j2,2] - u2[j1,j2]
        )

    制約:
        - 非負性: u1[j1,j2] >= 0, u2[j1,j2] >= 0 (全てのj1, j2で)
        - IC（参加者1）: u1(i1, j2) >= u1(k1, j2) + p1(k1, j2)・(x1(i1) - x1(k1)) for all i1, k1, j2
        - IC（参加者2）: u2(j1, i2) >= u2(j1, k2) + p2(j1, k2)・(x2(i2) - x2(k2)) for all i2, k2, j1
        - 1-Lipschitz: 0 <= p1[l,j1,j2] <= 1, 0 <= p2[l,j1,j2] <= 1
        - シナジーの配分制約（参加者1）:
            p1[2,j1,j2] <= p1[0,j1,j2]  (u₁_α ≤ u₁_a)
            p1[2,j1,j2] <= p1[1,j1,j2]  (u₁_α ≤ u₁_b)
            p1[2,j1,j2] >= p1[0,j1,j2] + p1[1,j1,j2] - 1  (u₁_α ≥ u₁_a + u₁_b - 1)
        - シナジーの配分制約（参加者2）:
            p2[2,j1,j2] <= p2[0,j1,j2]  (u₂_α ≤ u₂_a)
            p2[2,j1,j2] <= p2[1,j1,j2]  (u₂_α ≤ u₂_b)
            p2[2,j1,j2] >= p2[0,j1,j2] + p2[1,j1,j2] - 1  (u₂_α ≥ u₂_a + u₂_b - 1)
        - IR: u1(x) = 0, u2(x) = 0 for all x
              （実装では、変数の定義時に lowBound=0.0 として設定し、IR制約は明示的に追加しない）

    パラメータ:
        points1: list of tuples, 参加者1の型空間の点 (x₁_a, x₁_b, x₁_α) - 3次元
        weights1: list of floats, 参加者1の各点の重み w₁
        points2: list of tuples, 参加者2の型空間の点 (x₂_a, x₂_b, x₂_α) - 3次元
        weights2: list of floats, 参加者2の各点の重み w₂
        solver: PuLPソルバー（必須: Gurobi）

    戻り値:
        (status_string, objective_value, u1, u2, p1, p2)
        - u1: list of lists, u1[j1][j2] = 参加者1が型j1で参加者2が型j2のときの参加者1の効用
        - u2: list of lists, u2[j1][j2] = 参加者1が型j1で参加者2が型j2のときの参加者2の効用
        - p1: list of lists of lists, p1[l][j1][j2] = 参加者1への財lの配分確率 (l=0,1,2)
        - p2: list of lists of lists, p2[l][j1][j2] = 参加者2への財lの配分確率 (l=0,1,2)
    """
    J1 = len(points1)
    J2 = len(points2)
    assert J1 == len(weights1)
    assert J2 == len(weights2)
    assert len(points1[0]) == 3, "points1は3次元である必要があります (財a, 財b, シナジーα)"
    assert len(points2[0]) == 3, "points2は3次元である必要があります (財a, 財b, シナジーα)"

    # 問題設定
    prob = pulp.LpProblem("RC_2agents_2goods_1synergy", pulp.LpMaximize)

    # ========== 変数の定義 ==========
    
    # 変数 u1[j1, j2] (参加者1の効用、連続変数)
    u1 = {
        (j1, j2): pulp.LpVariable(f"u1_{j1}_{j2}", lowBound=0.0, cat=pulp.LpContinuous)
        for j1 in range(J1)
        for j2 in range(J2)
    }

    # 変数 u2[j1, j2] (参加者2の効用、連続変数)
    u2 = {
        (j1, j2): pulp.LpVariable(f"u2_{j1}_{j2}", lowBound=0.0, cat=pulp.LpContinuous)
        for j1 in range(J1)
        for j2 in range(J2)
    }

    # 変数 p1[l, j1, j2] (参加者1の配分確率のベクトル: l=0,1,2)
    # l=0: 財a, l=1: 財b, l=2: シナジーα
    p1 = {
        (l, j1, j2): pulp.LpVariable(f"p1_{l}_{j1}_{j2}", lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
        for l in range(3)
        for j1 in range(J1)
        for j2 in range(J2)
    }

    # 変数 p2[l, j1, j2] (参加者2の配分確率のベクトル: l=0,1,2)
    p2 = {
        (l, j1, j2): pulp.LpVariable(f"p2_{l}_{j1}_{j2}", lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
        for l in range(3)
        for j1 in range(J1)
        for j2 in range(J2)
    }

    # ========== 目的関数 ==========
    # Cursor.md: max Σ_{j1, j2} w1[j1] w2[j2] ((p1(j1,j2)・x1(j1) - u1(j1,j2)) + (p2(j1,j2)・x2(j2) - u2(j1,j2)))
    objective = pulp.lpSum(
        weights1[j1] * weights2[j2] * (
            # 参加者1の項
            p1[(0, j1, j2)] * points1[j1][0]  # 財aの価値
            + p1[(1, j1, j2)] * points1[j1][1]  # 財bの価値
            + p1[(2, j1, j2)] * points1[j1][2]  # シナジーαの価値
            - u1[(j1, j2)]
            # 参加者2の項
            + p2[(0, j1, j2)] * points2[j2][0]  # 財aの価値
            + p2[(1, j1, j2)] * points2[j2][1]  # 財bの価値
            + p2[(2, j1, j2)] * points2[j2][2]  # シナジーαの価値
            - u2[(j1, j2)]
        )
        for j1 in range(J1)
        for j2 in range(J2)
    )
    prob += objective

    # ========== 制約 ==========
    
    # 1. 非負性: u1[j1,j2] >= 0, u2[j1,j2] >= 0 (全てのj1, j2で)
    # 変数の定義で既に lowBound=0.0 として設定済み
    
    # 2. 1-Lipschitz制約: 0 ≤ p1[l,j1,j2] ≤ 1, 0 ≤ p2[l,j1,j2] ≤ 1
    # 変数の定義で既に lowBound=0.0, upBound=1.0 として設定済み
    
    # 3. IC制約（参加者1）: u1(i1, j2) >= u1(k1, j2) + p1(k1, j2)・(x1(i1) - x1(k1)) for all i1, k1, j2
    # Cursor.md: u_1(i) \ge u_1(j) + \mathbf{p}_1(j)^{\top}(\mathbf{x}_1(i) - \mathbf{x}_1(j)) \quad \forall i, j
    for i1 in range(J1):
        x1_i = points1[i1]
        for k1 in range(J1):
            x1_k = points1[k1]
            for j2 in range(J2):
                prob += u1[(i1, j2)] >= u1[(k1, j2)] + (
                    p1[(0, k1, j2)] * (x1_i[0] - x1_k[0])  # 財aの項
                    + p1[(1, k1, j2)] * (x1_i[1] - x1_k[1])  # 財bの項
                    + p1[(2, k1, j2)] * (x1_i[2] - x1_k[2])  # シナジーαの項
                ), f"ic1_{i1}_{k1}_{j2}"
    
    # 4. IC制約（参加者2）: u2(j1, i2) >= u2(j1, k2) + p2(j1, k2)・(x2(i2) - x2(k2)) for all i2, k2, j1
    # Cursor.md: u_2(i) \ge u_2(j) + \mathbf{p}_2(j)^{\top}(\mathbf{x}_2(i) - \mathbf{x}_2(j)) \quad \forall i, j
    for i2 in range(J2):
        x2_i = points2[i2]
        for k2 in range(J2):
            x2_k = points2[k2]
            for j1 in range(J1):
                prob += u2[(j1, i2)] >= u2[(j1, k2)] + (
                    p2[(0, j1, k2)] * (x2_i[0] - x2_k[0])  # 財aの項
                    + p2[(1, j1, k2)] * (x2_i[1] - x2_k[1])  # 財bの項
                    + p2[(2, j1, k2)] * (x2_i[2] - x2_k[2])  # シナジーαの項
                ), f"ic2_{j1}_{i2}_{k2}"
    
    # 5. シナジーの配分制約（参加者1）
    # Cursor.md:
    #   u₁_α(x) ≤ u₁_a(x)
    #   u₁_α(x) ≤ u₁_b(x)
    #   u₁_α(x) ≥ u₁_a(x) + u₁_b(x) - 1
    for j1 in range(J1):
        for j2 in range(J2):
            # 上界制約: p1[2,j1,j2] <= p1[0,j1,j2], p1[2,j1,j2] <= p1[1,j1,j2]
            prob += p1[(2, j1, j2)] <= p1[(0, j1, j2)], f"synergy1_item0_{j1}_{j2}_upper"
            prob += p1[(2, j1, j2)] <= p1[(1, j1, j2)], f"synergy1_item1_{j1}_{j2}_upper"
            
            # 下界制約（inclusion-exclusion）: p1[2,j1,j2] >= p1[0,j1,j2] + p1[1,j1,j2] - 1
            prob += p1[(2, j1, j2)] >= p1[(0, j1, j2)] + p1[(1, j1, j2)] - 1, f"synergy1_{j1}_{j2}_lower"
    
    # 6. シナジーの配分制約（参加者2）
    for j1 in range(J1):
        for j2 in range(J2):
            # 上界制約: p2[2,j1,j2] <= p2[0,j1,j2], p2[2,j1,j2] <= p2[1,j1,j2]
            prob += p2[(2, j1, j2)] <= p2[(0, j1, j2)], f"synergy2_item0_{j1}_{j2}_upper"
            prob += p2[(2, j1, j2)] <= p2[(1, j1, j2)], f"synergy2_item1_{j1}_{j2}_upper"
            
            # 下界制約（inclusion-exclusion）: p2[2,j1,j2] >= p2[0,j1,j2] + p2[1,j1,j2] - 1
            prob += p2[(2, j1, j2)] >= p2[(0, j1, j2)] + p2[(1, j1, j2)] - 1, f"synergy2_{j1}_{j2}_lower"
    
    # 7. Implementability制約（凸結合条件）
    # Cursor.md参照: u_{1,\{a\}} = p1[0], u_{1,\{b\}} = p1[1], u_{1,\{\alpha\}} = p1[2]
    # u_{2,\{a\}} = p2[0], u_{2,\{b\}} = p2[1], u_{2,\{\alpha\}} = p2[2]
    # 画像の制約に基づく実装
    for j1 in range(J1):
        for j2 in range(J2):
            # 記号の対応:
            # u_1,{a} = p1[0, j1, j2]
            # u_1,{b} = p1[1, j1, j2]
            # u_1,{α} = p1[2, j1, j2]
            # u_2,{a} = p2[0, j1, j2]
            # u_2,{b} = p2[1, j1, j2]
            # u_2,{α} = p2[2, j1, j2]
            
            # 制約1: 0 <= u_1,{α} <= 1, 0 <= u_2,{α} <= 1
            # (既に変数の定義で 0 <= p1[2] <= 1, 0 <= p2[2] <= 1 として設定済み)
            
            # 制約2: max{0, u_1,{α} - u_1,{a}, u_2,{α} - u_2,{b}} <= min{u_1,{a} - u_1,{α}, u_2,{b} - u_2,{α}, 1}
            # これは以下の制約に分解:
            # - u_1,{α} - u_1,{a} <= u_1,{a} - u_1,{α}  (これは常に成り立つ: p1[2] - p1[0] <= p1[0] - p1[2] は 2*p1[2] <= 2*p1[0] と等価)
            # - u_1,{α} - u_1,{a} <= u_2,{b} - u_2,{α}
            # - u_1,{α} - u_1,{a} <= 1
            # - u_2,{α} - u_2,{b} <= u_1,{a} - u_1,{α}
            # - u_2,{α} - u_2,{b} <= u_2,{b} - u_2,{α}  (これは常に成り立つ)
            # - u_2,{α} - u_2,{b} <= 1
            # - 0 <= u_1,{a} - u_1,{α}  (これは p1[0] >= p1[2] で、シナジー制約から既に保証されている)
            # - 0 <= u_2,{b} - u_2,{α}  (これは p2[1] >= p2[2] で、シナジー制約から既に保証されている)
            # - 0 <= 1  (常に成り立つ)
            
            # 実質的に必要な制約:
            prob += p1[(2, j1, j2)] - p1[(0, j1, j2)] <= p2[(1, j1, j2)] - p2[(2, j1, j2)], f"impl_max1_{j1}_{j2}"
            prob += p1[(2, j1, j2)] - p1[(0, j1, j2)] <= 1.0, f"impl_max2_{j1}_{j2}"
            prob += p2[(2, j1, j2)] - p2[(1, j1, j2)] <= p1[(0, j1, j2)] - p1[(2, j1, j2)], f"impl_max3_{j1}_{j2}"
            prob += p2[(2, j1, j2)] - p2[(1, j1, j2)] <= 1.0, f"impl_max4_{j1}_{j2}"
            
            # 制約3: max{0, u_1,{a,b,α} - u_1,{b}, u_2,{a,b,α} - u_2,{a}} <= min{u_1,{b} - u_1,{a}, u_2,{a} - u_2,{a}, 1}
            # 注: u_1,{a,b,α} は参加者1が財a, b, αをすべて獲得する確率
            # シナジーの制約から、これは min{p1[0], p1[1], p1[2]} で近似される
            # より正確には、p1[2] (シナジーαの配分確率) が p1[0] と p1[1] の両方以下であることから
            # u_1,{a,b,α} ≈ p1[2] と近似できる
            # 同様に、u_2,{a,b,α} ≈ p2[2] と近似できる
            
            # min{u_1,{b} - u_1,{a}, u_2,{a} - u_2,{a}, 1} = min{p1[1] - p1[0], 0, 1} = min{p1[1] - p1[0], 0}
            # p1[1] - p1[0] が負の場合は 0、正の場合は p1[1] - p1[0] と 1 の最小値
            
            # 実質的に必要な制約:
            # - u_1,{a,b,α} - u_1,{b} <= u_1,{b} - u_1,{a}  (p1[2] - p1[1] <= p1[1] - p1[0])
            # - u_1,{a,b,α} - u_1,{b} <= 0  (p1[2] - p1[1] <= 0, これはシナジー制約から既に保証されている)
            # - u_1,{a,b,α} - u_1,{b} <= 1  (p1[2] - p1[1] <= 1, これは常に成り立つ)
            # - u_2,{a,b,α} - u_2,{a} <= u_1,{b} - u_1,{a}  (p2[2] - p2[0] <= p1[1] - p1[0])
            # - u_2,{a,b,α} - u_2,{a} <= 0  (p2[2] - p2[0] <= 0, これはシナジー制約から既に保証されている)
            # - u_2,{a,b,α} - u_2,{a} <= 1  (p2[2] - p2[0] <= 1, これは常に成り立つ)
            
            # 実質的に必要な制約（既に保証されているものを除く）:
            prob += p1[(2, j1, j2)] - p1[(1, j1, j2)] <= p1[(1, j1, j2)] - p1[(0, j1, j2)], f"impl_max2_1_{j1}_{j2}"
            prob += p2[(2, j1, j2)] - p2[(0, j1, j2)] <= p1[(1, j1, j2)] - p1[(0, j1, j2)], f"impl_max2_2_{j1}_{j2}"
            
            # 制約4: u_1,{a} + u_1,{b} + u_2,{a} + u_2,{b} - u_1,{α} - u_2,{α} <= 1
            prob += (p1[(0, j1, j2)] + p1[(1, j1, j2)] + p2[(0, j1, j2)] + p2[(1, j1, j2)]
                     - p1[(2, j1, j2)] - p2[(2, j1, j2)] <= 1.0, f"impl_sum_{j1}_{j2}")
            
            # 補助変数: a_7^min = max{0, u_1,{a} - u_1,{α} - 1, u_2,{b} - u_2,{α} - 1}
            a7_min = pulp.LpVariable(f"a7_min_{j1}_{j2}", lowBound=0.0, cat=pulp.LpContinuous)
            prob += a7_min >= 0.0, f"impl_a7_0_{j1}_{j2}"
            prob += a7_min >= p1[(0, j1, j2)] - p1[(2, j1, j2)] - 1.0, f"impl_a7_1_{j1}_{j2}"
            prob += a7_min >= p2[(1, j1, j2)] - p2[(2, j1, j2)] - 1.0, f"impl_a7_2_{j1}_{j2}"
            
            # 補助変数: a_8^min = max{0, u_1,{b} - u_1,{α} - 1, u_2,{a} - u_2,{α} - 1}
            a8_min = pulp.LpVariable(f"a8_min_{j1}_{j2}", lowBound=0.0, cat=pulp.LpContinuous)
            prob += a8_min >= 0.0, f"impl_a8_0_{j1}_{j2}"
            prob += a8_min >= p1[(1, j1, j2)] - p1[(2, j1, j2)] - 1.0, f"impl_a8_1_{j1}_{j2}"
            prob += a8_min >= p2[(0, j1, j2)] - p2[(2, j1, j2)] - 1.0, f"impl_a8_2_{j1}_{j2}"
            
            # 追加制約: u_1,{a} + u_1,{b} + u_2,{a} + u_2,{b} - u_1,{α} - u_2,{α} <= 1 + a_7^min + a_8^min
            prob += (p1[(0, j1, j2)] + p1[(1, j1, j2)] + p2[(0, j1, j2)] + p2[(1, j1, j2)]
                     - p1[(2, j1, j2)] - p2[(2, j1, j2)] <= 1.0 + a7_min + a8_min, f"impl_sum2_{j1}_{j2}")

    # ========== 解く ==========
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)

    # ========== 結果 ==========
    # NumPy配列に直接変換
    u1_sol = np.array([[u1[(j1, j2)].varValue for j2 in range(J2)] for j1 in range(J1)], dtype=np.float64)
    u2_sol = np.array([[u2[(j1, j2)].varValue for j2 in range(J2)] for j1 in range(J1)], dtype=np.float64)
    p1_sol = np.array([[[p1[(l, j1, j2)].varValue for j2 in range(J2)] for j1 in range(J1)] for l in range(3)], dtype=np.float64)
    p2_sol = np.array([[[p2[(l, j1, j2)].varValue for j2 in range(J2)] for j1 in range(J1)] for l in range(3)], dtype=np.float64)

    return status, obj_val, u1_sol, u2_sol, p1_sol, p2_sol


def solve_mechanism_2agents_iterative(points1, weights1, grid_sizes1, points2, weights2, grid_sizes2, solver=None, max_iter=100, tol=1e-6):
    """
    反復的制約追加アルゴリズムによるRochet-Choné型LPの求解（2人2財1シナジー版）
    
    アルゴリズム:
    1. 局所的なIC制約（ic-local）のみで初期問題を定義
    2. 最適解を求める
    3. 違反している大局的なIC制約を検出（各参加者について独立に）
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
    assert J1 == len(weights1)
    assert J2 == len(weights2)
    assert len(points1[0]) == 3, "points1は3次元である必要があります"
    assert len(points2[0]) == 3, "points2は3次元である必要があります"
    assert len(grid_sizes1) == 3, "grid_sizes1は3次元である必要があります"
    assert len(grid_sizes2) == 3, "grid_sizes2は3次元である必要があります"
    
    nx1, ny1, nz1 = grid_sizes1
    nx2, ny2, nz2 = grid_sizes2
    assert nx1 * ny1 * nz1 == J1, f"grid_sizes1の積({nx1 * ny1 * nz1})がpoints1の数({J1})と一致しません"
    assert nx2 * ny2 * nz2 == J2, f"grid_sizes2の積({nx2 * ny2 * nz2})がpoints2の数({J2})と一致しません"
    
    # pointsをNumPy配列に変換（一度だけ、高速化のため）
    points1_arr = np.array(points1, dtype=np.float64)  # (J1, 3)
    points2_arr = np.array(points2, dtype=np.float64)  # (J2, 3)
    
    # グリッドインデックスを計算する関数（参加者1）
    def get_grid_indices1(point_idx):
        """参加者1の点のインデックスからグリッド座標を取得"""
        k = point_idx % nz1
        j = (point_idx // nz1) % ny1
        i = point_idx // (ny1 * nz1)
        return (i, j, k)
    
    def get_point_idx1(grid_indices):
        """参加者1のグリッド座標から点のインデックスを取得"""
        i, j, k = grid_indices
        return i * (ny1 * nz1) + j * nz1 + k
    
    def get_neighbors1(point_idx):
        """参加者1の局所的なIC制約に使用する隣接点のインデックスを取得"""
        i, j, k = get_grid_indices1(point_idx)
        neighbors = []
        for di in [-1, 1]:
            if 0 <= i + di < nx1:
                neighbor_idx = get_point_idx1((i + di, j, k))
                neighbors.append(neighbor_idx)
        for dj in [-1, 1]:
            if 0 <= j + dj < ny1:
                neighbor_idx = get_point_idx1((i, j + dj, k))
                neighbors.append(neighbor_idx)
        for dk in [-1, 1]:
            if 0 <= k + dk < nz1:
                neighbor_idx = get_point_idx1((i, j, k + dk))
                neighbors.append(neighbor_idx)
        return neighbors
    
    # グリッドインデックスを計算する関数（参加者2）
    def get_grid_indices2(point_idx):
        """参加者2の点のインデックスからグリッド座標を取得"""
        k = point_idx % nz2
        j = (point_idx // nz2) % ny2
        i = point_idx // (ny2 * nz2)
        return (i, j, k)
    
    def get_point_idx2(grid_indices):
        """参加者2のグリッド座標から点のインデックスを取得"""
        i, j, k = grid_indices
        return i * (ny2 * nz2) + j * nz2 + k
    
    def get_neighbors2(point_idx):
        """参加者2の局所的なIC制約に使用する隣接点のインデックスを取得"""
        i, j, k = get_grid_indices2(point_idx)
        neighbors = []
        for di in [-1, 1]:
            if 0 <= i + di < nx2:
                neighbor_idx = get_point_idx2((i + di, j, k))
                neighbors.append(neighbor_idx)
        for dj in [-1, 1]:
            if 0 <= j + dj < ny2:
                neighbor_idx = get_point_idx2((i, j + dj, k))
                neighbors.append(neighbor_idx)
        for dk in [-1, 1]:
            if 0 <= k + dk < nz2:
                neighbor_idx = get_point_idx2((i, j, k + dk))
                neighbors.append(neighbor_idx)
        return neighbors
    
    # ========== 問題の初期化 ==========
    prob = pulp.LpProblem("RC_iterative_2agents_2goods_1synergy", pulp.LpMaximize)
    
    # 変数の定義
    u1 = {
        (j1, j2): pulp.LpVariable(f"u1_{j1}_{j2}", lowBound=0.0, cat=pulp.LpContinuous)
        for j1 in range(J1)
        for j2 in range(J2)
    }
    u2 = {
        (j1, j2): pulp.LpVariable(f"u2_{j1}_{j2}", lowBound=0.0, cat=pulp.LpContinuous)
        for j1 in range(J1)
        for j2 in range(J2)
    }
    p1 = {
        (l, j1, j2): pulp.LpVariable(f"p1_{l}_{j1}_{j2}", lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
        for l in range(3)
        for j1 in range(J1)
        for j2 in range(J2)
    }
    p2 = {
        (l, j1, j2): pulp.LpVariable(f"p2_{l}_{j1}_{j2}", lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
        for l in range(3)
        for j1 in range(J1)
        for j2 in range(J2)
    }
    
    # 目的関数
    objective = pulp.lpSum(
        weights1[j1] * weights2[j2] * (
            p1[(0, j1, j2)] * points1[j1][0]
            + p1[(1, j1, j2)] * points1[j1][1]
            + p1[(2, j1, j2)] * points1[j1][2]
            - u1[(j1, j2)]
            + p2[(0, j1, j2)] * points2[j2][0]
            + p2[(1, j1, j2)] * points2[j2][1]
            + p2[(2, j1, j2)] * points2[j2][2]
            - u2[(j1, j2)]
        )
        for j1 in range(J1)
        for j2 in range(J2)
    )
    prob += objective
    
    # シナジーの配分制約
    for j1 in range(J1):
        for j2 in range(J2):
            # 参加者1
            prob += p1[(2, j1, j2)] <= p1[(0, j1, j2)], f"synergy1_item0_{j1}_{j2}_upper"
            prob += p1[(2, j1, j2)] <= p1[(1, j1, j2)], f"synergy1_item1_{j1}_{j2}_upper"
            prob += p1[(2, j1, j2)] >= p1[(0, j1, j2)] + p1[(1, j1, j2)] - 1, f"synergy1_{j1}_{j2}_lower"
            # 参加者2
            prob += p2[(2, j1, j2)] <= p2[(0, j1, j2)], f"synergy2_item0_{j1}_{j2}_upper"
            prob += p2[(2, j1, j2)] <= p2[(1, j1, j2)], f"synergy2_item1_{j1}_{j2}_upper"
            prob += p2[(2, j1, j2)] >= p2[(0, j1, j2)] + p2[(1, j1, j2)] - 1, f"synergy2_{j1}_{j2}_lower"
    
    # Implementability制約（凸結合条件）- 反復版にも追加
    for j1 in range(J1):
        for j2 in range(J2):
            # 制約2: max{0, u_1,{α} - u_1,{a}, u_2,{α} - u_2,{b}} <= min{u_1,{a} - u_1,{α}, u_2,{b} - u_2,{α}, 1}
            prob += p1[(2, j1, j2)] - p1[(0, j1, j2)] <= p2[(1, j1, j2)] - p2[(2, j1, j2)], f"impl_max1_{j1}_{j2}_iter"
            prob += p1[(2, j1, j2)] - p1[(0, j1, j2)] <= 1.0, f"impl_max2_{j1}_{j2}_iter"
            prob += p2[(2, j1, j2)] - p2[(1, j1, j2)] <= p1[(0, j1, j2)] - p1[(2, j1, j2)], f"impl_max3_{j1}_{j2}_iter"
            prob += p2[(2, j1, j2)] - p2[(1, j1, j2)] <= 1.0, f"impl_max4_{j1}_{j2}_iter"
            
            # 制約3: max{0, u_1,{a,b,α} - u_1,{b}, u_2,{a,b,α} - u_2,{a}} <= min{u_1,{b} - u_1,{a}, u_2,{a} - u_2,{a}, 1}
            prob += p1[(2, j1, j2)] - p1[(1, j1, j2)] <= p1[(1, j1, j2)] - p1[(0, j1, j2)], f"impl_max2_1_{j1}_{j2}_iter"
            prob += p2[(2, j1, j2)] - p2[(0, j1, j2)] <= p1[(1, j1, j2)] - p1[(0, j1, j2)], f"impl_max2_2_{j1}_{j2}_iter"
            
            # 制約4: u_1,{a} + u_1,{b} + u_2,{a} + u_2,{b} - u_1,{α} - u_2,{α} <= 1
            prob += (p1[(0, j1, j2)] + p1[(1, j1, j2)] + p2[(0, j1, j2)] + p2[(1, j1, j2)]
                     - p1[(2, j1, j2)] - p2[(2, j1, j2)] <= 1.0, f"impl_sum_{j1}_{j2}_iter")
            
            # 補助変数: a_7^min = max{0, u_1,{a} - u_1,{α} - 1, u_2,{b} - u_2,{α} - 1}
            a7_min = pulp.LpVariable(f"a7_min_{j1}_{j2}_iter", lowBound=0.0, cat=pulp.LpContinuous)
            prob += a7_min >= 0.0, f"impl_a7_0_{j1}_{j2}_iter"
            prob += a7_min >= p1[(0, j1, j2)] - p1[(2, j1, j2)] - 1.0, f"impl_a7_1_{j1}_{j2}_iter"
            prob += a7_min >= p2[(1, j1, j2)] - p2[(2, j1, j2)] - 1.0, f"impl_a7_2_{j1}_{j2}_iter"
            
            # 補助変数: a_8^min = max{0, u_1,{b} - u_1,{α} - 1, u_2,{a} - u_2,{α} - 1}
            a8_min = pulp.LpVariable(f"a8_min_{j1}_{j2}_iter", lowBound=0.0, cat=pulp.LpContinuous)
            prob += a8_min >= 0.0, f"impl_a8_0_{j1}_{j2}_iter"
            prob += a8_min >= p1[(1, j1, j2)] - p1[(2, j1, j2)] - 1.0, f"impl_a8_1_{j1}_{j2}_iter"
            prob += a8_min >= p2[(0, j1, j2)] - p2[(2, j1, j2)] - 1.0, f"impl_a8_2_{j1}_{j2}_iter"
            
            # 追加制約: u_1,{a} + u_1,{b} + u_2,{a} + u_2,{b} - u_1,{α} - u_2,{α} <= 1 + a_7^min + a_8^min
            prob += (p1[(0, j1, j2)] + p1[(1, j1, j2)] + p2[(0, j1, j2)] + p2[(1, j1, j2)]
                     - p1[(2, j1, j2)] - p2[(2, j1, j2)] <= 1.0 + a7_min + a8_min, f"impl_sum2_{j1}_{j2}_iter")
    
    # 局所的なIC制約（参加者1: 隣接点のみ）
    local_constraints1 = set()
    for i1 in range(J1):
        neighbors1 = get_neighbors1(i1)
        x1_i = points1_arr[i1]
        for k1 in neighbors1:
            x1_k = points1_arr[k1]
            for j2 in range(J2):
                prob += u1[(i1, j2)] >= u1[(k1, j2)] + (
                    p1[(0, k1, j2)] * (x1_i[0] - x1_k[0])
                    + p1[(1, k1, j2)] * (x1_i[1] - x1_k[1])
                    + p1[(2, k1, j2)] * (x1_i[2] - x1_k[2])
                ), f"ic1_local_{i1}_{k1}_{j2}"
                local_constraints1.add((i1, k1, j2))
    
    # 局所的なIC制約（参加者2: 隣接点のみ）
    local_constraints2 = set()
    for i2 in range(J2):
        neighbors2 = get_neighbors2(i2)
        x2_i = points2_arr[i2]
        for k2 in neighbors2:
            x2_k = points2_arr[k2]
            for j1 in range(J1):
                prob += u2[(j1, i2)] >= u2[(j1, k2)] + (
                    p2[(0, j1, k2)] * (x2_i[0] - x2_k[0])
                    + p2[(1, j1, k2)] * (x2_i[1] - x2_k[1])
                    + p2[(2, j1, k2)] * (x2_i[2] - x2_k[2])
                ), f"ic2_local_{j1}_{i2}_{k2}"
                local_constraints2.add((j1, i2, k2))
    
    # 追加された制約を記録（重複追加を防ぐ）
    added_constraints1 = local_constraints1.copy()
    added_constraints2 = local_constraints2.copy()
    
    # ========== 反復ループ ==========
    for iteration in range(max_iter):
        # 最適解を求める
        prob.solve(solver)
        
        if prob.status != pulp.LpStatusOptimal:
            status = pulp.LpStatus[prob.status]
            return status, None, None, None, None, None, iteration
        
        # 解を取得（NumPy配列に直接変換、高速化）
        u1_sol = np.array([[u1[(j1, j2)].varValue for j2 in range(J2)] for j1 in range(J1)], dtype=np.float64)  # (J1, J2)
        u2_sol = np.array([[u2[(j1, j2)].varValue for j2 in range(J2)] for j1 in range(J1)], dtype=np.float64)  # (J1, J2)
        p1_sol = np.array([[[p1[(l, j1, j2)].varValue for j2 in range(J2)] for j1 in range(J1)] for l in range(3)], dtype=np.float64)  # (3, J1, J2)
        p2_sol = np.array([[[p2[(l, j1, j2)].varValue for j2 in range(J2)] for j1 in range(J1)] for l in range(3)], dtype=np.float64)  # (3, J1, J2)
        
        # 違反している制約を検出（参加者1）
        violations1 = []
        constraint_mask1 = np.zeros((J1, J1, J2), dtype=bool)
        if added_constraints1:
            constraint_triples = np.array(list(added_constraints1), dtype=np.int32)
            if len(constraint_triples) > 0:
                constraint_mask1[constraint_triples[:, 0], constraint_triples[:, 1], constraint_triples[:, 2]] = True
        
        # 参加者1の違反チェック: u1(i1, j2) - u1(k1, j2) < p1(k1, j2)・(x1(i1) - x1(k1)) - tol
        for i1 in range(J1):
            x1_i = points1_arr[i1]
            for k1 in range(J1):
                x1_k = points1_arr[k1]
                x1_diff = x1_i - x1_k  # (3,)
                for j2 in range(J2):
                    if constraint_mask1[i1, k1, j2]:
                        continue
                    u1_diff = u1_sol[i1, j2] - u1_sol[k1, j2]
                    inner_product = (p1_sol[0, k1, j2] * x1_diff[0] +
                                     p1_sol[1, k1, j2] * x1_diff[1] +
                                     p1_sol[2, k1, j2] * x1_diff[2])
                    if u1_diff < inner_product - tol:
                        violations1.append((i1, k1, j2))
        
        # 違反している制約を検出（参加者2）
        violations2 = []
        constraint_mask2 = np.zeros((J1, J2, J2), dtype=bool)
        if added_constraints2:
            constraint_triples = np.array(list(added_constraints2), dtype=np.int32)
            if len(constraint_triples) > 0:
                constraint_mask2[constraint_triples[:, 0], constraint_triples[:, 1], constraint_triples[:, 2]] = True
        
        # 参加者2の違反チェック: u2(j1, i2) - u2(j1, k2) < p2(j1, k2)・(x2(i2) - x2(k2)) - tol
        for i2 in range(J2):
            x2_i = points2_arr[i2]
            for k2 in range(J2):
                x2_k = points2_arr[k2]
                x2_diff = x2_i - x2_k  # (3,)
                for j1 in range(J1):
                    if constraint_mask2[j1, i2, k2]:
                        continue
                    u2_diff = u2_sol[j1, i2] - u2_sol[j1, k2]
                    inner_product = (p2_sol[0, j1, k2] * x2_diff[0] +
                                     p2_sol[1, j1, k2] * x2_diff[1] +
                                     p2_sol[2, j1, k2] * x2_diff[2])
                    if u2_diff < inner_product - tol:
                        violations2.append((j1, i2, k2))
        
        # 違反がなければ終了
        if not violations1 and not violations2:
            status = pulp.LpStatus[prob.status]
            obj_val = pulp.value(prob.objective)
            return status, obj_val, u1_sol, u2_sol, p1_sol, p2_sol, iteration + 1
        
        # 違反している制約を追加
        for i1, k1, j2 in violations1:
            x1_i = points1_arr[i1]
            x1_k = points1_arr[k1]
            constraint = u1[(i1, j2)] >= u1[(k1, j2)] + (
                p1[(0, k1, j2)] * (x1_i[0] - x1_k[0])
                + p1[(1, k1, j2)] * (x1_i[1] - x1_k[1])
                + p1[(2, k1, j2)] * (x1_i[2] - x1_k[2])
            )
            prob += constraint, f"ic1_{i1}_{k1}_{j2}_iter{iteration}"
            added_constraints1.add((i1, k1, j2))
        
        for j1, i2, k2 in violations2:
            x2_i = points2_arr[i2]
            x2_k = points2_arr[k2]
            constraint = u2[(j1, i2)] >= u2[(j1, k2)] + (
                p2[(0, j1, k2)] * (x2_i[0] - x2_k[0])
                + p2[(1, j1, k2)] * (x2_i[1] - x2_k[1])
                + p2[(2, j1, k2)] * (x2_i[2] - x2_k[2])
            )
            prob += constraint, f"ic2_{j1}_{i2}_{k2}_iter{iteration}"
            added_constraints2.add((j1, i2, k2))
        
        print(f"Iteration {iteration + 1}: {len(violations1)} violations (agent1), {len(violations2)} violations (agent2), added {len(violations1) + len(violations2)} constraints")
    
    # 最大反復回数に達した場合
    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)
    return status, obj_val, u1_sol, u2_sol, p1_sol, p2_sol, max_iter


def save_results_2agents(
    points1, weights1, points2, weights2,
    u1_sol, u2_sol, p1_sol, p2_sol,
    obj_val, status,
    grid_sizes1=None, grid_sizes2=None, n_iter=None,
    filename=None, data_dir="data"
):
    """
    2人2財1シナジーの結果をNumPy形式で保存する。
    
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
        filename = f"results_2agents_{timestamp}.npz"
    
    filepath = os.path.join(data_dir, filename)
    
    # NumPy配列に変換
    points1_arr = np.array(points1, dtype=np.float64)
    weights1_arr = np.array(weights1, dtype=np.float64)
    points2_arr = np.array(points2, dtype=np.float64)
    weights2_arr = np.array(weights2, dtype=np.float64)
    u1_sol_arr = np.array(u1_sol, dtype=np.float64)
    u2_sol_arr = np.array(u2_sol, dtype=np.float64)
    p1_sol_arr = np.array(p1_sol, dtype=np.float64)
    p2_sol_arr = np.array(p2_sol, dtype=np.float64)
    
    # NumPy形式で保存（メタデータも含める）
    save_dict = {
        'points1': points1_arr,
        'weights1': weights1_arr,
        'points2': points2_arr,
        'weights2': weights2_arr,
        'u1_sol': u1_sol_arr,
        'u2_sol': u2_sol_arr,
        'p1_sol': p1_sol_arr,
        'p2_sol': p2_sol_arr,
        'obj_val': np.array([obj_val], dtype=np.float64) if obj_val is not None else np.array([np.nan], dtype=np.float64),
        'J1': np.array([len(points1)], dtype=np.int32),
        'J2': np.array([len(points2)], dtype=np.int32),
    }
    
    if grid_sizes1 is not None:
        save_dict['grid_sizes1'] = np.array(grid_sizes1, dtype=np.int32)
    if grid_sizes2 is not None:
        save_dict['grid_sizes2'] = np.array(grid_sizes2, dtype=np.int32)
    if n_iter is not None:
        save_dict['n_iter'] = np.array([n_iter], dtype=np.int32)
    
    # メタデータファイルも保存（文字列情報用）
    metadata_filepath = filepath.replace('.npz', '_metadata.txt')
    with open(metadata_filepath, 'w') as f:
        f.write(f"status: {status}\n")
        f.write(f"obj_val: {obj_val}\n")
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


def load_results_2agents(filepath):
    """
    保存された結果を読み込む。
    
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

