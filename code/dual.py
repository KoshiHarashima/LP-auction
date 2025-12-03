"""
Dual問題: Kash-Frongillo最適輸送LP
最適輸送問題の双対問題を解く
"""

import pulp


def cost_snew(x_i, x_j):
    """
    c_ij = max{0, d_a, d_b, d_a + d_b + d_c} を計算。
    x_i, x_j は長さ3のベクトル (a,b,c)。
    """
    d_a = x_i[0] - x_j[0]
    d_b = x_i[1] - x_j[1]
    d_c = x_i[2] - x_j[2]
    return max(0.0, d_a, d_b, d_a + d_b + d_c)


def build_cost_matrix(points_plus, points_minus):
    """
    2つの点列の間のコスト c_ij を行列として返す。
    """
    I = len(points_plus)
    J = len(points_minus)
    C = [[0.0 for _ in range(J)] for _ in range(I)]
    for i in range(I):
        for j in range(J):
            C[i][j] = cost_snew(points_plus[i], points_minus[j])
    return C


def discretize_signed_measure(points, cell_vol, signed_density_fn):
    """
    任意の signed density f(x) から、離散 μ^+, μ^- を作る。

    μ_j = f(x_j) * cell_vol
    μ^+_j = max(μ_j, 0)
    μ^-_j = max(-μ_j, 0)

    返り値:
        mu_plus, mu_minus  (どちらも len(points) のリスト)
    """
    mu = [signed_density_fn(x) * cell_vol for x in points]
    mu_plus = [max(m, 0.0) for m in mu]
    mu_minus = [max(-m, 0.0) for m in mu]

    # μ(X)=0 のはずなので、数値誤差があれば軽く正規化しておく
    sum_plus = sum(mu_plus)
    sum_minus = sum(mu_minus)
    if sum_plus > 0 and sum_minus > 0:
        # 平均に合わせて rescale
        scale_plus = (sum_plus + sum_minus) / (2 * sum_plus)
        scale_minus = (sum_plus + sum_minus) / (2 * sum_minus)
        mu_plus = [m * scale_plus for m in mu_plus]
        mu_minus = [m * scale_minus for m in mu_minus]

    return mu_plus, mu_minus


def solve_dual(points_plus, mu_plus, points_minus, mu_minus, solver=None):
    """
    Dual (transport) LP:

        min  sum_{i,j} c_ij * gamma_ij
        s.t. sum_j gamma_ij = mu_plus[i]
             sum_i gamma_ij = mu_minus[j]
             gamma_ij >= 0

    ここで c_ij は S_new の support function によるコスト。

    戻り値:
        status, objective_value, gamma (2D list)
    """
    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=True)
    
    I = len(points_plus)
    J = len(points_minus)
    assert I == len(mu_plus)
    assert J == len(mu_minus)

    # cost matrix
    C = build_cost_matrix(points_plus, points_minus)

    # 問題セットアップ
    prob = pulp.LpProblem("Dual_OT_Snew", pulp.LpMinimize)

    # 変数 gamma_ij >= 0
    gamma_var = {
        (i, j): pulp.LpVariable(f"gamma_{i}_{j}", lowBound=0.0)
        for i in range(I) for j in range(J)
    }

    # 目的関数
    prob += pulp.lpSum(C[i][j] * gamma_var[(i, j)] for i in range(I) for j in range(J))

    # 行側マージナル: sum_j gamma_ij = mu_plus[i]
    for i in range(I):
        prob += pulp.lpSum(gamma_var[(i, j)] for j in range(J)) == mu_plus[i], f"row_{i}"

    # 列側マージナル: sum_i gamma_ij = mu_minus[j]
    for j in range(J):
        prob += pulp.lpSum(gamma_var[(i, j)] for i in range(I)) == mu_minus[j], f"col_{j}"

    # 解く
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)

    gamma_sol = [[gamma_var[(i, j)].varValue for j in range(J)] for i in range(I)]

    return status, obj_val, gamma_sol

