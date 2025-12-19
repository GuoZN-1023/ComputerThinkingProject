# src/offline_model.py
import numpy as np


def nearest_assignment(D, Dist, y):
    """
    给定选址向量 y[j]∈{0,1}，简单按“最近启用站点”分配需求。
    输出 x[i,j,t] 为 0/1，表示需求单元 i 在 t 时刻是否把全部需求交给站点 j。
    """
    I, T = D.shape
    J = Dist.shape[1]
    x = np.zeros((I, J, T), dtype=float)

    active_js = np.where(y > 0.5)[0]
    if active_js.size == 0:
        return x

    for i in range(I):
        dists_i = Dist[i, active_js]
        j_best = active_js[np.argmin(dists_i)]
        for t in range(T):
            if D[i, t] > 0:
                x[i, j_best, t] = 1.0

    return x


def evaluate_solution(D, Dist, cap, y,
                      alpha_walk=1.0,
                      lam_over=1.0,
                      mu_imb=1.0,
                      eta_build=1.0):
    """
    计算给定选址 y 的总损失和各项子指标。

    total = alpha_walk * 步行成本
            + lam_over * overflow 总量
            + mu_imb * Imbalance (KL(pi || pi*))
            + eta_build * 建设成本
    """
    I, T = D.shape
    J = Dist.shape[1]

    y = np.asarray(y, dtype=float)
    assert y.shape == (J,)

    # 分配决策 x[i,j,t]
    x = nearest_assignment(D, Dist, y)

    # 各站各时刻累积负载 L[j,t]
    L = np.einsum("it,ijt->jt", D, x)

    # 步行成本
    walk_cost = 0.0
    for i in range(I):
        for j in range(J):
            if not np.any(x[i, j, :] > 0):
                continue
            for t in range(T):
                if x[i, j, t] > 0:
                    walk_cost += D[i, t] * Dist[i, j] * x[i, j, t]
    walk_cost *= alpha_walk

    # overflow 成本
    cap_mat = cap[:, None] * y[:, None]
    overflow = np.maximum(0.0, L - cap_mat)
    overflow_cost = lam_over * overflow.sum()

    # Imbalance 成本
    eps = 1e-9
    imbalance_cost = 0.0
    for t in range(T):
        total_L_t = L[:, t].sum()
        if total_L_t < eps:
            continue
        pi = L[:, t] / (total_L_t + eps)

        cap_active = cap * y
        total_cap_active = cap_active.sum()
        if total_cap_active < eps:
            continue
        pi_star = cap_active / (total_cap_active + eps)

        imbalance_cost += np.sum(pi * np.log((pi + eps) / (pi_star + eps)))
    imbalance_cost *= mu_imb

    # 建设成本
    build_cost = eta_build * y.sum()

    total = walk_cost + overflow_cost + imbalance_cost + build_cost

    return dict(
        total=float(total),
        walk_cost=float(walk_cost),
        overflow_cost=float(overflow_cost),
        imbalance_cost=float(imbalance_cost),
        build_cost=float(build_cost),
        L=L,
        overflow=overflow,
    )


def random_search(D, Dist, cap, budget_min=2, budget_max=None,
                  n_samples=200,
                  alpha_walk=1.0, lam_over=1.0, mu_imb=1.0, eta_build=1.0,
                  rng_seed=0):
    """
    一个非常简单的随机搜索，用于示范：
    随机生成多组选址方案 y，在给定预算范围内选出 total 最小的方案。
    """
    rng = np.random.default_rng(rng_seed)
    J = Dist.shape[1]
    if budget_max is None:
        budget_max = J

    best = None
    history = []

    for _ in range(n_samples):
        budget = rng.integers(budget_min, budget_max + 1)
        # 从 J 个站里随机选 budget 个启用
        y = np.zeros(J, dtype=float)
        active_js = rng.choice(J, size=budget, replace=False)
        y[active_js] = 1.0

        metrics = evaluate_solution(
            D, Dist, cap, y,
            alpha_walk=alpha_walk,
            lam_over=lam_over,
            mu_imb=mu_imb,
            eta_build=eta_build,
        )
        history.append(dict(y=y.copy(), **metrics))

        if (best is None) or (metrics["total"] < best["total"]):
            best = dict(y=y.copy(), **metrics)

    return best, history
