# src/offline_model.py
import numpy as np


def nearest_assignment(D, Dist, y):
    """
    给定选址向量 y[j]∈{0,1}，按“最近启用站点”分配需求。

    输出:
      x: ndarray [I, J, T]，0/1，表示需求单元 i 在 t 时刻把全部需求交给站点 j。
    """
    I, T = D.shape
    J = Dist.shape[1]
    x = np.zeros((I, J, T), dtype=float)

    y = np.asarray(y, dtype=float)
    active_js = np.where(y > 0.5)[0]
    if active_js.size == 0:
        return x

    for i in range(I):
        # 如果这个 cell 整天都没需求，直接跳过
        if np.all(D[i, :] <= 0):
            continue
        j_best = active_js[np.argmin(Dist[i, active_js])]
        mask = D[i, :] > 0
        x[i, j_best, mask] = 1.0
    return x


def _quantile_positive(a, q=0.95, fallback=1.0):
    """
    取数组 a 的正值分位数（常用于距离尺度），避免 0 造成分母为 0。
    """
    a = np.asarray(a, dtype=float).ravel()
    pos = a[a > 0]
    if pos.size == 0:
        return float(fallback)
    return float(np.quantile(pos, q))


def _build_norm(k, J, mode="minmax", power=2.0, kmin=None, kmax=None):
    """
    站点数惩罚的无量纲化（并支持“下凸/凸增”形状）。

    mode:
      - "count": 直接用 k（不归一化）；可用 power>1 使其更凸。
      - "ratio": 用 (k/J)^power，落在 [0,1]。
      - "minmax": 用 ((k-kmin)/(kmax-kmin))^power，预算区间内落在 [0,1]。

    返回:
      build_norm: float
    """
    eps = 1e-12
    k = float(k)
    J = int(J)

    p = float(power)
    if p <= 0:
        raise ValueError("station_penalty_power must be > 0")

    mode = str(mode).lower().strip()
    if mode == "count":
        base = k
        return float(base ** p) if abs(p - 1.0) > 1e-12 else float(base)

    if mode == "ratio":
        denom = max(1, J)
        z = k / denom
        z = float(np.clip(z, 0.0, 1.0))
        return float(z ** p)

    if mode == "minmax":
        if kmin is None:
            kmin = 0
        if kmax is None:
            kmax = J
        denom = max(eps, float(kmax) - float(kmin))
        z = (k - float(kmin)) / denom
        z = float(np.clip(z, 0.0, 1.0))
        return float(z ** p)

    raise ValueError(f"Unknown station_penalty_mode: {mode}")


def evaluate_solution(
    D,
    Dist,
    cap,
    y,
    *,
    alpha_walk=1.0,
    lam_over=1.0,
    mu_imb=1.0,
    eta_build=1.0,
    normalize_terms=True,
    # 站点惩罚（“下凸/凸增” + 归一化）
    station_penalty_mode="minmax",
    station_penalty_power=2.0,
    budget_min=None,
    budget_max=None,
    # 距离归一化尺度
    dist_scale=None,
    # imbalance 归一化方式：只做 per-time 平均（"per_time"）或再除 log(J)（"logJ"）
    imbalance_scale="logJ",
):
    """
    计算给定选址 y 的总损失和各项子指标。

    你们作业里遇到的核心问题是：walk_cost 会随距离单位缩放而膨胀（km→m 就 ×1000），
    导致其他项（overflow / imbalance / build）被量级掩盖。

    因此默认 normalize_terms=True：把每一项做成“无量纲 / O(1)”后再加权求和。

    返回 dict 中包含：
      - *_raw: 原始量纲（受单位影响）
      - *_norm: 无量纲化后的量
      - *_cost: 进入 total 的加权项（与你设置的 alpha/lam/mu/eta 对应）
    """
    I, T = D.shape
    J = Dist.shape[1]
    y = np.asarray(y, dtype=float)
    assert y.shape == (J,)

    # 分配决策 x[i,j,t]
    x = nearest_assignment(D, Dist, y)

    # 各站各时刻累积负载 L[j,t]
    L = np.einsum("it,ijt->jt", D, x)

    # ---------- raw 指标（受单位影响） ----------
    # 步行成本 raw：sum_{i,j,t} D[i,t] * Dist[i,j] * x[i,j,t]
    walk_raw = float(np.sum(D[:, None, :] * Dist[:, :, None] * x))

    # overflow raw：超出容量的总量
    cap_mat = cap[:, None] * y[:, None]
    overflow = np.maximum(0.0, L - cap_mat)
    overflow_raw = float(overflow.sum())

    # imbalance raw：按时间片求 KL(pi || pi*)
    eps = 1e-9
    imbalance_raw = 0.0
    for t in range(T):
        total_L_t = float(L[:, t].sum())
        if total_L_t < eps:
            continue
        pi = L[:, t] / (total_L_t + eps)
        cap_active = cap * y
        total_cap_active = float(cap_active.sum())
        if total_cap_active < eps:
            continue
        pi_star = cap_active / (total_cap_active + eps)
        imbalance_raw += float(np.sum(pi * np.log((pi + eps) / (pi_star + eps))))

    build_raw = float(y.sum())

    # ---------- 无量纲化（默认开启） ----------
    if normalize_terms:
        sumD = float(D.sum())
        if dist_scale is None:
            dist_scale = _quantile_positive(Dist, q=0.95, fallback=1.0)
        dist_scale = float(dist_scale)

        walk_norm = walk_raw / (sumD * dist_scale + eps)
        overflow_norm = overflow_raw / (sumD + eps)

        # imbalance：先做 per-time 平均，再（可选）除以 log(J) 让量级更稳定
        imb_per_time = imbalance_raw / max(1, T)
        imb_denom = 1.0
        if str(imbalance_scale).lower().strip() == "logj":
            imb_denom = np.log(max(2, J))
        imbalance_norm = float(imb_per_time / (imb_denom + eps))

        build_norm = _build_norm(
            build_raw,
            J,
            mode=station_penalty_mode,
            power=station_penalty_power,
            kmin=budget_min,
            kmax=budget_max,
        )

        walk_cost = float(alpha_walk * walk_norm)
        overflow_cost = float(lam_over * overflow_norm)
        imbalance_cost = float(mu_imb * imbalance_norm)
        build_cost = float(eta_build * build_norm)

        total = float(walk_cost + overflow_cost + imbalance_cost + build_cost)

        return dict(
            total=total,
            walk_cost=walk_cost,
            overflow_cost=overflow_cost,
            imbalance_cost=imbalance_cost,
            build_cost=build_cost,
            # raw
            walk_raw=walk_raw,
            overflow_raw=overflow_raw,
            imbalance_raw=float(imbalance_raw),
            build_raw=build_raw,
            # norm
            walk_norm=float(walk_norm),
            overflow_norm=float(overflow_norm),
            imbalance_norm=float(imbalance_norm),
            build_norm=float(build_norm),
            # scales (for debugging / report)
            sumD=sumD,
            dist_scale=dist_scale,
            imbalance_scale=str(imbalance_scale),
            L=L,
            overflow=overflow,
        )

    # ---------- 旧版：不做归一化 ----------
    walk_cost = float(alpha_walk * walk_raw)
    overflow_cost = float(lam_over * overflow_raw)
    imbalance_cost = float(mu_imb * float(imbalance_raw))
    build_cost = float(eta_build * build_raw)
    total = float(walk_cost + overflow_cost + imbalance_cost + build_cost)

    return dict(
        total=total,
        walk_cost=walk_cost,
        overflow_cost=overflow_cost,
        imbalance_cost=imbalance_cost,
        build_cost=build_cost,
        walk_raw=walk_raw,
        overflow_raw=overflow_raw,
        imbalance_raw=float(imbalance_raw),
        build_raw=build_raw,
        walk_norm=None,
        overflow_norm=None,
        imbalance_norm=None,
        build_norm=None,
        sumD=float(D.sum()),
        dist_scale=None,
        imbalance_scale=None,
        L=L,
        overflow=overflow,
    )


def random_search(
    D,
    Dist,
    cap,
    *,
    budget_min=2,
    budget_max=None,
    n_samples=200,
    alpha_walk=1.0,
    lam_over=1.0,
    mu_imb=1.0,
    eta_build=1.0,
    normalize_terms=True,
    station_penalty_mode="minmax",
    station_penalty_power=2.0,
    dist_scale=None,
    imbalance_scale="logJ",
    rng_seed=0,
):
    """
    随机生成多组选址方案 y（在预算范围内），返回 total 最小的方案。

    建议：normalize_terms=True（默认），这样 Dist 的单位缩放不会改变最优解趋势。
    """
    rng = np.random.default_rng(rng_seed)
    J = Dist.shape[1]
    if budget_max is None:
        budget_max = J
    budget_min = int(budget_min)
    budget_max = int(budget_max)
    if budget_min < 0 or budget_max < 0 or budget_min > budget_max:
        raise ValueError("Invalid budget_min/budget_max")

    # 为了让不同样本的 walk_norm 可比，dist_scale 推荐固定一次
    if (dist_scale is None) and normalize_terms:
        dist_scale = _quantile_positive(Dist, q=0.95, fallback=1.0)

    best = None
    history = []

    for _ in range(int(n_samples)):
        budget = int(rng.integers(budget_min, budget_max + 1))
        y = np.zeros(J, dtype=float)
        active_js = rng.choice(J, size=budget, replace=False)
        y[active_js] = 1.0

        metrics = evaluate_solution(
            D,
            Dist,
            cap,
            y,
            alpha_walk=alpha_walk,
            lam_over=lam_over,
            mu_imb=mu_imb,
            eta_build=eta_build,
            normalize_terms=normalize_terms,
            station_penalty_mode=station_penalty_mode,
            station_penalty_power=station_penalty_power,
            budget_min=budget_min,
            budget_max=budget_max,
            dist_scale=dist_scale,
            imbalance_scale=imbalance_scale,
        )

        history.append(dict(y=y.copy(), **metrics))

        if (best is None) or (metrics["total"] < best["total"]):
            best = dict(y=y.copy(), **metrics)

    return best, history
# ============================================================
# Extra baselines for algorithm comparison: k-center / k-supplier
# (Designed to be OPTIONAL. Existing functions remain unchanged.)
# ============================================================

def _weighted_quantile(values, weights, q):
    """
    Weighted quantile for 1D arrays.
    q in [0,1]. Returns the smallest v such that CDF_w(v) >= q.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float(np.nan)
    v = v[mask]
    w = w[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cum = np.cumsum(w)
    cutoff = q * cum[-1]
    idx = int(np.searchsorted(cum, cutoff, side="left"))
    idx = min(max(idx, 0), len(v) - 1)
    return float(v[idx])


def distance_stats(D, Dist, y, q=0.95):
    """
    Compute distance statistics under 'nearest open station' rule.

    Returns:
      - wmean_dist: demand-weighted mean of min distance per demand cell
      - wp{q}_dist: demand-weighted quantile (default p95)
      - max_dist: max of min distance across demand cells
    """
    y = np.asarray(y, dtype=float)
    open_idx = np.where(y > 0.5)[0]
    if open_idx.size == 0:
        return dict(wmean_dist=float("inf"), wpq_dist=float("inf"), max_dist=float("inf"))

    # Aggregate demand weights per cell
    W = np.asarray(D, dtype=float).sum(axis=1)
    # Min distance to any open station
    dmin = Dist[:, open_idx].min(axis=1)

    wsum = float(W.sum())
    if wsum <= 0:
        # fallback to unweighted
        wmean = float(np.mean(dmin))
        wpq = float(np.quantile(dmin, q))
    else:
        wmean = float((W * dmin).sum() / wsum)
        wpq = _weighted_quantile(dmin, W, q)

    return dict(
        wmean_dist=wmean,
        wpq_dist=wpq,
        max_dist=float(np.max(dmin)),
    )


def greedy_ksupplier(D, Dist, k, *, seed=0, demand_positive_only=True):
    """
    Greedy k-supplier baseline (farthest-first driven by demand points, then
    project each selected demand point to its nearest supplier/station).

    - Demand points: I (cells)
    - Supplier candidates: J (stations)

    This baseline optimizes a *worst-case* coverage intuition (minimize max distance),
    and intentionally ignores overflow/imbalance/capacity to serve as a contrast
    against the k-median-style objective in this project.
    """
    rng = np.random.default_rng(seed)
    I, J = Dist.shape
    W = np.asarray(D, dtype=float).sum(axis=1)
    if demand_positive_only:
        cand_i = np.where(W > 0)[0]
        if cand_i.size == 0:
            cand_i = np.arange(I)
    else:
        cand_i = np.arange(I)

    # start from a random demand point among candidates (classic greedy often starts arbitrary)
    i0 = int(rng.choice(cand_i))
    selected_j = []
    selected_set = set()

    # current min distance from each i to selected suppliers
    min_d = np.full(I, np.inf, dtype=float)

    for _ in range(int(k)):
        # update min_d with last selected supplier if exists
        if selected_j:
            j_last = selected_j[-1]
            min_d = np.minimum(min_d, Dist[:, j_last])

        # pick farthest demand point (worst-covered)
        i_star = int(cand_i[np.argmax(min_d[cand_i])])

        # choose nearest supplier to i_star (projection)
        # avoid duplicates by scanning nearest list
        js_sorted = np.argsort(Dist[i_star])
        picked = None
        for j in js_sorted:
            if int(j) not in selected_set:
                picked = int(j)
                break
        if picked is None:
            break  # ran out of unique suppliers (shouldn't happen unless k > J)

        selected_j.append(picked)
        selected_set.add(picked)

    y = np.zeros(J, dtype=float)
    y[selected_j] = 1.0
    return y


def greedy_kcenter(D, Dist, k, *, seed=0, gamma=0.0):
    """
    A k-center-style greedy baseline with an OPTIONAL demand-awareness knob.

    We keep the same farthest-first skeleton, but choose the next 'farthest' demand point
    using an *effective distance*:
        eff(i) = minDist(i, S) * (w_i / mean_w)^gamma
    - gamma = 0.0 -> classic unweighted farthest-first (behaves like k-supplier here)
    - gamma > 0   -> high-demand cells exert more influence (a "weighted k-center" flavor)

    This is included to provide a visually distinct baseline for teaching/analysis.
    """
    rng = np.random.default_rng(seed)
    I, J = Dist.shape
    W = np.asarray(D, dtype=float).sum(axis=1)
    mean_w = float(np.mean(W[W > 0])) if np.any(W > 0) else 1.0
    weight_factor = np.ones(I, dtype=float)
    if gamma and gamma > 0:
        weight_factor = np.power(np.maximum(W / mean_w, 1e-12), gamma)

    cand_i = np.where(W > 0)[0] if np.any(W > 0) else np.arange(I)
    i0 = int(rng.choice(cand_i))

    selected_j = []
    selected_set = set()
    min_d = np.full(I, np.inf, dtype=float)

    for _ in range(int(k)):
        if selected_j:
            j_last = selected_j[-1]
            min_d = np.minimum(min_d, Dist[:, j_last])

        eff = min_d * weight_factor
        i_star = int(cand_i[np.argmax(eff[cand_i])])

        js_sorted = np.argsort(Dist[i_star])
        picked = None
        for j in js_sorted:
            if int(j) not in selected_set:
                picked = int(j)
                break
        if picked is None:
            break

        selected_j.append(picked)
        selected_set.add(picked)

    y = np.zeros(J, dtype=float)
    y[selected_j] = 1.0
    return y
