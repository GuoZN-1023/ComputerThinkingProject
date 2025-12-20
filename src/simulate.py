# src/simulate.py
import numpy as np
from .online_policy import build_feature_vector


class BikeEnv:
    """
    简化版共享单车环境（可切换两种模式）：

    - trip_mode="deposit_only"（旧行为，兼容演示）：
        把 D[i,t] 当成“车辆落地需求”，每次到达只做 self.L[j] += 1。

    - trip_mode="pickup_dropoff"（更接近真实，默认）：
        把 D[i,t] 当成“出发（取车）需求”，每次到达会：
          1) 在起点 cell i 附近找一个有车的站点取走 1 辆（L[pick]-=1）
          2) 采样一个目的地 cell i2，并在 i2 附近选择一个站点还车（L[drop]+=1）
        因此总车数守恒，站点分布会随时间“迁移”，更容易出现
        早高峰抽空/晚高峰灌满的动态。

    说明：
      - 这里把 OD（起点到终点）用一个轻量的“需求滞后”近似：
        目的地 i2 的采样概率 ∝ D[i2, t + dest_time_lag]。
        dest_time_lag 可以理解为“骑行耗时 = 若干时间片”。
      - 在线策略仍只在“取车站点”上做选择（接口不变），
        还车站点用就近策略（优先有空位）选取，保持实现最小侵入。
    """

    def __init__(
        self,
        D,
        Dist,
        cap,
        y,
        alpha_walk=1.0,
        lam_over=1.0,
        mu_imb=1.0,
        rng_seed=0,
        # --- new knobs ---
        trip_mode="pickup_dropoff",
        init_fill_ratio=0.55,
        init_fill_jitter=0.10,
        dest_time_lag=1,
        stockout_penalty=2.0,
        K_pickup=3,
        K_dropoff=3,
    ):
        self.D = D
        self.Dist = Dist
        self.cap = np.asarray(cap, dtype=float)
        self.y = np.asarray(y, dtype=float)

        self.alpha_walk = float(alpha_walk)
        self.lam_over = float(lam_over)
        self.mu_imb = float(mu_imb)

        self.trip_mode = str(trip_mode)
        self.init_fill_ratio = float(init_fill_ratio)
        self.init_fill_jitter = float(init_fill_jitter)
        self.dest_time_lag = int(dest_time_lag)
        self.stockout_penalty = float(stockout_penalty)
        self.K_pickup = int(K_pickup)
        self.K_dropoff = int(K_dropoff)

        self.rng = np.random.default_rng(rng_seed)

        I, T = D.shape
        J = Dist.shape[1]
        self.I = int(I)
        self.T = int(T)
        self.J = int(J)

        # active station indices cache
        self.active_js = np.where(self.y > 0.5)[0]

        self.reset()

    # -------------------------
    # Initialization & metrics
    # -------------------------
    def reset(self):
        """
        初始化时间 t 与站点负载 L（站点内车辆数）。
        旧版本是全 0，会导致“只加车”的模式必然越堆越满；
        新版本在默认 trip_mode 下需要有初始车才能产生取车行为，
        因此按容量给一个合理的初始填充率。
        """
        self.t = 0

        L0 = np.zeros((self.J,), dtype=float)
        if self.active_js.size > 0:
            cap_active = self.cap[self.active_js]
            base = self.init_fill_ratio * cap_active
            if self.init_fill_jitter > 0:
                jitter = self.rng.normal(loc=0.0, scale=self.init_fill_jitter, size=cap_active.shape)
                base = base * (1.0 + jitter)
            base = np.clip(base, 0.0, cap_active)
            base = np.round(base)
            L0[self.active_js] = base

        self.L = np.clip(L0, 0.0, None)

        self.history = dict(
            walk_cost=[],
            overflow_cost=[],
            imbalance_cost=[],
            total_reward=[],
            # --- extras (safe) ---
            stockout_cost=[],
            # --- station-level snapshots (safe) ---
            L_hist=[],
            sat_hist=[],
            overflow_hist=[],
            used_stations=[],
            total_bikes_in_stations=[],
            num_full_stations=[],
            num_empty_stations=[],
        )
        return self.L

    def _current_imbalance(self):
        """
        计算当前时刻全局 Imbalance = KL(pi || pi*)，和 offline 模型保持一致。
        """
        eps = 1e-9
        total_L = float(self.L.sum())
        if total_L < eps:
            return 0.0

        pi = self.L / (total_L + eps)
        cap_active = self.cap * self.y
        if float(cap_active.sum()) < eps:
            return 0.0
        pi_star = cap_active / (float(cap_active.sum()) + eps)
        imbalance = float(np.sum(pi * np.log((pi + eps) / (pi_star + eps))))
        return imbalance

    # -------------------------
    # Trip mechanics (NEW)
    # -------------------------
    def _sample_destination_cell(self, t):
        """
        目的地 cell 采样：P(i2 | t) ∝ D[i2, t + lag]。
        """
        lag_t = (t + self.dest_time_lag) % self.T
        w = np.asarray(self.D[:, lag_t], dtype=float)
        w = np.clip(w, 0.0, None)
        s = float(w.sum())
        if s <= 1e-12:
            return int(self.rng.integers(0, self.I))
        p = w / s
        return int(self.rng.choice(self.I, p=p))

    def _choose_nearest_active(self, i, K):
        """返回 cell i 附近距离最近的 K 个 active station。"""
        if self.active_js.size == 0:
            return np.array([], dtype=int)
        dists_i = self.Dist[i, self.active_js]
        idx_sorted = np.argsort(dists_i)
        K = int(min(max(1, K), len(idx_sorted)))
        return self.active_js[idx_sorted[:K]]

    def _choose_dropoff_station(self, i2):
        """
        还车站点选择：就近优先有空位；若都满，则仍选择最近站点并允许 overflow。
        """
        cand = self._choose_nearest_active(i2, self.K_dropoff)
        if cand.size == 0:
            return None, 0.0

        for j in cand:
            if self.L[j] < self.cap[j] - 1e-9:
                return int(j), float(self.Dist[i2, j])

        j0 = int(cand[0])
        return j0, float(self.Dist[i2, j0])

    # -------------------------
    # Core simulation
    # -------------------------
    def step_one_time_slice(self, policy=None, max_arrivals_per_cell=20):
        """
        模拟单个时间片 t 内的所有用户到达，并用给定策略做决策。
        policy: 需要有 select(features_list), update(feature, reward) 接口。
        """
        I = self.I
        t = self.t
        D_t = self.D[:, t]

        L_before = self.L.copy()

        total_walk = 0.0
        total_overflow_pen = 0.0
        total_imb_pen = 0.0
        total_stockout_pen = 0.0

        for i in range(I):
            lam = float(D_t[i])
            if lam <= 0:
                continue

            k = int(self.rng.poisson(lam=lam))
            k = int(min(k, max_arrivals_per_cell))
            if k <= 0:
                continue

            for _ in range(k):
                if self.active_js.size == 0:
                    continue

                # -----------------------------
                # A) deposit-only (legacy)
                # -----------------------------
                if self.trip_mode == "deposit_only":
                    cand_js = self._choose_nearest_active(i, self.K_pickup)
                    if cand_js.size == 0:
                        continue

                    imbalance_before = self._current_imbalance()
                    features_list = []
                    for rank, j in enumerate(cand_js):
                        dist_ij = float(self.Dist[i, j])
                        load_ratio = float(self.L[j] / self.cap[j]) if self.cap[j] > 0 else 0.0
                        overflow_j = float(max(0.0, self.L[j] - self.cap[j]))
                        feat = build_feature_vector(
                            dist_ij=dist_ij,
                            load_j_ratio=load_ratio,
                            overflow_j=overflow_j,
                            imbalance_t=imbalance_before,
                            action_bias=float(rank),
                        )
                        features_list.append(feat)

                    chosen_idx = int(self.rng.integers(0, len(cand_js))) if policy is None else int(policy.select(features_list))
                    j_chosen = int(cand_js[chosen_idx])
                    feature_chosen = features_list[chosen_idx]

                    dist_ij = float(self.Dist[i, j_chosen])
                    self.L[j_chosen] += 1.0

                    overflow_after = max(0.0, self.L[j_chosen] - self.cap[j_chosen])
                    overflow_before = max(0.0, self.L[j_chosen] - 1.0 - self.cap[j_chosen])
                    delta_overflow = overflow_after - overflow_before

                    imbalance_after = self._current_imbalance()
                    delta_imbalance = imbalance_after - imbalance_before

                    r = (
                        - self.alpha_walk * dist_ij
                        - self.lam_over * delta_overflow
                        - self.mu_imb * delta_imbalance
                    )
                    if policy is not None:
                        policy.update(feature_chosen, r)

                    total_walk += self.alpha_walk * dist_ij
                    total_overflow_pen += self.lam_over * max(0.0, delta_overflow)
                    total_imb_pen += self.mu_imb * max(0.0, delta_imbalance)
                    continue

                # -----------------------------
                # B) pickup + dropoff (default)
                # -----------------------------
                cand_js = self._choose_nearest_active(i, self.K_pickup)
                if cand_js.size == 0:
                    continue

                cand_js_has_bike = cand_js[self.L[cand_js] >= 1.0 - 1e-9]
                if cand_js_has_bike.size == 0:
                    total_stockout_pen += self.stockout_penalty
                    continue

                imbalance_before = self._current_imbalance()

                features_list = []
                for rank, j in enumerate(cand_js_has_bike):
                    dist_ij = float(self.Dist[i, j])
                    load_ratio = float(self.L[j] / self.cap[j]) if self.cap[j] > 0 else 0.0
                    overflow_j = float(max(0.0, self.L[j] - self.cap[j]))
                    feat = build_feature_vector(
                        dist_ij=dist_ij,
                        load_j_ratio=load_ratio,
                        overflow_j=overflow_j,
                        imbalance_t=imbalance_before,
                        action_bias=float(rank),
                    )
                    features_list.append(feat)

                chosen_idx = int(self.rng.integers(0, len(cand_js_has_bike))) if policy is None else int(policy.select(features_list))
                j_pick = int(cand_js_has_bike[chosen_idx])
                feature_chosen = features_list[chosen_idx]

                walk_pick = float(self.Dist[i, j_pick])

                # pickup
                self.L[j_pick] -= 1.0
                if self.L[j_pick] < 0:
                    self.L[j_pick] = 0.0

                # destination + dropoff
                i2 = self._sample_destination_cell(t)
                j_drop, walk_drop = self._choose_dropoff_station(i2)
                if j_drop is None:
                    self.L[j_pick] += 1.0
                    total_stockout_pen += self.stockout_penalty
                    continue

                overflow_before = max(0.0, self.L[j_drop] - self.cap[j_drop])
                self.L[j_drop] += 1.0
                overflow_after = max(0.0, self.L[j_drop] - self.cap[j_drop])
                delta_overflow = overflow_after - overflow_before

                imbalance_after = self._current_imbalance()
                delta_imbalance = imbalance_after - imbalance_before

                walk_total = walk_pick + walk_drop
                r = (
                    - self.alpha_walk * walk_total
                    - self.lam_over * delta_overflow
                    - self.mu_imb * delta_imbalance
                )

                if policy is not None:
                    policy.update(feature_chosen, r)

                total_walk += self.alpha_walk * walk_total
                total_overflow_pen += self.lam_over * max(0.0, delta_overflow)
                total_imb_pen += self.mu_imb * max(0.0, delta_imbalance)

        total_reward = - (total_walk + total_overflow_pen + total_imb_pen + total_stockout_pen)

        self.history["walk_cost"].append(total_walk)
        self.history["overflow_cost"].append(total_overflow_pen)
        self.history["imbalance_cost"].append(total_imb_pen)
        self.history["stockout_cost"].append(total_stockout_pen)
        self.history["total_reward"].append(total_reward)

        cap_arr = self.cap
        L_now = self.L
        inc = L_now - L_before
        used_cnt = int(np.sum(np.abs(inc) > 1e-9))

        self.history["L_hist"].append(L_now.copy())
        self.history["sat_hist"].append((L_now / np.maximum(cap_arr, 1e-9)).copy())
        self.history["overflow_hist"].append(np.maximum(0.0, L_now - cap_arr).copy())
        self.history["used_stations"].append(used_cnt)
        self.history["total_bikes_in_stations"].append(float(L_now.sum()))
        self.history["num_full_stations"].append(int(np.sum(L_now >= cap_arr - 1e-9)))
        self.history["num_empty_stations"].append(int(np.sum(L_now <= 1e-9)))

        self.t += 1
        done = self.t >= self.T
        return done

    def run_episode(self, policy=None):
        """
        从 t=0 一直跑到 t=T-1。
        """
        self.reset()
        done = False
        while not done:
            done = self.step_one_time_slice(policy=policy)
        return self.history