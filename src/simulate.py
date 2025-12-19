# src/simulate.py
import numpy as np
from .online_policy import build_feature_vector


class BikeEnv:
    """
    简化版的共享单车环境：
      给定离线选址 y 和容量 cap，
      按 D[i,t] 的平均值生成 Poisson 到达，
      每个到达时，让在线策略选择落地站点 j。
    """

    def __init__(self, D, Dist, cap, y,
                 alpha_walk=1.0, lam_over=1.0, mu_imb=1.0,
                 rng_seed=0):
        self.D = D
        self.Dist = Dist
        self.cap = cap
        self.y = np.asarray(y, dtype=float)

        self.alpha_walk = alpha_walk
        self.lam_over = lam_over
        self.mu_imb = mu_imb

        self.rng = np.random.default_rng(rng_seed)

        I, T = D.shape
        J = Dist.shape[1]
        self.I = I
        self.J = J
        self.T = T

        self.reset()

    def reset(self):
        """
        把时间 t 和负载 L 初始化为零。
        """
        self.t = 0
        self.L = np.zeros((self.J,), dtype=float)
        self.history = dict(
            walk_cost=[],
            overflow_cost=[],
            imbalance_cost=[],
            total_reward=[],
        )

    def _current_imbalance(self):
        """
        计算当前时刻全局 Imbalance = KL(pi || pi*)，和 offline 模型保持一致。
        """
        eps = 1e-9
        total_L = self.L.sum()
        if total_L < eps:
            return 0.0

        pi = self.L / (total_L + eps)
        cap_active = self.cap * self.y
        if cap_active.sum() < eps:
            return 0.0
        pi_star = cap_active / (cap_active.sum() + eps)
        imbalance = np.sum(pi * np.log((pi + eps) / (pi_star + eps)))
        return float(imbalance)

    def step_one_time_slice(self, policy=None, max_arrivals_per_cell=20):
        """
        模拟单个时间片 t 内的所有用户到达，并用给定策略做决策。
        policy: 需要有 select(features_list), update(feature, reward) 接口。
        """
        I = self.I
        J = self.J
        t = self.t

        D_t = self.D[:, t]

        total_walk = 0.0
        total_overflow_pen = 0.0
        total_imb_pen = 0.0

        for i in range(I):
            lam = D_t[i]
            if lam <= 0:
                continue
            # 该 cell 在该时间片的到达数量
            k = self.rng.poisson(lam=lam)
            k = int(min(k, max_arrivals_per_cell))
            if k <= 0:
                continue

            for _ in range(k):
                # 只考虑已启用的站点作为候选
                active_js = np.where(self.y > 0.5)[0]
                if active_js.size == 0:
                    continue
                # 拿若干最近的候选作为动作
                dists_i = self.Dist[i, active_js]
                idx_sorted = np.argsort(dists_i)
                K = min(3, len(idx_sorted))
                cand_js = active_js[idx_sorted[:K]]

                imbalance_before = self._current_imbalance()

                features_list = []
                for rank, j in enumerate(cand_js):
                    dist_ij = self.Dist[i, j]
                    load_ratio = (
                        self.L[j] / self.cap[j]
                        if self.cap[j] > 0 else 0.0
                    )
                    overflow_j = max(0.0, self.L[j] - self.cap[j])
                    feat = build_feature_vector(
                        dist_ij=dist_ij,
                        load_j_ratio=load_ratio,
                        overflow_j=overflow_j,
                        imbalance_t=imbalance_before,
                        action_bias=float(rank),
                    )
                    features_list.append(feat)

                if policy is None:
                    chosen_idx = self.rng.integers(0, len(cand_js))
                else:
                    chosen_idx = policy.select(features_list)

                j_chosen = cand_js[chosen_idx]
                feature_chosen = features_list[chosen_idx]

                # 落地这辆车
                dist_ij = self.Dist[i, j_chosen]
                self.L[j_chosen] += 1.0

                # 计算 overflow 和 Imbalance 的变化
                overflow_after = max(0.0, self.L[j_chosen] - self.cap[j_chosen])
                overflow_before = max(0.0, self.L[j_chosen] - 1.0 - self.cap[j_chosen])
                delta_overflow = overflow_after - overflow_before

                imbalance_after = self._current_imbalance()
                delta_imbalance = imbalance_after - imbalance_before

                # 奖励函数（负号，因为越大越坏）
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

        total_reward = - (total_walk + total_overflow_pen + total_imb_pen)

        self.history["walk_cost"].append(total_walk)
        self.history["overflow_cost"].append(total_overflow_pen)
        self.history["imbalance_cost"].append(total_imb_pen)
        self.history["total_reward"].append(total_reward)

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
