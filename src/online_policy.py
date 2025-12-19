# src/online_policy.py
import numpy as np


class LinUCB:
    """
    简单版 LinUCB，用于情境多臂赌博机。
    """
    def __init__(self, dim, alpha=1.0):
        self.dim = dim
        self.alpha = alpha
        self.A = np.eye(dim)
        self.b = np.zeros(dim)

    def select(self, features_list):
        """
        输入若干候选动作的特征向量列表，返回被选中的动作下标。
        features_list: list[np.ndarray]，每个形状为 [dim]
        """
        if len(features_list) == 1:
            return 0

        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b

        feats = np.stack(features_list, axis=0)  # [K, dim]
        mu = feats @ theta
        sigma = np.sqrt(np.einsum("kd,df,kf->k", feats, A_inv, feats))
        ucb = mu + self.alpha * sigma
        idx = int(np.argmax(ucb))
        return idx

    def update(self, feature, reward):
        """
        用真实 reward 对选中动作的特征进行更新。
        """
        feature = np.asarray(feature, dtype=float)
        self.A += np.outer(feature, feature)
        self.b += reward * feature


def build_feature_vector(
    dist_ij,
    load_j_ratio,
    overflow_j,
    imbalance_t,
    action_bias=0.0,
):
    """
    一个非常简洁的特征向量构造方式，用于在线策略：
      dist_ij: 当前用户到候选站点的距离
      load_j_ratio: 当前站点负载 / 容量
      overflow_j: 当前站点 overflow
      imbalance_t: 当前全局 Imbalance 指标
      action_bias: 可以为不同类型动作预留一个偏置
    """
    return np.array(
        [
            dist_ij,
            load_j_ratio,
            overflow_j,
            imbalance_t,
            action_bias,
        ],
        dtype=float,
    )
