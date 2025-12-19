# src/data_prep.py
from pathlib import Path
import pandas as pd
import numpy as np


def load_data(data_dir="data/processed"):
    """
    读取 demand.csv, stations.csv, dist_matrix.csv 三个文件，
    输出:
      D: ndarray [I, T]  每个需求单元 i、时间片 t 的需求 d_{i,t}
      Dist: ndarray [I, J]  需求单元到站点的距离 dist(i,j)
      cap: ndarray [J]  每个站点的基础容量 cap_j
      metadata: dict  各种索引和 id 信息
      stations: DataFrame  原始站点信息表
    """
    data_dir = Path(data_dir)

    demand = pd.read_csv(data_dir / "demand.csv")
    stations = pd.read_csv(data_dir / "stations.csv")
    dist_df = pd.read_csv(data_dir / "dist_matrix.csv")

    I_ids = sorted(demand["i_id"].unique())
    J_ids = sorted(stations["j_id"].unique())
    T_vals = sorted(demand["t"].unique())

    i_index = {i: idx for idx, i in enumerate(I_ids)}
    j_index = {j: idx for idx, j in enumerate(J_ids)}
    t_index = {t: idx for idx, t in enumerate(T_vals)}

    # 需求矩阵 D[i, t]
    D = np.zeros((len(I_ids), len(T_vals)), dtype=float)
    for _, row in demand.iterrows():
        ii = i_index[row["i_id"]]
        tt = t_index[row["t"]]
        D[ii, tt] = row["demand"]

    # 距离矩阵 Dist[i, j]
    Dist = np.zeros((len(I_ids), len(J_ids)), dtype=float)
    for _, row in dist_df.iterrows():
        ii = i_index[row["i_id"]]
        jj = j_index[row["j_id"]]
        Dist[ii, jj] = row["dist_ij"]

    # 容量向量 cap[j]
    cap = (
        stations.set_index("j_id")["base_cap"]
        .reindex(J_ids)
        .to_numpy(dtype=float)
    )

    metadata = dict(
        I_ids=I_ids,
        J_ids=J_ids,
        T_vals=T_vals,
        i_index=i_index,
        j_index=j_index,
        t_index=t_index,
    )

    return D, Dist, cap, metadata, stations


def describe_data(D, Dist, cap, metadata):
    """
    打印一些基本信息，方便在 01_data_explore.ipynb 里快速看数据结构。
    """
    I, T = D.shape
    J = Dist.shape[1]
    print("Number of demand cells (I):", I)
    print("Number of stations (J):", J)
    print("Number of time slices (T):", T)
    print("Total demand:", D.sum())
    print("Average demand per hour:", D.sum(axis=0).mean())
    print("Total base capacity:", cap.sum())
    print("Time slices:", metadata["T_vals"])
