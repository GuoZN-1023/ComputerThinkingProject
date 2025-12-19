# src/viz.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def demand_grid_from_dataframe(demand_df, grid_n, hour):
    """
    假设 i_id 是按 i = row * grid_n + col 生成，
    把指定 hour 的需求 reshape 成 [grid_n, grid_n] 矩阵。
    """
    df_h = (
        demand_df[demand_df["t"] == hour]
        .sort_values("i_id")
    )
    vals = df_h["demand"].to_numpy()
    if vals.size != grid_n * grid_n:
        raise ValueError(
            f"期望 {grid_n*grid_n} 个 i_id，实际只有 {vals.size} 个，"
            "检查一下 i_id 是否按网格生成。"
        )
    grid = vals.reshape(grid_n, grid_n)
    return grid


def plot_morning_evening_heatmap(demand_csv_path, grid_n=5,
                                 morning_h=8, evening_h=18):
    """
    在 01_data_explore 里可以直接调用这个函数，
    一次性画出早高峰和晚高峰两个热力图。
    """
    demand = pd.read_csv(demand_csv_path)

    grid_m = demand_grid_from_dataframe(demand, grid_n, morning_h)
    grid_e = demand_grid_from_dataframe(demand, grid_n, evening_h)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(grid_m, origin="lower")
    axes[0].set_title(f"Demand heatmap at {morning_h}:00")
    axes[0].set_xlabel("y")
    axes[0].set_ylabel("x")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(grid_e, origin="lower")
    axes[1].set_title(f"Demand heatmap at {evening_h}:00")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("x")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def plot_station_layout(stations_df, y=None):
    """
    把站点在平面上画出来，y 可以表示哪些站点被选中。
    """
    xs = stations_df["lon"].to_numpy()
    ys = stations_df["lat"].to_numpy()

    fig, ax = plt.subplots(figsize=(5, 5))

    if y is None:
        ax.scatter(xs, ys)
    else:
        y = np.asarray(y, dtype=float)
        active = y > 0.5
        ax.scatter(xs[~active], ys[~active], marker="o", label="inactive")
        ax.scatter(xs[active], ys[active], marker="s", label="active")
        ax.legend()

    ax.set_xlabel("lon (toy)")
    ax.set_ylabel("lat (toy)")
    ax.set_title("Station layout")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_offline_history(history):
    """
    把 random_search 的历史搜索结果画成散点图，可以看到
    不同建设规模下总损失的分布。
    """
    budgets = [h["y"].sum() for h in history]
    totals = [h["total"] for h in history]

    plt.figure(figsize=(5, 4))
    plt.scatter(budgets, totals)
    plt.xlabel("Number of active stations (budget)")
    plt.ylabel("Total loss")
    plt.title("Offline random search results")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_online_curves(history):
    """
    把在线仿真中的各项损失随时间片变化画出来。
    """
    walk = np.array(history["walk_cost"])
    overflow = np.array(history["overflow_cost"])
    imb = np.array(history["imbalance_cost"])
    total_r = np.array(history["total_reward"])

    T = len(walk)
    t = np.arange(T)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    axes[0, 0].plot(t, walk)
    axes[0, 0].set_title("Walk cost per time slice")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_ylabel("walk_cost")
    axes[0, 0].grid(True)

    axes[0, 1].plot(t, overflow)
    axes[0, 1].set_title("Overflow penalty per time slice")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].set_ylabel("overflow_penalty")
    axes[0, 1].grid(True)

    axes[1, 0].plot(t, imb)
    axes[1, 0].set_title("Imbalance penalty per time slice")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].set_ylabel("imbalance_penalty")
    axes[1, 0].grid(True)

    axes[1, 1].plot(t, total_r)
    axes[1, 1].set_title("Total reward per time slice")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].set_ylabel("total_reward")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()
