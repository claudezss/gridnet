"""
Visualize base case vs post-contingency grid state.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
from gridnet.research.simulator import generate_transmission_topology, GridSimulator


def draw_grid_state(ax, G, buses, branches, state, title, dropped_ids=None):
    dropped_ids = set(dropped_ids or [])
    pos = nx.spring_layout(G, seed=7, k=1.5)

    # node colors = voltage
    v_vals = np.array([state["buses"][n]["V"] for n in G.nodes()])
    node_cmap = cm.RdYlGn
    v_norm = mcolors.Normalize(vmin=0.92, vmax=1.08)
    node_colors = [node_cmap(v_norm(v)) for v in v_vals]

    # node markers by type
    type_map = {n: state["buses"][n]["type"] for n in G.nodes()}

    for shape, marker in [("slack", "*"), ("PV", "^"), ("PQ", "o"), ("passive", "s")]:
        nodelist = [n for n in G.nodes() if type_map[n] == shape]
        if not nodelist:
            continue
        nc = [node_colors[list(G.nodes()).index(n)] for n in nodelist]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodelist,
            node_color=nc,
            node_shape=marker,
            node_size=250 if shape != "slack" else 400,
            ax=ax,
            edgecolors="black",
            linewidths=0.8,
        )

    # edge colors = loading%, dropped = dashed red
    edge_list = list(G.edges(data=True))
    for u, v, edata in edge_list:
        brid = edata.get("branch_id")
        if brid in dropped_ids:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                width=2.5,
                edge_color="red",
                style="dashed",
                ax=ax,
                alpha=0.8,
            )
        else:
            loading = state["branches"].get(brid, {}).get("loading_pct", 0)
            lc = cm.YlOrRd(min(loading / 100.0, 1.0))
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], width=1.8, edge_color=[lc], ax=ax, alpha=0.85
            )

    nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    return pos, v_norm, node_cmap


def plot_delta_distributions(axes, base_state, post_state, dropped_ids):
    dropped_ids = set(dropped_ids)

    # voltage delta
    bus_ids = list(base_state["buses"].keys())
    dv = [post_state["buses"][b]["V"] - base_state["buses"][b]["V"] for b in bus_ids]
    axes[0].bar(
        range(len(dv)), dv, color=["tomato" if d < 0 else "steelblue" for d in dv]
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("ΔV (post − base) per bus", fontsize=10)
    axes[0].set_xlabel("Bus index")
    axes[0].set_ylabel("ΔV [pu]")

    # loading delta (in-service only)
    br_ids = [b for b in base_state["branches"] if b not in dropped_ids]
    dl = [
        post_state["branches"][b]["loading_pct"]
        - base_state["branches"][b]["loading_pct"]
        for b in br_ids
    ]
    axes[1].bar(
        range(len(dl)), dl, color=["tomato" if d > 0 else "steelblue" for d in dl]
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Δloading% (post − base) per branch", fontsize=10)
    axes[1].set_xlabel("Branch index")
    axes[1].set_ylabel("Δloading [%]")


def main():
    n_buses = 6
    buses, branches = generate_transmission_topology(n_buses=n_buses, seed=42)
    sim = GridSimulator(buses, branches)

    # build nx graph for drawing (with branch_id on edges)
    G = nx.Graph()
    for bus in buses:
        G.add_node(bus.bus_id)
    for br in branches:
        G.add_edge(br.from_bus, br.to_bus, branch_id=br.branch_id)

    base_state = sim.solve()

    # pick a branch with moderate base loading to drop
    sorted_brs = sorted(
        [(bid, d["loading_pct"]) for bid, d in base_state["branches"].items()],
        key=lambda x: -x[1],
    )
    drop_id = sorted_brs[len(sorted_brs) // 4][0]  # ~75th percentile loading
    post_state = sim.apply_contingency([drop_id])

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        f"Grid Simulator — N-1 Contingency (branch {drop_id} dropped)",
        fontsize=13,
        fontweight="bold",
    )

    ax_base = fig.add_subplot(2, 3, 1)
    ax_post = fig.add_subplot(2, 3, 2)
    ax_legend = fig.add_subplot(2, 3, 3)
    ax_dv = fig.add_subplot(2, 3, 4)
    ax_dl = fig.add_subplot(2, 3, 5)
    ax_stats = fig.add_subplot(2, 3, 6)

    draw_grid_state(ax_base, G, buses, branches, base_state, "Base Case")
    _, v_norm, node_cmap = draw_grid_state(
        ax_post,
        G,
        buses,
        branches,
        post_state,
        f"Post-Contingency (branch {drop_id} ✕)",
        dropped_ids=[drop_id],
    )

    # legend panel
    ax_legend.axis("off")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_els = [
        Patch(facecolor=node_cmap(v_norm(0.93)), label="Low V (0.93 pu)"),
        Patch(facecolor=node_cmap(v_norm(1.00)), label="Nominal V (1.00 pu)"),
        Patch(facecolor=node_cmap(v_norm(1.07)), label="High V (1.07 pu)"),
        Line2D([0], [0], color=cm.YlOrRd(0.1), lw=2, label="Low loading (<10%)"),
        Line2D([0], [0], color=cm.YlOrRd(0.6), lw=2, label="Medium loading (~60%)"),
        Line2D([0], [0], color=cm.YlOrRd(1.0), lw=2, label="High loading (>100%)"),
        Line2D([0], [0], color="red", lw=2, linestyle="--", label="Dropped branch"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gray",
            markersize=12,
            label="Slack bus",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            label="PV bus (generator)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label="PQ bus (load)",
        ),
    ]
    ax_legend.legend(handles=legend_els, loc="center", fontsize=8, frameon=True)
    ax_legend.set_title("Legend", fontsize=10)

    plot_delta_distributions([ax_dv, ax_dl], base_state, post_state, [drop_id])

    # stats panel
    ax_stats.axis("off")
    bus_ids = list(base_state["buses"].keys())
    dv_arr = np.array(
        [post_state["buses"][b]["V"] - base_state["buses"][b]["V"] for b in bus_ids]
    )
    br_ids = [b for b in base_state["branches"] if b != drop_id]
    dl_arr = np.array(
        [
            post_state["branches"][b]["loading_pct"]
            - base_state["branches"][b]["loading_pct"]
            for b in br_ids
        ]
    )
    stats_text = (
        f"Network: {n_buses} buses, {len(branches)} branches\n"
        f"Dropped: branch {drop_id}\n\n"
        f"Base V:    [{min(v['V'] for v in base_state['buses'].values()):.3f}, "
        f"{max(v['V'] for v in base_state['buses'].values()):.3f}] pu\n"
        f"Post V:    [{min(v['V'] for v in post_state['buses'].values()):.3f}, "
        f"{max(v['V'] for v in post_state['buses'].values()):.3f}] pu\n\n"
        f"Max |ΔV|:  {np.max(np.abs(dv_arr)):.4f} pu\n"
        f"Max Δload: {np.max(dl_arr):.1f}%\n"
        f"Min Δload: {np.min(dl_arr):.1f}%\n\n"
        f"Branches > 80% loading (post): "
        f"{sum(1 for b in br_ids if post_state['branches'][b]['loading_pct'] > 80)}\n"
        f"Branches > 100% loading (post): "
        f"{sum(1 for b in br_ids if post_state['branches'][b]['loading_pct'] > 100)}\n"
    )
    ax_stats.text(
        0.05,
        0.95,
        stats_text,
        transform=ax_stats.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    ax_stats.set_title("Summary Stats", fontsize=10)

    plt.tight_layout()
    plt.savefig("./grid_sim_contingency.png", dpi=150, bbox_inches="tight")
    print("Saved.")


if __name__ == "__main__":
    main()
