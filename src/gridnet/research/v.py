import matplotlib.pyplot as plt
import networkx as nx
from simulator import generate_transmission_topology

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Topology scaling: avg_degree target=2.8", fontsize=12, fontweight="bold")

for ax, n in zip(axes.flat, [3, 5, 8, 10, 15, 20, 50, 100]):
    buses, branches = generate_transmission_topology(n_buses=n, seed=0)
    G = nx.Graph()
    for bus in buses:
        G.add_node(bus.bus_id)
    for br in branches:
        G.add_edge(br.from_bus, br.to_bus)
    avg_deg = 2 * len(branches) / n
    pos = nx.spring_layout(G, seed=7)

    colors = [
        "gold"
        if b.bus_type == "slack"
        else "tomato"
        if b.bus_type == "PV"
        else "steelblue"
        for b in buses
    ]
    nx.draw(
        G,
        pos,
        ax=ax,
        node_color=colors,
        node_size=max(30, 300 // n * 5),
        width=0.8,
        edge_color="gray",
        with_labels=n <= 15,
        font_size=7,
    )
    ax.set_title(f"n={n}  edges={len(branches)}  avg_deg={avg_deg:.2f}", fontsize=9)

plt.tight_layout()
plt.savefig("./topology_scaling.png", dpi=130, bbox_inches="tight")
print("saved")
