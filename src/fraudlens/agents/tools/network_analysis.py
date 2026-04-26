"""Tool: transaction network analysis using NetworkX (mock — deterministic from customer_id)."""

from __future__ import annotations

import json

import networkx as nx
from langchain_core.tools import tool


def _build_customer_graph(customer_id: str, seed: int) -> nx.DiGraph:
    """Build a synthetic directed transaction graph seeded from customer data.

    The graph simulates a customer's transaction network: nodes are account IDs,
    edges are money flows. Structure is deterministic so the same customer_id
    always produces the same graph.
    """
    rng_seed = seed
    g = nx.DiGraph()

    node_count = 3 + (rng_seed % 8)
    nodes = [customer_id] + [f"ACC-{(rng_seed * (i + 7)) % 9999:04d}" for i in range(node_count - 1)]
    g.add_nodes_from(nodes)

    edge_count = node_count + (rng_seed % 5)
    for i in range(edge_count):
        src_idx = (rng_seed * (i + 1)) % len(nodes)
        dst_idx = (rng_seed * (i + 3) + 1) % len(nodes)
        if src_idx != dst_idx:
            amount = 100 + ((rng_seed * (i + 11)) % 9900)
            g.add_edge(nodes[src_idx], nodes[dst_idx], amount=float(amount))

    return g


@tool
def deep_network_analysis(customer_id: str, transaction_id: str) -> str:
    """Analyze the transaction network graph for a customer to detect layering or smurfing.

    Examines the customer's graph structure: degree centrality, clustering,
    unusual hub nodes, and circular fund flows — all indicators of money laundering.

    Args:
        customer_id: The sender account ID of the transaction under investigation.
        transaction_id: Transaction ID used as secondary seed for graph variance.

    Returns:
        JSON with node_count, edge_count, in_degree, out_degree, hub_detected,
        circular_flow_detected, and a risk_assessment string.
    """
    seed = (sum(ord(c) for c in customer_id) * 31 + sum(ord(c) for c in transaction_id)) % 10000
    g = _build_customer_graph(customer_id, seed)

    node_count = g.number_of_nodes()
    edge_count = g.number_of_edges()

    in_deg = dict(g.in_degree())
    out_deg = dict(g.out_degree())
    max_in = max(in_deg.values(), default=0)
    max_out = max(out_deg.values(), default=0)

    # Hub: a node receiving from >=3 different senders
    hub_detected = max_in >= 3

    # Circular flow: simple cycle exists in graph
    try:
        nx.find_cycle(g, orientation="original")
        circular_flow_detected = True
    except nx.NetworkXNoCycle:
        circular_flow_detected = False

    # Density as a layering signal (dense graph = many hops)
    density = round(nx.density(g), 4)
    high_density = density > 0.4

    risk_signals = []
    if hub_detected:
        risk_signals.append("hub_node_detected")
    if circular_flow_detected:
        risk_signals.append("circular_flow")
    if high_density:
        risk_signals.append("high_graph_density")

    risk_level = "high" if len(risk_signals) >= 2 else ("medium" if risk_signals else "low")

    return json.dumps(
        {
            "customer_id": customer_id,
            "node_count": node_count,
            "edge_count": edge_count,
            "max_in_degree": max_in,
            "max_out_degree": max_out,
            "graph_density": density,
            "hub_detected": hub_detected,
            "circular_flow_detected": circular_flow_detected,
            "risk_signals": risk_signals,
            "risk_level": risk_level,
            "risk_assessment": (f"Network shows {risk_level} risk. " + (f"Signals: {', '.join(risk_signals)}." if risk_signals else "No structural anomalies detected.")),
        }
    )
