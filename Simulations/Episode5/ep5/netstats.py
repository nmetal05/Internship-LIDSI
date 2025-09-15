#!/usr/bin/env python3
import argparse, collections
import sumolib

def is_internal(e):
    f = getattr(e, "getFunction", lambda: None)()
    return f == "internal"

def lane_count(e):
    return len(getattr(e, "getLanes", lambda: [])())

def edge_type(e):
    t = getattr(e, "getType", lambda: None)()
    return t if t else "unknown"

def approx_roundabouts(net):
    # essaye plusieurs méthodes selon la version
    try:
        c = sum(1 for e in net.getEdges() if getattr(e, "isRoundabout", lambda: False)())
        if c: return c
    except Exception:
        pass
    try:
        return sum(1 for n in net.getNodes() if getattr(n, "getType", lambda: "")() == "roundabout")
    except Exception:
        return 0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("net")
    p.add_argument("--html", action="store_true")
    args = p.parse_args()

    net = sumolib.net.readNet(args.net)
    xmin, ymin, xmax, ymax = net.getBoundary()
    area_km2 = ((xmax - xmin) * (ymax - ymin)) / 1e6

    edges = [e for e in net.getEdges() if not is_internal(e)]
    nodes = net.getNodes()

    tot_edge_len = sum(e.getLength() for e in edges)
    tot_lane_len = sum(e.getLength() * lane_count(e) for e in edges)

    by_lanes = collections.Counter(lane_count(e) for e in edges)
    len_by_type = collections.Counter()
    for e in edges:
        len_by_type[edge_type(e)] += e.getLength()

    tls = len(net.getTrafficLights())
    prio = sum(1 for n in nodes if getattr(n, "getType", lambda: "")() == "priority")
    rbs = approx_roundabouts(net)

    # rendu
    def out(s): print(s)
    if args.html:
        out("<html><body>")
        out("<h2>Network</h2>")
        out(f"<p>Total Area: {area_km2:.2f} km²</p>")
        out("<h2>Edges</h2>")
        out(f"<p>Edge number: {len(edges)}<br/>"
            f"Edgelength sum: {tot_edge_len:.2f} m<br/>"
            f"Lanelength sum: {tot_lane_len:.2f} m</p>")
        out("<h3>Edges by lanes</h3><ul>")
        for k in sorted(by_lanes):
            out(f"<li>{k} lane(s): {by_lanes[k]}</li>")
        out("</ul>")
        out("<h3>Length by type (m)</h3><ul>")
        order = ["motorway","primary","secondary","tertiary","service","residential",
                 "pedestrian","steps","railway"]
        for t in order + [t for t in sorted(len_by_type) if t not in order]:
            if t in len_by_type:
                out(f"<li>{t}: {len_by_type[t]:.2f}</li>")
        out("</ul>")
        out("<h2>Nodes</h2>")
        out(f"<p>Total Junctions: {len(nodes)}<br/>"
            f"Priority: {prio}<br/>Traffic Lights: {tls}<br/>"
            f"Roundabouts (approx): {rbs}</p>")
        out("</body></html>")
    else:
        out("=== Network Topology ===")
        out(f"Total Area: {area_km2:.2f} km^2")
        out(f"Edges length: {tot_edge_len:.2f} m")
        out(f"Lanelength sum: {tot_lane_len:.2f} m")
        out(f"Total Junctions: {len(nodes)}")
        out(f"Priority: {prio}")
        out(f"Traffic Lights: {tls}")
        out(f"Roundabouts (approx): {rbs}")
        out("\nEdges by lanes:")
        for k in sorted(by_lanes):
            out(f"  {k} lane(s): {by_lanes[k]}")
        out("\nLength by type (m):")
        order = ["motorway","primary","secondary","tertiary","service","residential",
                 "pedestrian","steps","railway"]
        for t in order + [t for t in sorted(len_by_type) if t not in order]:
            out(f"  {t}: {len_by_type[t]:.2f}")

if __name__ == "__main__":
    main()
