import pandas as pd
import networkx as nx
import numpy as np
import pygame
import math
from itertools import combinations
import pulp
import os
import csv

import imageio

# =============================================================
# PYGAME SETUP
# =============================================================
pygame.init()
infoObject = pygame.display.Info()
WIDTH, HEIGHT = infoObject.current_w, infoObject.current_h
FPS = 10
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Chinese Postman Problem Animation")
clock = pygame.time.Clock()
font_large = pygame.font.Font(None, 36)
font_small = pygame.font.Font(None, 16)
controls_text = font_small.render("SPACE: Next | ESC: Quit | ARROW KEYS: Seek", True, (100, 100, 100))
legend_original = font_small.render("Original", True, (100, 100, 100))
legend_duplicated = font_small.render("Duplicated", True, (100, 100, 100))

# =============================================================
# FRAME STORAGE
# =============================================================
frames = []
duplicated_edges = {}  # Track count of duplications per edge: {(u,v): count}
startup_frame = True  # Flag to show startup screen

def save_frame(G, highlight_nodes=None, highlight_edges=None, title="", step_info="", traveler_pos=None, cumulative_path=None, duplicated_edges=None):
    # Use provided duplicated_edges override if given, otherwise fall back to global duplicated_edges
    global_dup = globals().get('duplicated_edges', {})
    dup_to_store = dict(duplicated_edges) if duplicated_edges is not None else dict(global_dup)

    frames.append({
        "nodes": list(G.nodes()),
        "edges": list(G.edges()),
        "highlight_nodes": highlight_nodes or [],
        "highlight_edges": highlight_edges or [],
        "title": title,
        "step_info": step_info,
        "graph": G.copy(),
        "duplicated_edges": dup_to_store,
        "traveler_pos": traveler_pos,
        "cumulative_path": cumulative_path or []
    })

# =============================================================
# STEP 0 — LOAD GRAPH
# =============================================================
FILE = os.path.join(os.path.dirname(__file__), "Graaf Randwijk - Copy.xlsx")
df = pd.read_excel(FILE, index_col=0)
df.index = df.index.astype(str)
df.columns = df.columns.astype(str)
df = (df + df.T) / 2

nodes = list(df.index)
G = nx.MultiGraph()
G.add_nodes_from(nodes)

for i, u in enumerate(nodes):
    for j, v in enumerate(nodes):
        if j <= i:
            continue
        w = df.loc[u, v]
        if w != 0:
            G.add_edge(u, v, weight=w)

save_frame(G, title="Initial Graph", step_info="Loaded from Graaf Randwijk.xlsx")

# =============================================================
# STEP 1 — Dominating Set calculation 
# =============================================================
# Calculate distance matrix and dominating set (for later use, but no animation)
R = 500
dist = dict(nx.floyd_warshall(G, weight="weight"))

G_reach = nx.Graph()
G_reach.add_nodes_from(G.nodes())
for u, v in combinations(G.nodes(), 2):
    if dist[u][v] <= R:
        G_reach.add_edge(u, v)

prob = pulp.LpProblem("Minimum_Dominating_Set", pulp.LpMinimize)
x = {v: pulp.LpVariable(f"x_{v}", cat="Binary") for v in G_reach.nodes()}
prob += pulp.lpSum(x[v] for v in G_reach.nodes())

for v in G_reach.nodes():
    prob += pulp.lpSum(x[u] for u in [v] + list(G_reach.neighbors(v))) >= 1

prob.solve(pulp.PULP_CBC_CMD(msg=False))
dominating_set = [v for v in G_reach.nodes() if x[v].varValue == 1]

# =============================================================
# STEP 2 — ODD NODES
# =============================================================
odd_nodes = [v for v in G.nodes() if G.degree(v) % 2 == 1]


# =============================================================
# STEP 3 — MATCHING
# =============================================================
K = nx.Graph()
for u, v in combinations(odd_nodes, 2):
    K.add_edge(u, v, weight=dist[u][v])

matching = nx.min_weight_matching(K)

for u, v in matching:
    path = nx.shortest_path(G, u, v, weight="weight")

    highlight_path_edges = list(zip(path[:-1], path[1:]))

    save_frame(
        G,
        highlight_edges=highlight_path_edges,
        title="Matching Odd Nodes",
        step_info=f"Pairing odd-degree vertices via shortest path: {u} - {v}"
    )


# =============================================================
# STEP 4 — DUPLICATE EDGES
# =============================================================
for u, v in matching:
    path = nx.shortest_path(G, u, v, weight="weight")

    for a, b in zip(path[:-1], path[1:]):
        G.add_edge(a, b, weight=df.loc[a, b])

        edge_key = (min(a, b), max(a, b))
        duplicated_edges[edge_key] = duplicated_edges.get(edge_key, 0) + 1

        # Save frame AFTER EACH EDGE DUPLICATION
        save_frame(
            G,
            highlight_edges=[(a, b)],
            title="Making Graph Eulerian",
            step_info=f"Duplicating edge {a} - {b}",
            duplicated_edges=dict(duplicated_edges)
        )



# =============================================================
# STEP 5 — EULER TOUR
# =============================================================
tour = list(nx.eulerian_circuit(G, keys=True))

# Reorder tour to traverse duplicated edges first when possible
def reorder_tour_duplicated_first(tour, duplicated_edges_dict):
    """Reorder tour so duplicated edges are traversed before original edges."""
    reordered = []
    used = set()
    
    # First traverse all duplicated edges
    for idx, (u, v, key) in enumerate(tour):
        edge_key = (min(u, v), max(u, v))
        if duplicated_edges_dict.get(edge_key, 0) > 0:
            reordered.append((u, v, key))
            used.add(idx)
    
    # Then traverse original edges
    for idx, (u, v, key) in enumerate(tour):
        if idx not in used:
            reordered.append((u, v, key))
    
    return reordered

# Keep the Eulerian circuit order (continuous), but choose a start index
# that maximizes a run of duplicated edges so duplicated edges are traversed early
def rotate_tour_to_max_duplicated_run(tour, duplicated_edges_dict):
    flags = [duplicated_edges_dict.get((min(u, v), max(u, v)), 0) > 0 for u, v, k in tour]
    if not any(flags):
        return tour

    # find longest consecutive run of True in the circular list
    n = len(flags)
    max_run = 0
    best_idx = 0
    for i in range(n):
        run = 0
        for j in range(n):
            if flags[(i + j) % n]:
                run += 1
            else:
                break
        if run > max_run:
            max_run = run
            best_idx = i

    if best_idx == 0:
        return tour
    return tour[best_idx:] + tour[:best_idx]

tour = rotate_tour_to_max_duplicated_run(tour, duplicated_edges)
total_cost = 0
for (u, v, key) in tour:
    total_cost += G[u][v][key]["weight"]
# Create frames with moving traveler along edges - traverse duplicated edges first (tour already reordered)
cumulative_path = []
for idx, (u, v, key) in enumerate(tour, start=1):
    for progress in (0.0, 0.5, 1.0):
        if progress == 1.0:
            cumulative_path.append((u, v, key))

        save_frame(
            G,
            highlight_edges=[(u, v)],
            title="Hierholzer's Algorithm - Traversing Eulerian Circuit",
            step_info=f"Traversed {len(cumulative_path)}/{len(tour)} edges",
            traveler_pos=(u, v, progress),
            cumulative_path=list(cumulative_path),
            duplicated_edges=duplicated_edges
        )

save_frame(
     G,
    title="CPP Solution",
    step_info=f"Total cost: {total_cost} | Edges traversed: {len(tour)}",
    cumulative_path=list(cumulative_path),
    duplicated_edges=duplicated_edges
)

# =============================================================
# STEP 6 — BRANCH & BOUND FOR MDS
# =============================================================
save_frame(
    G,
    title="Branch & Bound - Finding Minimum Dominating Set",
    step_info=("Branch & Bound: maak een binaire zoekboom; bij elke stap kies je of een knoop in de\n"
               "verzameling zit of niet. Begin bij de knoop met de meeste verbindingen om sneller te\n"
               "een goede grens te vinden.")
)

# Implement Branch & Bound to find MDS
def is_dominating_set(candidate_set, graph):
    """Check if a set is a dominating set."""
    for v in graph.nodes():
        if v not in candidate_set:
            neighbors = list(graph.neighbors(v))
            if not any(u in candidate_set for u in neighbors):
                return False
    return True

# Branch & Bound: explore subsets
nodes_to_cover = list(G_reach.nodes())
best_mds = dominating_set.copy()
best_size = len(best_mds)

# Prioritize starting Branch & Bound with the node that has the highest degree (most connections)
if nodes_to_cover:
    start_node = max(nodes_to_cover, key=lambda n: G_reach.degree(n))
    nodes_to_cover = [start_node] + [n for n in sorted(nodes_to_cover, key=lambda n: G_reach.degree(n), reverse=True) if n != start_node]
    # show the chosen start node briefly
    save_frame(G, highlight_nodes=[start_node], title="Branch & Bound - Start Node", step_info=f"Starting with node {start_node} (highest degree)")

# Explore branches by systematically including/excluding nodes
explored_branches = 0
for branch_size in range(1, best_size):
    # Try different combinations of branch_size nodes
    num_combinations = min(10, len(list(combinations(nodes_to_cover, branch_size))))
    
    for idx, candidate_subset in enumerate(list(combinations(nodes_to_cover, branch_size))[:num_combinations]):
        candidate_set = set(candidate_subset)
        explored_branches += 1
        
        # Check if this could lead to a solution (pruning)
        uncovered = set()
        for v in G_reach.nodes():
            if v not in candidate_set and not any(u in candidate_set for u in [v] + list(G_reach.neighbors(v))):
                uncovered.add(v)
        
        # Estimate: lower bound on additional nodes needed
        lower_bound = len(candidate_set) + (len(uncovered) + 1) // 2
        
        if lower_bound < best_size:
            if is_dominating_set(candidate_set, G_reach):
                best_mds = candidate_set.copy()
                best_size = len(best_mds)
                save_frame(G, highlight_nodes=list(best_mds), 
                          title="Branch & Bound - Better Solution Found", 
                          step_info=f"Improved MDS size: {best_size} | Branch {explored_branches}")
            else:
                save_frame(G, highlight_nodes=list(candidate_set), 
                          title="Branch & Bound - Exploring Branch", 
                          step_info=f"Testing subset of size {branch_size} | Lower bound: {lower_bound} | Branch {explored_branches}")
        else:
            save_frame(G, highlight_nodes=list(candidate_set), 
                      title="Branch & Bound - Pruning Branch", 
                      step_info=f"Pruned: bound {lower_bound} >= best {best_size} | Branch {explored_branches}")
        
        if explored_branches >= 20:  # Limit animation length
            break
    
    if explored_branches >= 20:
        break

# Final MDS
save_frame(G, highlight_nodes=list(best_mds), 
          title="Branch & Bound - Optimal MDS Found", 
          step_info=f"Final MDS size: {best_size}, Total branches explored: {explored_branches}")

# =============================================================
# VISUALIZATION SETUP
# =============================================================
def compute_node_positions(G):
    """Compute positions using Kamada-Kawai layout which naturally places connected nodes closer."""
    # Kamada-Kawai is designed to keep adjacent nodes close together
    pos = nx.kamada_kawai_layout(G)
    
    # Scale positions to screen
    if pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        margin = 60
        width = WIDTH - 2 * margin
        height = HEIGHT - 2 * margin
        
        scaled_pos = {}
        for node, (x, y) in pos.items():
            if x_max - x_min > 0:
                scaled_x = margin + (x - x_min) / (x_max - x_min) * width
            else:
                scaled_x = WIDTH / 2
            
            if y_max - y_min > 0:
                scaled_y = margin + (y - y_min) / (y_max - y_min) * height
            else:
                scaled_y = HEIGHT / 2 - 50
            
            scaled_pos[node] = (scaled_x, scaled_y)
    
    return scaled_pos

def draw_dashed_line(surface, color, start_pos, end_pos, width=1, dash_length=5):
    """Draw a dashed/dotted line."""
    x1, y1 = start_pos
    x2, y2 = end_pos
    
    dx = x2 - x1
    dy = y2 - y1
    distance = math.sqrt(dx*dx + dy*dy)
    
    if distance == 0:
        return
    
    dx = dx / distance
    dy = dy / distance
    
    is_dash = True
    current_x, current_y = x1, y1
    
    while True:
        next_x = current_x + dx * dash_length
        next_y = current_y + dy * dash_length
        
        if dx * (next_x - x1) + dy * (next_y - y1) > dx * (x2 - x1) + dy * (y2 - y1):
            next_x, next_y = x2, y2
        
        if is_dash:
            pygame.draw.line(surface, color, (int(current_x), int(current_y)), (int(next_x), int(next_y)), width)

        
        is_dash = not is_dash
        current_x, current_y = next_x, next_y
        
        if (dx * (next_x - x1) + dy * (next_y - y1)) >= (dx * (x2 - x1) + dy * (y2 - y1)):
            next_x, next_y = x2, y2
            if is_dash:
                pygame.draw.line(surface, color, (int(current_x), int(current_y)), (int(next_x), int(next_y)), width)

            break

# Compute positions once
pos = compute_node_positions(G)
int_pos = {node: (int(x), int(y)) for node, (x, y) in pos.items()}

def draw_graph(G, pos, highlight_nodes=None, highlight_edges=None, title="", step_info="",
               duplicated_edges=None, traveler_pos=None, cumulative_path=None,
               surface_override=None):
    
    surface = surface_override or screen
    surface.fill((240, 240, 240))

    highlight_nodes = highlight_nodes or []
    highlight_edges = highlight_edges or []
    duplicated_edges = duplicated_edges or {}
    cumulative_path = cumulative_path or []

    # --- DRAW EDGES (original and duplicated) ---
    seen_edges = set()
    # Normalize highlight edges once
    highlight_set = {tuple(sorted(e)) for e in highlight_edges}

    for u, v in G.edges():
        edge_key = tuple(sorted((u, v)))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        x1, y1 = int_pos[u]
        x2, y2 = int_pos[v]

        color = (255, 0, 0) if edge_key in highlight_set else (100, 100, 100)

        pygame.draw.line(
            surface,
            color,
            (x1, y1),
            (x2, y2),
            2 if color == (255, 0, 0) else 1
    )

    # --- Draw duplicated edges (dashed) ---

    # Build set of traversed edges (ignore direction)
    traversed_counts = {}
    for u2, v2, k2 in cumulative_path:
        edge_key2 = tuple(sorted((u2, v2)))
        traversed_counts[edge_key2] = traversed_counts.get(edge_key2, 0) + 1

    # subtract the original traversal
    for edge_key2 in list(traversed_counts.keys()):
        traversed_counts[edge_key2] = max(0, traversed_counts[edge_key2] - 1)

        # Draw duplicated edges (dashed)
    for edge_key, dup_count in duplicated_edges.items():
        u, v = edge_key
        x1, y1 = int_pos[u]
        x2, y2 = int_pos[v]

        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            continue

        px = -dy / length
        py = dx / length

        traversed_times = traversed_counts.get(edge_key, 0)

        for dup_idx in range(dup_count):
            offset = (dup_idx + 1) * 6
            ox1 = x1 + px * offset
            oy1 = y1 + py * offset
            ox2 = x2 + px * offset
            oy2 = y2 + py * offset

            # BLUE only if this duplicate has been traversed
            color = (100, 149, 237) if dup_idx < traversed_times else (120, 120, 120)

            draw_dashed_line(surface, color, (ox1, oy1), (ox2, oy2), width=2, dash_length=4)


    # Draw cumulative path (blue)
    for u, v, key in cumulative_path:
        x1, y1 = int_pos[u]
        x2, y2 = int_pos[v]
        pygame.draw.line(surface, (100, 149, 237), (x1, y1), (x2, y2), 2)

    # Draw nodes
    for node, (x, y) in pos.items():
        circle_radius = 12
        color = (255, 140, 0) if node in highlight_nodes else (173, 216, 230)
        pygame.draw.circle(surface, color, (int(x), int(y)), circle_radius)
        pygame.draw.circle(surface, (0, 0, 0), (int(x), int(y)), circle_radius, 2)
        node_text = font_small.render(str(node), True, (0, 0, 0))
        surface.blit(node_text, node_text.get_rect(center=(int(x), int(y))))

    # Draw traveler
    if traveler_pos:
        u, v, progress = traveler_pos
        x1, y1 = int_pos[u]
        x2, y2 = int_pos[v]
        traveler_x = x1 + (x2 - x1) * progress
        traveler_y = y1 + (y2 - y1) * progress
        pygame.draw.circle(surface, (255, 0, 0), (int(traveler_x), int(traveler_y)), 8)
        pygame.draw.circle(surface, (255, 255, 255), (int(traveler_x), int(traveler_y)), 8, 2)

    # --- Draw title, step info, and legend/controls text ---
    title_text = font_large.render(title, True, (0, 0, 0))
    surface.blit(title_text, (20, 20))

    step_text = font_small.render(step_info, True, (0, 0, 0))
    surface.blit(step_text, (20, 60))

    # Always draw the controls & legend
    surface.blit(controls_text, (20, HEIGHT - 70))
    surface.blit(legend_original, (20, HEIGHT - 50))
    surface.blit(legend_duplicated, (20, HEIGHT - 30))

def export_video_after_animation():
    import os
    import imageio
    import numpy as np
    import pygame

    folder = os.path.dirname(os.path.abspath(__file__))
    video_file = os.path.join(folder, "cpp_animation.mp4")
    print("Saving video to:", video_file)

    # Hidden surface to draw frames (same size as screen)
    hidden_surface = pygame.Surface((WIDTH, HEIGHT))

    images = []
    for frame in frames:
        draw_graph(
            frame["graph"],
            pos,
            highlight_nodes=frame["highlight_nodes"],
            highlight_edges=frame["highlight_edges"],
            title=frame["title"],
            step_info=frame["step_info"],
            duplicated_edges=frame["duplicated_edges"],
            traveler_pos=frame["traveler_pos"],
            cumulative_path=frame["cumulative_path"],
            surface_override=hidden_surface
        )

        pixels = pygame.surfarray.array3d(hidden_surface)
        pixels = np.transpose(pixels, (1, 0, 2))
        images.append(pixels)

    imageio.mimsave(video_file, images, fps=FPS)
    print("Video saved successfully!")



def run_animation():
    """Run the pygame animation."""
    current_frame = -1  # -1 for startup screen
    paused = True  # Start paused
    animation_started = False

    # Force repeated key detection
    pygame.key.set_repeat(200, 50)  # delay 200ms, interval 50ms
    
    while True:
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return
                if event.key == pygame.K_SPACE:
                    if current_frame == -1:  # Start animation
                        animation_started = True
                        paused = False
                        current_frame = 0
                    else:
                        paused = not paused  # Toggle pause/play
                if event.key == pygame.K_RIGHT and animation_started:
                    current_frame = min(current_frame + 1, len(frames) - 1)
                    paused = True
                if event.key == pygame.K_LEFT and animation_started:
                    current_frame = max(current_frame - 1, 0)
                    paused = True
                if event.key == pygame.K_UP and animation_started:
                    current_frame = min(current_frame + 5, len(frames) - 1)
                    paused = True
                if event.key == pygame.K_DOWN and animation_started:
                    current_frame = max(current_frame - 5, 0)
                    paused = True

        # Additional check: SPACE key outside event loop
        if current_frame == -1 and keys[pygame.K_SPACE]:
            animation_started = True
            paused = False
            current_frame = 0

        # Draw startup screen
        if current_frame == -1:
            screen.fill((240, 240, 240))
            title = font_large.render("Chinese Postman Problem Solver", True, (0, 0, 0))
            screen.blit(title, (WIDTH // 2 - 200, HEIGHT // 2 - 100))
            
            start_text = font_large.render("Press SPACE to Start", True, (0, 100, 200))
            screen.blit(start_text, (WIDTH // 2 - 150, HEIGHT // 2))
            
            info_lines = [
                "Step 1: Load weighted graph",
                "Step 2: Identify odd-degree nodes",
                "Step 3: Minimum weight matching",
                "Step 4: Duplicate edges (make Eulerian)",
                "Step 5: Hierholzer's algorithm (Eulerian circuit)",
                "Step 6: Branch & Bound tour selection"
            ]
            
            for i, line in enumerate(info_lines):
                info_text = font_small.render(line, True, (80, 80, 80))
                screen.blit(info_text, (50, HEIGHT // 2 + 50 + i * 25))
            

            screen.blit(controls_text, (20, HEIGHT - 70))
            screen.blit(legend_original, (20, HEIGHT - 50))
            screen.blit(legend_duplicated, (20, HEIGHT - 30))
            pygame.display.flip()



           
        elif current_frame < len(frames):
            if not paused and current_frame < len(frames) - 1:
                current_frame += 1
            
            frame = frames[current_frame]
            draw_graph(
                frame["graph"],
                pos,
                highlight_nodes=frame["highlight_nodes"],
                highlight_edges=frame["highlight_edges"],
                title=frame["title"],
                step_info=f"Frame {current_frame + 1}/{len(frames)} - {frame['step_info']}",
                duplicated_edges=frame["duplicated_edges"],
                traveler_pos=frame["traveler_pos"],
                cumulative_path=frame["cumulative_path"]
            )
        
        pygame.display.flip()
        clock.tick(FPS)

# =============================================================
# START PYGAME ANIMATION
# =============================================================
if __name__ == "__main__":
    # 1️⃣ Run your animation normally
    run_animation()  
    pygame.quit()  # Quit the animation window

    # 2️⃣ Re-initialize Pygame for video export
    pygame.init()
    pygame.font.init()  # Needed for font rendering
    font_small = pygame.font.Font(None, 16)
    font_large = pygame.font.Font(None, 36)

    # 3️⃣ Create a hidden surface for rendering frames
    hidden_surface = pygame.Surface((WIDTH, HEIGHT))

    # 4️⃣ Export the video
    import os
    import imageio
    import numpy as np

    folder = os.path.dirname(os.path.abspath(__file__))
    video_file = os.path.join(folder, "cpp_animation.mp4")
    print("Saving video to:", video_file)

    images = []
    for frame in frames:
        draw_graph(
            frame["graph"],
            pos,
            highlight_nodes=frame["highlight_nodes"],
            highlight_edges=frame["highlight_edges"],
            title=frame["title"],
            step_info=frame["step_info"],
            duplicated_edges=frame["duplicated_edges"],
            traveler_pos=frame["traveler_pos"],
            cumulative_path=frame["cumulative_path"],
            surface_override=hidden_surface  # Draw on hidden surface
        )

        # Convert surface to numpy array
        pixels = pygame.surfarray.array3d(hidden_surface)
        pixels = np.transpose(pixels, (1, 0, 2))
        images.append(pixels)

    # Save as video
    imageio.mimsave(video_file, images, fps=FPS)
    print("Video saved successfully!")


    # =========================
    # STRUCTURED CSV EXPORT
    # =========================

    csv_file = os.path.join(folder, "cpp_results_v2.csv")

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)

        # ---- TOUR TABLE ----
        writer.writerow(["Step", "From", "To", "Weight"])
        for i, (u, v, key) in enumerate(tour):
            writer.writerow([i + 1, u, v, G[u][v][key]["weight"]])

        writer.writerow([])

        # ---- SUMMARY ----
        writer.writerow(["Total Cost", total_cost])
        writer.writerow(["Edges Traversed", len(tour)])

        writer.writerow([])

        # ---- MDS ----
        writer.writerow(["Minimum Dominating Set"])
        for node in sorted(best_mds):
            writer.writerow([node])

    print("CSV saved to:", csv_file)



    # 5️⃣ Quit Pygame after export
    pygame.quit()
               # quit Pygame after video is saved





