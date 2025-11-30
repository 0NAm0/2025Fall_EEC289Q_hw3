import math
import random
import time

# Load Data
# Graph A
def load_euclidean_graph(path):
    with open(path, "r") as f:
        n = int(f.readline().strip())  # number of nodes
        header = f.readline()          # skip header

        dist = [[0.0] * n for _ in range(n)]

        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1  # 1-based -> 0-based
            j = int(parts[1]) - 1
            d = float(parts[2])
            dist[i][j] = d
            dist[j][i] = d

    return dist


# Graph B
def load_random_graph(path):
    with open(path, "r") as f:
        n = int(f.readline().strip())
        header = f.readline()  # skip header

        dist = [[0.0] * n for _ in range(n)]

        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            j = int(parts[1]) - 1
            d = float(parts[2])
            dist[i][j] = d
            dist[j][i] = d

    return dist


# TSP 
def tour_cost(tour, dist):
    n = len(tour)
    cost = 0.0
    for k in range(n):
        i = tour[k]
        j = tour[(k + 1) % n]
        cost += dist[i][j]
    return cost


def nearest_neighbor(start, dist):
    n = len(dist)
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start

    while unvisited:
        nxt = min(unvisited, key=lambda u: dist[current][u])
        unvisited.remove(nxt)
        tour.append(nxt)
        current = nxt

    return tour


def two_opt_local_search(tour, current_cost, dist, deadline_time):
    n = len(tour)
    cycles_eval = 0
    improved = True

    while improved and time.time() < deadline_time:
        improved = False
        for i in range(1, n - 2):
            if time.time() >= deadline_time:
                break
            for j in range(i + 1, n - 1):
                if time.time() >= deadline_time:
                    break

                new_tour = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                new_cost = tour_cost(new_tour, dist)
                cycles_eval += 1

                if new_cost < current_cost:
                    tour = new_tour
                    current_cost = new_cost
                    improved = True
                    break
            if improved:
                break

    return tour, current_cost, cycles_eval


def tsp_heuristic(dist, time_limit=55.0, seed=0):
    random.seed(seed)
    start_time = time.time()
    deadline_time = start_time + time_limit

    n = len(dist)
    best_tour = None
    best_cost = float("inf")
    total_cycles = 0

    while time.time() < deadline_time:
        start_node = random.randrange(n)
        tour = nearest_neighbor(start_node, dist)
        cost = tour_cost(tour, dist)
        total_cycles += 1

        tour, cost, extra = two_opt_local_search(tour, cost, dist, deadline_time)
        total_cycles += extra

        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    return best_tour, best_cost, total_cycles


# Save solution (for text file)
def save_solution(tour, student_id, graph_name):
    # convert to 1-based indices for output
    tour_1_based = [v + 1 for v in tour]

    filename = f"solution_{student_id}_{graph_name}.txt"
    with open(filename, "w") as f:
        f.write(", ".join(map(str, tour_1_based)))
    print(f"[Saved] Best tour for Graph {graph_name} -> {filename}")



# Main execution for both graphs
def run_for_graphs():
    TIME_LIMIT = 55.0

    file_A = "TSP_1000_euclidianDistance.txt"
    file_B = "TSP_1000_randomDistance.txt"

    print("Loading Graph A (euclidean edge list)...")
    dist_A = load_euclidean_graph(file_A)

    print("Loading Graph B (random edge list)...")
    dist_B = load_random_graph(file_B)

    print("\nRunning heuristic on Graph A...")
    tour_A, cost_A, cycles_A = tsp_heuristic(dist_A, TIME_LIMIT, seed=42)
    print(f"Graph A best tour cost: {cost_A:.2f}")
    print(f"Graph A cycles evaluated: {cycles_A:.1e}")

    print("\nRunning heuristic on Graph B...")
    tour_B, cost_B, cycles_B = tsp_heuristic(dist_B, TIME_LIMIT, seed=43)
    print(f"Graph B best tour cost: {cost_B:.2f}")
    print(f"Graph B cycles evaluated: {cycles_B:.1e}")


    student_id = "924238136" 
    save_solution(tour_A, student_id, "A")
    save_solution(tour_B, student_id, "B")

    return (tour_A, cost_A, cycles_A), (tour_B, cost_B, cycles_B)


if __name__ == "__main__":
    run_for_graphs()
