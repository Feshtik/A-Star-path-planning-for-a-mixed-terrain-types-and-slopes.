import numpy as np 
import heapq
import matplotlib.pyplot as plt

# Constants
GRID_SIZE = 100
TARGET = (50, 50)
START = (0, 0)
VELOCITY_BASE = 5  # cells/minute
TERRAIN_SPEED = {"sand": 0.7, "mud": 0.4, "asphalt": 1.5}  # Multipliers
SLOPE_LIMIT = 0.2

# Terrain values
terrain_types = {0: "sand", 1: "mud", 2: "asphalt"}

# Create the grid map
def generate_grid(grid_size):
    grid = np.zeros((grid_size, grid_size), dtype=[
        ("coordinates", "2i4"),
        ("terrain", "i4"),
        ("height", "f4")
    ])
    
    for x in range(grid_size):
        for y in range(grid_size):
            terrain = np.random.choice([0, 1, 2])
            # Introduce random variations to the height to allow both increases and decreases
            height = np.random.uniform(-SLOPE_LIMIT, SLOPE_LIMIT) * (x + y) / (grid_size * 2)
            grid[x, y] = ((x, y), terrain, height)
    
    return grid

# Calculate cost for moving from one cell to another
def calculate_cost(current, neighbor, grid):
    x1, y1 = current["coordinates"]
    x2, y2 = neighbor["coordinates"]
    
    terrain_cost = 1 / TERRAIN_SPEED[terrain_types[neighbor["terrain"]]]
    height_diff = neighbor["height"] - current["height"]  # Allow both increase and decrease in slope
    slope_cost = 1 + abs(height_diff) / SLOPE_LIMIT  # Use absolute value for slope cost
    
    distance_cost = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    return distance_cost * terrain_cost * slope_cost

# A* Pathfinding
def find_fastest_path(grid, start, target):
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    cost_so_far = {start: 0}
    
    while open_set:
        current_priority, current = heapq.heappop(open_set)
        
        if current == target:
            break
        
        x, y = current
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                neighbor = grid[nx, ny]
                new_cost = cost_so_far[current] + calculate_cost(grid[x, y], neighbor, grid)
                
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    priority = new_cost + np.sqrt((TARGET[0] - nx)**2 + (TARGET[1] - ny)**2)
                    heapq.heappush(open_set, (priority, (nx, ny)))
                    came_from[(nx, ny)] = current
    
    return came_from, cost_so_far

# Reconstruct the path
def reconstruct_path(came_from, start, target):
    current = target
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

# Visualization
def visualize(grid, path):
    # Extract data from the grid
    terrain_map = np.array([[cell["terrain"] for cell in row] for row in grid])
    height_map = np.array([[cell["height"] for cell in row] for row in grid])
    slope_map = np.gradient(height_map)
    slope_magnitude = np.sqrt(slope_map[0]**2 + slope_map[1]**2)  # Overall slope

    # Create a custom colormap for terrain types
    from matplotlib.colors import ListedColormap

    # Define custom colors for terrain types
    terrain_colors = ["yellow", "darkgoldenrod", "black"]  # Sand, Mud, Asphalt
    cmap = ListedColormap(terrain_colors)

    # Create the plot
    plt.figure(figsize=(12, 10))
    plt.title("Grid Map with Terrain, Height, and Path", fontsize=14)

    # Display terrain types using the custom colormap
    plt.imshow(terrain_map, cmap=cmap, alpha=0.8)

    # Add height contours
    contours = plt.contour(height_map, levels=15, colors='black', linewidths=0.5)
    plt.clabel(contours, inline=True, fontsize=8, fmt="%.2f")  # Label contours with height

    # Overlay the robot's path
    path_x, path_y = zip(*path)
    plt.plot(path_y, path_x, color="red", linewidth=2, label="Robot's Path")

    # Customize legend for terrain types
    terrain_labels = {0: "Sand", 1: "Mud", 2: "Asphalt"}
    legend_handles = [plt.Line2D([0], [0], marker='s', color=color, markersize=10,
                                 label=label, linestyle='None')
                      for label, color in zip(terrain_labels.values(), terrain_colors)]
    plt.legend(handles=legend_handles + [plt.Line2D([], [], color="red", linewidth=2, label="Robot's Path")],
               loc="upper right", fontsize=10)

    # Add labels
    plt.xlabel("X-axis (Grid Columns)")
    plt.ylabel("Y-axis (Grid Rows)")

    # Show the plot
    plt.tight_layout()
    plt.show()

# Main
if __name__ == "__main__":
    grid = generate_grid(GRID_SIZE)
    came_from, cost_so_far = find_fastest_path(grid, START, TARGET)
    path = reconstruct_path(came_from, START, TARGET)
    visualize(grid, path)
