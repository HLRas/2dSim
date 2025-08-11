from __future__ import annotations

import os
# Configure SDL for headless environments before importing pygame
if not os.environ.get("DISPLAY") and not os.environ.get("SDL_VIDEODRIVER"):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
# Silence audio errors when running headless
if not os.environ.get("SDL_AUDIODRIVER"):
    os.environ["SDL_AUDIODRIVER"] = "dummy"

import math
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pygame


# ----------------------------
# Configuration
# ----------------------------
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
BACKGROUND_COLOR = (245, 245, 245)
GRID_COLOR = (230, 230, 230)
PATH_COLOR = (0, 120, 215)
ROBOT_COLOR = (40, 40, 40)
ROBOT_HEADING_COLOR = (0, 180, 0)
PARKING_SPACE_COLOR = (220, 240, 255)
PARKING_SPACE_OUTLINE = (100, 140, 200)
TEXT_COLOR = (10, 10, 10)

CELL_SIZE = 20  # pixels per grid cell for pathfinding

# Robot physical params (in pixels and seconds domain)
WHEEL_BASE = 60.0  # distance between wheels (pixels)
ROBOT_RADIUS = 18.0
MAX_LINEAR_SPEED = 160.0  # px/s
MAX_ANGULAR_SPEED = 2.5  # rad/s
LOOKAHEAD_DISTANCE = 60.0  # px for pure pursuit

dt_fixed = 1.0 / 60.0


@dataclass
class ParkingSpace:
    center: Tuple[float, float]
    width: float
    height: float
    orientation_rad: float  # Desired final heading when parked

    def rect(self) -> pygame.Rect:
        x = int(self.center[0] - self.width / 2)
        y = int(self.center[1] - self.height / 2)
        return pygame.Rect(x, y, int(self.width), int(self.height))


class DifferentialDriveRobot:
    def __init__(self, x: float, y: float, heading_rad: float):
        self.x = x
        self.y = y
        self.heading = heading_rad

        self.left_wheel_speed = 0.0  # px/s
        self.right_wheel_speed = 0.0  # px/s

        self.nominal_speed = 120.0  # px/s

    def set_wheel_speeds(self, left_speed: float, right_speed: float) -> None:
        self.left_wheel_speed = float(np.clip(left_speed, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED))
        self.right_wheel_speed = float(np.clip(right_speed, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED))

    def set_velocity(self, linear: float, angular: float) -> None:
        linear = float(np.clip(linear, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED))
        angular = float(np.clip(angular, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))
        # Convert to wheel speeds: v_r = v + (omega * L/2), v_l = v - (omega * L/2)
        right_speed = linear + angular * (WHEEL_BASE / 2.0)
        left_speed = linear - angular * (WHEEL_BASE / 2.0)
        self.set_wheel_speeds(left_speed, right_speed)

    def update(self, dt: float) -> None:
        # Differential drive kinematics
        v = (self.right_wheel_speed + self.left_wheel_speed) / 2.0
        omega = (self.right_wheel_speed - self.left_wheel_speed) / WHEEL_BASE

        self.x += v * math.cos(self.heading) * dt
        self.y += v * math.sin(self.heading) * dt
        self.heading += omega * dt

        # Keep heading bounded for numeric stability
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi

        # Clamp within bounds (simple world walls)
        self.x = float(np.clip(self.x, ROBOT_RADIUS, SCREEN_WIDTH - ROBOT_RADIUS))
        self.y = float(np.clip(self.y, ROBOT_RADIUS, SCREEN_HEIGHT - ROBOT_RADIUS))

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.circle(surface, ROBOT_COLOR, (int(self.x), int(self.y)), int(ROBOT_RADIUS), width=2)
        # Heading line
        hx = self.x + math.cos(self.heading) * ROBOT_RADIUS
        hy = self.y + math.sin(self.heading) * ROBOT_RADIUS
        pygame.draw.line(surface, ROBOT_HEADING_COLOR, (int(self.x), int(self.y)), (int(hx), int(hy)), width=3)


# ----------------------------
# Pathfinding (A*) on a coarse grid
# ----------------------------

def world_to_grid(x: float, y: float) -> Tuple[int, int]:
    return int(x // CELL_SIZE), int(y // CELL_SIZE)


def grid_to_world(i: int, j: int) -> Tuple[float, float]:
    return i * CELL_SIZE + CELL_SIZE / 2.0, j * CELL_SIZE + CELL_SIZE / 2.0


def build_occupancy_grid(spaces: List[ParkingSpace]) -> np.ndarray:
    width_cells = SCREEN_WIDTH // CELL_SIZE
    height_cells = SCREEN_HEIGHT // CELL_SIZE
    grid = np.zeros((width_cells, height_cells), dtype=np.uint8)

    # Mark world boundaries as blocked by leaving outer ring free but robot will be clamped anyway.
    # Optionally mark decorative obstacles: For now, none beyond boundaries.

    # Keep parking spaces free; optionally add vertical separator lines as visuals only.

    return grid


def astar(start: Tuple[int, int], goal: Tuple[int, int], grid: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    width, height = grid.shape

    def in_bounds(n):
        return 0 <= n[0] < width and 0 <= n[1] < height

    def passable(n):
        return grid[n[0], n[1]] == 0

    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    neighbors8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    import heapq

    open_heap = []
    heapq.heappush(open_heap, (0.0, start))
    came_from = {start: None}
    g_score = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            # Reconstruct path
            path = [current]
            while came_from[current] is not None:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dx, dy in neighbors8:
            neighbor = (current[0] + dx, current[1] + dy)
            if not in_bounds(neighbor) or not passable(neighbor):
                continue
            cost = math.hypot(dx, dy)
            tentative_g = g_score[current] + cost
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f, neighbor))

    return None


# ----------------------------
# Pure Pursuit Controller
# ----------------------------

def find_lookahead_point(robot_pos: Tuple[float, float], path_world: List[Tuple[float, float]], lookahead: float) -> Tuple[float, float]:
    rx, ry = robot_pos

    # Walk along path and find first point at distance >= lookahead from current robot position, projected along segments
    # If none found, return the last path point
    # Also include simple segment-circle intersection to get better tracking

    # Start with closest segment search
    for idx in range(len(path_world) - 1):
        p1 = np.array(path_world[idx])
        p2 = np.array(path_world[idx + 1])
        d = p2 - p1
        f = p1 - np.array([rx, ry])

        a = np.dot(d, d)
        b = 2.0 * np.dot(f, d)
        c = np.dot(f, f) - lookahead * lookahead

        discriminant = b * b - 4 * a * c
        if a < 1e-6:
            continue
        if discriminant < 0:
            continue
        discriminant_sqrt = math.sqrt(discriminant)
        t1 = (-b - discriminant_sqrt) / (2 * a)
        t2 = (-b + discriminant_sqrt) / (2 * a)
        for t in (t1, t2):
            if 0.0 <= t <= 1.0:
                point = p1 + t * d
                return float(point[0]), float(point[1])

    # Fallback: farthest point found by distance threshold, else last point
    for pt in path_world[::-1]:
        if math.hypot(pt[0] - rx, pt[1] - ry) >= lookahead * 0.8:
            return pt
    return path_world[-1]


def pure_pursuit_control(robot: DifferentialDriveRobot, target_point: Tuple[float, float]) -> Tuple[float, float]:
    # Compute control to reach target lookahead point
    dx = target_point[0] - robot.x
    dy = target_point[1] - robot.y
    target_heading = math.atan2(dy, dx)
    heading_error = (target_heading - robot.heading + math.pi) % (2 * math.pi) - math.pi

    # Use curvature based on lateral error approximation
    distance = math.hypot(dx, dy)
    if distance < 1e-5:
        return 0.0, 0.0

    # Simple proportional control on heading error; limit angular rate
    linear_speed = robot.nominal_speed

    # Slow down when near target to prevent overshoot
    linear_speed = min(linear_speed, 60.0 + 100.0 * (distance / (LOOKAHEAD_DISTANCE * 2.0)))

    k_heading = 2.5
    angular_speed = float(np.clip(k_heading * heading_error, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))

    return linear_speed, angular_speed


# ----------------------------
# Parking Space generation and selection
# ----------------------------

def generate_parking_spaces() -> List[ParkingSpace]:
    spaces: List[ParkingSpace] = []
    num_spaces = 6
    margin_right = 40
    space_width = 80
    space_height = 120

    start_y = 80
    spacing = space_height + 20

    x_center = SCREEN_WIDTH - margin_right - space_width / 2

    for i in range(num_spaces):
        y_center = start_y + i * spacing
        if y_center + space_height / 2.0 > SCREEN_HEIGHT - 20:
            break
        # Orientation: facing left when parked
        spaces.append(ParkingSpace(center=(x_center, y_center), width=space_width, height=space_height, orientation_rad=math.pi))

    return spaces


def find_closest_space(robot_pos: Tuple[float, float], spaces: List[ParkingSpace]) -> ParkingSpace:
    best_space = min(spaces, key=lambda s: math.hypot(robot_pos[0] - s.center[0], robot_pos[1] - s.center[1]))
    return best_space


# ----------------------------
# Path Utilities
# ----------------------------

def plan_path_to_space(robot_pos: Tuple[float, float], target_space: ParkingSpace, grid: np.ndarray) -> List[Tuple[float, float]]:
    # Plan to a pre-parking waypoint, then to the parking pose center for smoother entry
    approach_offset = 0.45 * target_space.width  # back off from center along orientation
    approach_dx = math.cos(target_space.orientation_rad) * approach_offset
    approach_dy = math.sin(target_space.orientation_rad) * approach_offset

    approach_point = (target_space.center[0] - approach_dx, target_space.center[1] - approach_dy)

    start_ij = world_to_grid(robot_pos[0], robot_pos[1])
    approach_ij = world_to_grid(approach_point[0], approach_point[1])
    goal_ij = world_to_grid(target_space.center[0], target_space.center[1])

    path1 = astar(start_ij, approach_ij, grid)
    if path1 is None:
        # Fallback to direct goal
        path1 = astar(start_ij, goal_ij, grid)
        if path1 is None:
            return [robot_pos, target_space.center]
        path2 = []
    else:
        path2 = astar(approach_ij, goal_ij, grid) or []

    full_grid_path = path1 + (path2[1:] if path2 else [])
    # Convert to world coors
    world_path = [grid_to_world(i, j) for (i, j) in full_grid_path]

    # Prune redundant points with RDP-like simplification
    simplified = simplify_path(world_path, epsilon=4.0)
    return simplified


def simplify_path(points: List[Tuple[float, float]], epsilon: float) -> List[Tuple[float, float]]:
    if len(points) <= 2:
        return points

    def perpendicular_distance(pt, line_start, line_end):
        if line_start == line_end:
            return math.hypot(pt[0] - line_start[0], pt[1] - line_start[1])
        x0, y0 = pt
        x1, y1 = line_start
        x2, y2 = line_end
        num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        den = math.hypot(y2 - y1, x2 - x1)
        return num / den

    # Ramer–Douglas–Peucker
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = perpendicular_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax > epsilon:
        rec1 = simplify_path(points[: index + 1], epsilon)
        rec2 = simplify_path(points[index:], epsilon)
        return rec1[:-1] + rec2
    else:
        return [points[0], points[-1]]


# ----------------------------
# Visualization helpers
# ----------------------------

def draw_grid(surface: pygame.Surface) -> None:
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, SCREEN_HEIGHT), 1)
    for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRID_COLOR, (0, y), (SCREEN_WIDTH, y), 1)


def draw_parking_spaces(surface: pygame.Surface, spaces: List[ParkingSpace]) -> None:
    for sp in spaces:
        rect = sp.rect()
        pygame.draw.rect(surface, PARKING_SPACE_COLOR, rect)
        pygame.draw.rect(surface, PARKING_SPACE_OUTLINE, rect, width=2)
        # Draw orientation arrow inside spot
        arrow_len = min(sp.width, sp.height) * 0.35
        ax = sp.center[0]
        ay = sp.center[1]
        ex = ax + math.cos(sp.orientation_rad) * arrow_len
        ey = ay + math.sin(sp.orientation_rad) * arrow_len
        pygame.draw.line(surface, (120, 160, 220), (int(ax), int(ay)), (int(ex), int(ey)), width=3)


def draw_path(surface: pygame.Surface, path_world: List[Tuple[float, float]]) -> None:
    if len(path_world) < 2:
        return
    pygame.draw.lines(surface, PATH_COLOR, False, [(int(x), int(y)) for (x, y) in path_world], width=3)
    for pt in path_world:
        pygame.draw.circle(surface, PATH_COLOR, (int(pt[0]), int(pt[1])), 3)


# ----------------------------
# Main Simulation
# ----------------------------

def main() -> None:
    pygame.init()
    pygame.display.set_caption("Differential Drive Parking Simulator")

    is_headless = False
    try:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    except pygame.error:
        # Fallback to headless mode (offscreen rendering)
        is_headless = True
        screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

    # Disable mixer in headless to avoid audio device warnings
    if is_headless:
        try:
            pygame.mixer.quit()
        except Exception:
            pass

    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 16)

    # World setup
    robot = DifferentialDriveRobot(x=120.0, y=SCREEN_HEIGHT / 2.0, heading_rad=0.0)
    spaces = generate_parking_spaces()

    # Pathfinding grid
    grid = build_occupancy_grid(spaces)

    # Target selection and path planning
    target_space = find_closest_space((robot.x, robot.y), spaces)
    path_world = plan_path_to_space((robot.x, robot.y), target_space, grid)

    reached_goal = False
    current_lookahead = LOOKAHEAD_DISTANCE

    last_time = time.time()
    accumulator = 0.0
    sim_time = 0.0

    running = True
    while running:
        now = time.time()
        delta = now - last_time
        accumulator += delta
        sim_time += delta
        last_time = now

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    # Reset robot and recompute
                    robot.x, robot.y, robot.heading = 120.0, SCREEN_HEIGHT / 2.0, 0.0
                    reached_goal = False
                    grid = build_occupancy_grid(spaces)
                    target_space = find_closest_space((robot.x, robot.y), spaces)
                    path_world = plan_path_to_space((robot.x, robot.y), target_space, grid)

        # Fixed-step update for stability
        while accumulator >= dt_fixed:
            accumulator -= dt_fixed

            if not reached_goal:
                goal_pos = target_space.center
                distance_to_goal = math.hypot(robot.x - goal_pos[0], robot.y - goal_pos[1])

                # Adapt lookahead when near goal for precision
                current_lookahead = max(30.0, LOOKAHEAD_DISTANCE * (0.5 + min(distance_to_goal / 200.0, 1.0)))

                lookahead_point = find_lookahead_point((robot.x, robot.y), path_world, current_lookahead)
                linear_speed, angular_speed = pure_pursuit_control(robot, lookahead_point)

                # Further reduce speed when very close
                if distance_to_goal < 90:
                    linear_speed *= 0.6
                if distance_to_goal < 50:
                    linear_speed *= 0.5

                robot.set_velocity(linear_speed, angular_speed)
                robot.update(dt_fixed)

                # Check goal with position and orientation
                heading_error_to_spot = (target_space.orientation_rad - robot.heading + math.pi) % (2 * math.pi) - math.pi
                if distance_to_goal < 22.0 and abs(heading_error_to_spot) < math.radians(15):
                    robot.set_velocity(0.0, 0.0)
                    reached_goal = True
            else:
                robot.set_velocity(0.0, 0.0)
                robot.update(dt_fixed)

        # Rendering
        screen.fill(BACKGROUND_COLOR)
        draw_grid(screen)
        draw_parking_spaces(screen, spaces)
        draw_path(screen, path_world)
        robot.draw(screen)

        # Debug HUD
        hud_lines = [
            f"Left wheel: {robot.left_wheel_speed:6.1f} px/s",
            f"Right wheel: {robot.right_wheel_speed:6.1f} px/s",
            f"Pose: ({robot.x:6.1f}, {robot.y:6.1f}) th={math.degrees(robot.heading):6.1f} deg",
            f"Target space: ({target_space.center[0]:.0f}, {target_space.center[1]:.0f})",
            f"Lookahead: {current_lookahead:.1f} px",
            "Status: PARKED" if reached_goal else "Status: DRIVING",
            "Press R to reset, ESC to quit",
        ]
        for i, line in enumerate(hud_lines):
            text = font.render(line, True, TEXT_COLOR)
            screen.blit(text, (10, 10 + i * 18))

        # Update or save frame depending on mode
        if not is_headless:
            pygame.display.flip()
        else:
            # Save a final image and quit if parked or timeout reached
            if reached_goal or sim_time > 25.0:
                try:
                    pygame.image.save(screen, "/workspace/parking_result.png")
                except Exception:
                    pass
                running = False

        clock.tick(60)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()