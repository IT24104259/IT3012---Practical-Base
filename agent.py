import math
import random
from collections import deque
import heapq
import itertools
 
FACINGS = ['Up', 'Right', 'Down', 'Left']  # clockwise order, shared with visual_grid_game.py
 
 
class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""
 
    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']
 
    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)
 
 
class SearchAgent:
    """Uninformed + informed search agent: plans a full route to the nearest
    food using BFS, DFS, UCS, or A*, over the (position, facing) state space
    for BFS/DFS/UCS, and over plain grid positions for A* (converted back to
    turn/move actions afterwards)."""
 
    def __init__(self, active_algo='BFS'):
        self.plan = []
        self.active_algo = active_algo
 
        # The percept never exposes the agent's true global position, so we
        # track a believed position/facing ourselves (Transition Model).
        self.believed_pos = [0, 0]
        self.facing_estimate = 'Up'
        self.last_action = None
        self.last_move_blocked = False
 
    #shared helpers
    def _delta(self, facing):
        return {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}[facing]
 
    def _update_believed_state(self):
        if self.last_action == 'move_forward' and not self.last_move_blocked:
            dx, dy = self._delta(self.facing_estimate)
            self.believed_pos[0] += dx
            self.believed_pos[1] += dy
        elif self.last_action == 'turn_left':
            idx = FACINGS.index(self.facing_estimate)
            self.facing_estimate = FACINGS[(idx - 1) % 4]
        elif self.last_action == 'turn_right':
            idx = FACINGS.index(self.facing_estimate)
            self.facing_estimate = FACINGS[(idx + 1) % 4]
 
    def _closest_food(self, all_food):
        if not all_food:
            return None
        bx, by = self.believed_pos
        return min(all_food, key=lambda f: abs(f[0] - bx) + abs(f[1] - by))
 
    # BFS / DFS / UCS state space (position + facing)
    def _successors(self, state, grid_size, walls):
        pos, facing = state
        width, height = grid_size
        idx = FACINGS.index(facing)
 
        results = [
            ((pos, FACINGS[(idx - 1) % 4]), 'turn_left', 1),
            ((pos, FACINGS[(idx + 1) % 4]), 'turn_right', 1),
        ]
 
        dx, dy = self._delta(facing)
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls:
            results.append((((nx, ny), facing), 'move_forward', 1))
 
        return results
 
    def _reconstruct_path(self, came_from, goal_state):
        actions = []
        state = goal_state
        while came_from[state] is not None:
            prev_state, action = came_from[state]
            actions.append(action)
            state = prev_state
        actions.reverse()
        return actions
 
    def bfs_search(self, start, goal_pos, grid_size, walls):
        frontier = deque([start])
        came_from = {start: None}
        reached = {start}
 
        while frontier:
            state = frontier.popleft()
            if state[0] == goal_pos:
                return self._reconstruct_path(came_from, state)
            for nxt_state, action, _cost in self._successors(state, grid_size, walls):
                if nxt_state not in reached:
                    reached.add(nxt_state)
                    came_from[nxt_state] = (state, action)
                    frontier.append(nxt_state)
        return []
 
    def dfs_search(self, start, goal_pos, grid_size, walls):
        frontier = [start]
        came_from = {start: None}
        reached = {start}
 
        while frontier:
            state = frontier.pop()
            if state[0] == goal_pos:
                return self._reconstruct_path(came_from, state)
            for nxt_state, action, _cost in self._successors(state, grid_size, walls):
                if nxt_state not in reached:
                    reached.add(nxt_state)
                    came_from[nxt_state] = (state, action)
                    frontier.append(nxt_state)
        return []
 
    def ucs_search(self, start, goal_pos, grid_size, walls):
        counter = itertools.count()
        frontier = [(0, next(counter), start)]
        came_from = {start: None}
        best_cost = {start: 0}
        reached = set()
 
        while frontier:
            cost, _, state = heapq.heappop(frontier)
            if state in reached:
                continue
            reached.add(state)
 
            if state[0] == goal_pos:
                return self._reconstruct_path(came_from, state)
 
            for nxt_state, action, step_cost in self._successors(state, grid_size, walls):
                new_cost = cost + step_cost
                if nxt_state not in best_cost or new_cost < best_cost[nxt_state]:
                    best_cost[nxt_state] = new_cost
                    came_from[nxt_state] = (state, action)
                    heapq.heappush(frontier, (new_cost, next(counter), nxt_state))
        return []
 
    # A* (heuristics + plain-position search) 
    def manhattan_distance(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
 
    def euclidean_distance(self, pos, goal):
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)
 
    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'):
        heuristic = self.manhattan_distance if heuristic_type == 'manhattan' else self.euclidean_distance
 
        frontier = []
        g0 = 0
        h0 = heuristic(start_pos, goal_pos)
        heapq.heappush(frontier, (g0 + h0, g0, start_pos, [start_pos]))
 
        reached_states = set()
        width, height = grid_size
 
        while frontier:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(frontier)
 
            if current_pos == goal_pos:
                return path_taken
 
            if current_pos in reached_states:
                continue
            reached_states.add(current_pos)
 
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current_pos[0] + dx, current_pos[1] + dy
                neighbor = (nx, ny)
 
                if (0 <= nx < width and 0 <= ny < height
                        and neighbor not in walls
                        and neighbor not in reached_states):
                    g_new = g_cost + 1
                    h_new = heuristic(neighbor, goal_pos)
                    f_new = g_new + h_new
                    heapq.heappush(frontier, (f_new, g_new, neighbor, path_taken + [neighbor]))
 
        return []
 
    def _positions_to_actions(self, path, start_facing):
        """Converts a list of grid positions (from astar_search) into the
        turn_left / turn_right / move_forward commands the environment
        actually executes."""
        deltas_to_facing = {(0, 1): 'Up', (0, -1): 'Down', (-1, 0): 'Left', (1, 0): 'Right'}
        actions = []
        facing = start_facing
 
        for i in range(len(path) - 1):
            cur = path[i]
            nxt = path[i + 1]
            dx, dy = nxt[0] - cur[0], nxt[1] - cur[1]
            target_facing = deltas_to_facing[(dx, dy)]
 
            while facing != target_facing:
                idx_cur = FACINGS.index(facing)
                idx_target = FACINGS.index(target_facing)
                diff = (idx_target - idx_cur) % 4
                if diff == 3:
                    actions.append('turn_left')
                    facing = FACINGS[(idx_cur - 1) % 4]
                else:
                    actions.append('turn_right')
                    facing = FACINGS[(idx_cur + 1) % 4]
 
            actions.append('move_forward')
 
        return actions
 
    # plan selection / execution
    def sense_and_act(self, percept: dict) -> str:
        self._update_believed_state()
 
        if not self.plan:
            grid_size = percept['grid_size']
            walls = set(tuple(w) for w in percept['walls'])
            all_food = [tuple(f) for f in percept['all_food']]
            goal = self._closest_food(all_food)
 
            if goal is None:
                self.last_action = 'turn_left'
                return self.last_action
 
            start = (tuple(self.believed_pos), self.facing_estimate)
 
            if self.active_algo == 'BFS':
                self.plan = self.bfs_search(start, goal, grid_size, walls)
            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(start, goal, grid_size, walls)
            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(start, goal, grid_size, walls)
            elif self.active_algo == 'AStar':
                path = self.astar_search(tuple(self.believed_pos), goal, walls, grid_size,
                                          heuristic_type='manhattan')
                self.plan = self._positions_to_actions(path, self.facing_estimate)
            else:
                self.plan = self.bfs_search(start, goal, grid_size, walls)
 
            if not self.plan:
                self.last_action = 'turn_left'
                return self.last_action
 
        action = self.plan.pop(0)
        self.last_move_blocked = (action == 'move_forward' and percept['wall_ahead'])
        self.last_action = action
        return action
    
    if __name__ == "__main__":
       sa = SearchAgent()
       print("Manhattan:", sa.manhattan_distance((0, 0), (3, 4)))
       print("Euclidean:", sa.euclidean_distance((0, 0), (3, 4)))
