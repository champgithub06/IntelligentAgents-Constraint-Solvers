import random
from collections import defaultdict, deque
import copy

class B22AI039:
    

    def __init__(self, initial_state):
        self.DEBUG = False

        if self.DEBUG: print("Definitive Agent (MAC + Global Explore) initialized.")

        # Core State
        self.graph = defaultdict(set)
        self.assignments = {} # Agent's own decisions
        self.pre_colored_nodes = {}
        self.all_nodes = set()
        self.colors = list(initial_state['available_colors'])
        self.domains = {}
        self.decision_path = []
        
        self.backtracking_mode = False
        self.path_to_target = []
        self.new_graph_info_discovered = False

        self._peek_cache = None

        self._update_memory(initial_state)

    def _update_memory(self, visible_state):
        visible_nodes = set(visible_state['visible_graph']['nodes'])
        new_nodes = visible_nodes - self.all_nodes
        for node in new_nodes:
            self.all_nodes.add(node)
            observed_color = visible_state['node_colors'].get(node)
            if observed_color is not None:
                self.domains[node] = {observed_color}
                self.pre_colored_nodes[node] = observed_color
                self.assignments[node] = observed_color
            else:
                self.domains[node] = set(self.colors)

        for u, v in visible_state['visible_graph']['edges']:
            if v not in self.graph[u]:
                self.graph[u].add(v)
                self.graph[v].add(u)
                self.new_graph_info_discovered = True

    def _revise(self, xi, xj, domains):
        revised = False
        for x_color in list(domains.get(xi, set())):
            if not any(x_color != y_color for y_color in domains.get(xj, set())):
                domains[xi].remove(x_color)
                revised = True
        return revised

    def _arc_consistency(self, domains_to_check, source_node=None):
        domains = copy.deepcopy(domains_to_check)
        queue = deque()
        unassigned_nodes = {n for n in self.all_nodes if n not in self.assignments}
        
        if source_node:
            for neighbor in self.graph.get(source_node, []):
                if neighbor in unassigned_nodes: queue.append((neighbor, source_node))
        else:
            for node in unassigned_nodes:
                for neighbor in self.graph.get(node, []):
                    if neighbor in unassigned_nodes: queue.append((node, neighbor))
        while queue:
            xi, xj = queue.popleft()
            if self._revise(xi, xj, domains):
                if not domains.get(xi): return False, domains
                for xk in self.graph.get(xi, []):
                    if xk != xj and xk in unassigned_nodes: queue.append((xk, xi))
        return True, domains

    def _find_path(self, start_node, end_node):
        if start_node == end_node: return []
        queue = deque([[start_node]])
        visited = {start_node}
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == end_node: return path[1:]
            for neighbor in sorted(list(self.graph.get(node, []))):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
        return None

    def _peek_best_color(self, node_to_color):
        neighbor_colors = {self.assignments[n] for n in self.graph.get(node_to_color, []) if n in self.assignments}
        node_domain = self.domains.get(node_to_color, set(self.colors))
        legal_colors = sorted(list(node_domain - neighbor_colors))
        
        if not legal_colors:
            return None, None

        color_damages = []
        for color in legal_colors:
            temp_domains = copy.deepcopy(self.domains)
            temp_domains[node_to_color] = {color}
            is_consistent, pruned = self._arc_consistency(temp_domains, source_node=node_to_color)
            if not is_consistent: continue
            damage = sum(max(0, len(self.domains.get(n, set())) - len(pruned.get(n, set()))) for n in self.all_nodes if n not in self.assignments)
            color_damages.append((damage, color, pruned))

        if not color_damages:
            return None, None

        color_damages.sort(key=lambda x: x[0])
        _, best_color, best_pruned = color_damages[0]
        return best_color, best_pruned

    def get_next_move(self, visible_state):
        self._update_memory(visible_state)
        current_node = visible_state['current_node']

        if self.path_to_target:
            next_step = self.path_to_target.pop(0)
            return {'action': 'move', 'node': next_step}

        # If new information discovered, run AC and update domains
        if self.new_graph_info_discovered:
            consistent, pruned = self._arc_consistency(self.domains)
            if consistent: self.domains = pruned
            else: self.backtracking_mode = True
            self.new_graph_info_discovered = False

        # Backtracking handling remains prioritized
        if self.backtracking_mode:
            if not self.decision_path: return {'action': 'move', 'node': current_node}
            last_node, _, old_domains = self.decision_path.pop()
            self.domains = old_domains
            self.domains[last_node].discard(self.assignments.get(last_node))
            self.assignments.clear()
            self.assignments.update(self.pre_colored_nodes)
            for (node, color, _) in self.decision_path: self.assignments[node] = color
            if self.domains[last_node]:
                self.backtracking_mode = False
                if last_node in visible_state['visible_graph']['nodes']:
                    return {'action': 'move', 'node': last_node}
                else:
                    path = self._find_path(current_node, last_node)
                    if path:
                        self.path_to_target = path
                        return {'action': 'move', 'node': self.path_to_target.pop(0)}
            else:
                self.backtracking_mode = True
                return {'action': 'move', 'node': current_node}


        if current_node not in self.assignments:
            best_color, best_pruned = self._peek_best_color(current_node)
            if best_color is not None:
        
                domains_snapshot = copy.deepcopy(self.domains)
                self._peek_cache = (current_node, best_color, best_pruned, domains_snapshot)
  
                return {'action': 'move', 'node': current_node}

        # Continue with exploration / movement selection
        uncolored_visible = [n for n in visible_state['visible_graph']['nodes'] if n not in self.assignments]
        if uncolored_visible:
            uncolored_visible.sort()
            best_node = min(uncolored_visible, key=lambda n: (len(self.domains.get(n, self.colors)), -len(self.graph.get(n, []))))
            return {'action': 'move', 'node': best_node}
        
        uncolored_global = [n for n in self.all_nodes if n not in self.assignments]
        if uncolored_global:
            uncolored_global.sort() 
            # Find the best uncolored node anywhere in the known world
            global_target = min(uncolored_global, key=lambda n: (len(self.domains.get(n, self.colors)), -len(self.graph.get(n, []))))
            path = self._find_path(current_node, global_target)
            if path:
                self.path_to_target = path
                return {'action': 'move', 'node': self.path_to_target.pop(0)}

        adjacent_nodes = [n for n in visible_state['visible_graph']['nodes'] if n != current_node]
        if adjacent_nodes:
            adjacent_nodes.sort()
            best_explore_node = max(adjacent_nodes, key=lambda n: len(self.graph.get(n, set()) - self.all_nodes))
            return {'action': 'move', 'node': best_explore_node}
            
        return {'action': 'move', 'node': current_node}

    def get_color_for_node(self, node_to_color, visible_state):
        # recomputing or calling _update_memory again.
        if self._peek_cache and self._peek_cache[0] == node_to_color:
            _, best_color, best_pruned, domains_snapshot = self._peek_cache
            self.decision_path.append((node_to_color, best_color, domains_snapshot))
            self.domains = best_pruned
            self.assignments[node_to_color] = best_color
            self._peek_cache = None
            return {'action': 'color', 'node': node_to_color, 'color': best_color}

        # Otherwise proceed with the original, safe destructive path
        self._update_memory(visible_state)
        
        neighbor_colors = {self.assignments[n] for n in self.graph.get(node_to_color, []) if n in self.assignments}
        node_domain = self.domains.get(node_to_color, set(self.colors))
        legal_colors = sorted(list(node_domain - neighbor_colors))
        
        if not legal_colors:
            self.backtracking_mode = True
            return {'action': 'color', 'node': node_to_color, 'color': self.colors[0]}

        color_damages = []
        for color in legal_colors:
            temp_domains = copy.deepcopy(self.domains)
            temp_domains[node_to_color] = {color}
            is_consistent, pruned = self._arc_consistency(temp_domains, source_node=node_to_color)
            if not is_consistent: continue
            damage = sum(max(0, len(self.domains.get(n, set())) - len(pruned.get(n, set()))) for n in self.all_nodes if n not in self.assignments)
            color_damages.append((damage, color, pruned))

        if not color_damages:
            self.backtracking_mode = True
            return {'action': 'color', 'node': node_to_color, 'color': self.colors[0]}

        color_damages.sort(key=lambda x: x[0])
        best_damage, best_color, best_pruned = color_damages[0]

        domains_snapshot = copy.deepcopy(self.domains)
        self.decision_path.append((node_to_color, best_color, domains_snapshot))
        
        self.domains = best_pruned
        self.assignments[node_to_color] = best_color
        
        return {'action': 'color', 'node': node_to_color, 'color': best_color}
