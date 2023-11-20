from collections import defaultdict
import copy
import math
import numpy as np

class TrajectoryFilter(object):
    def __init__(self, wtime=4, wpixel=60, wcomp=5):
        self.wtime = wtime
        self.wpixel = wpixel
        self.wcomp = wcomp
        
    def create_graph(self, trajectory):
        graph = defaultdict(list)
        points = list(zip(trajectory.X, trajectory.Y))
        
        for i in range(len(points)):
            # Search back and forth Lt steps and find points within eps
            # Only need to iterate on backwards range since we add the edges bidirectionally
            for k in range(max(0, i-self.wtime), i):
                diff = np.array(points[i]) - np.array(points[k])
                if np.all(np.abs(diff) < self.wpixel):
                    graph[i].append(k)
                    graph[k].append(i)
        return graph            
    
    def find_components(self, graph):
        # Now find the components of the graph
        seen = {}
        def dfs(v, num):
            stack = [v]
            seen[v] = num
            comp = [v]
            while len(stack):
                v = stack.pop()
                if v not in graph:
                    continue
                for w in graph[v]:
                    if w not in seen:
                        seen[w] = num
                        stack.append(w)
                        comp.append(w)
            return comp

        compid = 0
        for v in graph:
            if v not in seen:
                comp = dfs(v, compid)
                compid += 1

        components = defaultdict(list)
        for v in graph:
            components[seen[v]].append(v)
        return components
    
    def filter_trajectory(self, trajectory):
        graph = self.create_graph(trajectory)
        components = self.find_components(graph)
        
        sorted_comps = []
        for compv in components.values():
            if len(compv) >= self.wcomp:
                sorted_comps.append((-len(compv), compv))
        sorted_comps.sort()
        
        # Iterate through components in decreasing weight, doing linear interpolation / collision checking with other comps
        L = len(trajectory.X)
        done = np.array([0] * L)
        filtered = []
        for score, comp in sorted_comps:
            m, M = min(x for x in comp), max(x for x in comp)
            if any(done[m:M+1]):
                continue
            done[m:M] = 1
            
            comp = sorted(comp)
            for t in comp:
                filtered.append((t, (trajectory.X[t], trajectory.Y[t])))
        
        new_traj = copy.deepcopy(trajectory)

        # new_traj.X, new_traj.Y = [math.nan] * L, [math.nan] * L
        
        cnt = 0
        for i, p in sorted(filtered):
            while cnt < i:
                cnt += 1
            new_traj.X[i] = p[0]
            new_traj.Y[i] = p[1]
            cnt += 1
        
        
        return new_traj