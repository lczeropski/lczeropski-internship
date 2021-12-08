#%%
from typing import List


class Solution:
    def __init__(self,grid) -> None:
        self.g = grid
    def numIslands(self) -> int:
        def dfs(grid, i, j): #depth first search
            if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[i]) or grid[i][j] == '0':
                return

            grid[i][j] = '0'
            dfs(self.g, i - 1, j)  # up
            dfs(self.g, i + 1, j)  # down
            dfs(self.g, i, j - 1)  # left
            dfs(self.g, i, j + 1)  # right

        res = 0
        for i, iv in enumerate(self.g):
            for j, jv in enumerate(self.g[i]):
                if self.g[i][j] == '1':
                    res += 1
                    dfs(self.g, i, j)
        return res
# %%
s=Solution([
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","1","0"]
])
ss=Solution([
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
])
# %%
s.numIslands()
# %%
ss.numIslands()
# %%
#TODO:DFS
#       A
#      /  \
#     B    C
#    / \     \
#   D   E --> F
#The Algorithm
#Pick any node. If it is unvisited, mark it as visited and recur on all its adjacent nodes.
#Repeat until all the nodes are visited, or the node to be searched is found.
graph = {
    'A' : ['B','C'],
    'B' : ['D', 'E'],
    'C' : ['F'],
    'D' : [],
    'E' : ['F'],
    'F' : []
}
def dfs(visited, graph, node):
    if node not in visited:
        print (node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)

#%%
visited = set() # Set to keep track of visited nodes.
dfs(visited ,graph, 'A')
# %%
