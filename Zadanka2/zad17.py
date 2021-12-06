#%%
from typing import List


class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        def dfs(grid, i, j): #depth first search
            if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[i]) or grid[i][j] == '0':
                return

            grid[i][j] = '0'
            dfs(grid, i - 1, j)  # up
            dfs(grid, i + 1, j)  # down
            dfs(grid, i, j - 1)  # left
            dfs(grid, i, j + 1)  # right

        res = 0
        for i, iv in enumerate(grid):
            for j, jv in enumerate(grid[i]):
                if grid[i][j] == '1':
                    res += 1
                    dfs(grid, i, j)
        return res
# %%
s=Solution
# %%
s.numIslands(s,grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","1","0"]
])
# %%
s.numIslands(s,grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
])
# %%
