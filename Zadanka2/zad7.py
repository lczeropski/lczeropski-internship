#%%
from typing import List


class Solution:
    def __init__(self, nums) -> None:
        self.n = nums
    def maxSubArray(self) -> int:
        max_sub = [0 for i in range(len(self.n))]
        max_sub[0] = self.n[0]
        for i in range(1,len(self.n)):
            max_sub[i] = max(max_sub[i-1]+self.n[i],self.n[i])
        return max(max_sub)
# %%
s=Solution([-2,1,-3,4,-1,2,1,-5,4])

# %%
s.maxSubArray()
# %%
