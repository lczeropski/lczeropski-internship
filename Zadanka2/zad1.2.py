#%%
from typing import List


class Solution:
    def __init__(self, nums):
        self.n = nums
    def twoSum(self, target: int) -> List[int]:
        return [ [i,j] for i in range(0,len(self.n)) for j in range(i,len(self.n)) if self.n[i]+self.n[j]==target and i!=j][0]

# %%
a=Solution([3,2,4])
b=Solution([1,2,3,4,5,6,7,8,9,10,11,12,13,14,16])
# %%
a.twoSum(6)
# %%
b.twoSum(30)
# %%
