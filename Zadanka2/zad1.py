#%%
from typing import List


class Solution:
    def __init__(self, nums):
        self.n = nums
    def twoSum(self, target):
        for i in range(0,len(self.n)):
            for j in range(i+1,len(self.n)):
                if (self.n[i]+self.n[j])==target:
                    return [i,j]
# %%
# %%
a=Solution([3,2,4])
b=Solution([1,2,3,4,5,6,7,8,9,10,11,12,13,14,16])
# %%
a.twoSum(6)
# %%
b.twoSum(30)
# %%
