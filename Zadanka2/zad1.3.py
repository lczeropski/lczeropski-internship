#%%
from typing import List


class Solution:
    def __init__(self, nums):
        self.n = nums
    def twoSum(self, target: int) -> List[int]:
        for i in range(0,len(self.n)-1):
            tgt_list = self.n[i+1:]
            if target - self.n[i] in tgt_list:
                return[i,tgt_list.index(target - self.n[i])+i+1]


            
# %%
a=Solution()
# %%
a.twoSum([3,2,4],6)
# %%
a.twoSum([1,2,3,4,5,6,7,8,9,10,11,12,13,14,16],30)
# %%
a.twoSum([3,3],6)
# %%
# %%