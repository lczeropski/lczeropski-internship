#%%
from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(0,len(nums)-1):
            tgt_list = nums[i+1:]
            if target - nums[i] in tgt_list:
                return[i,tgt_list.index(target - nums[i])+i+1]


            
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