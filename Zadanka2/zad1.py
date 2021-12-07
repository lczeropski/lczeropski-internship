#%%
from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(0,len(nums)):
            for j in range(i+1,len(nums)):
                if (nums[i]+nums[j])==target:
                    return [i,j]
# %%
# %%
a=Solution()
# %%
a.twoSum([3,2,4],6)
# %%
a.twoSum([1,2,3,4,5,6,7,8,9,10,11,12,13,14,16],30)