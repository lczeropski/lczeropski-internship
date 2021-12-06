#%%
from typing import List
import time
#%%
class Solution:
    def findMin(self, nums: List[int]) -> int:
        if len(nums)<=1 : return nums[0]
        if nums[-1] > nums [0]:
            return nums[0]
        for i in range(len(nums)):
            if nums[i] > nums[i+1]:
                return nums[i+1]
# %%
s=Solution
# %%
start_time = time.time()
s.findMin(s,[2,3,4,5,1])
print(time.time() - start_time)
# %%
class Solution:
    def findMin(self, nums: List[int]) -> int:
        return min(nums)
# %%
