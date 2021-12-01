#%%
import numpy as np
#%%
class Solution:
    def productExceptSelf(self, nums: list[int]) -> list[int]:
        return [np.prod(nums[:i] + nums[i+1:]) for i in range(0,len(nums)) ]
 
# %%
s=Solution()
# %%
s.productExceptSelf([1,2,3,4])
# %%
l=[1,2,3,4]
# %%
class Solution:
    def productExceptSelf(self, nums):
        t1 = [1] + list(accumulate(nums, mul))[:-1]
        t2 = list(accumulate(nums[::-1], mul))[::-1][1:] + [1]
        return [x*y for x, y in zip(t1, t2)]