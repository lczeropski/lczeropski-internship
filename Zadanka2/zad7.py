#%%
#%%
class Solution:
    def maxSubArray(self, nums: list[int]) -> int:
        max_sub = [0 for i in range(len(nums))]
        max_sub[0] = nums[0]
        for i in range(1,len(nums)):
            max_sub[i] = max(max_sub[i-1]+nums[i],nums[i])
        return max(max_sub)
# %%
s=Solution

# %%
s.maxSubArray(s,[-2,1,-3,4,-1,2,1,-5,4])
# %%
