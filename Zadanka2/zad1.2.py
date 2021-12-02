#%%
class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        return [ [i,j] for i in range(0,len(nums)) for j in range(i,len(nums)) if nums[i]+nums[j]==target and i!=j][0]
# %%
a=Solution()
# %%
a.twoSum([3,2,4],6)
# %%
a.twoSum([1,2,3,4,5,6,7,8,9,10,11,12,13,14,16],30)
# %%
