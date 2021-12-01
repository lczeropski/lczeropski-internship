#%%
class Solution:
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        if len(nums)< 3:
            return []
        sol = set()
        nums = sorted(nums)
        
        for i in range(len(nums)-2):
            j = i+1
            k = len(nums)-1
            while j < k:
                while nums[i] + nums[j] + nums[k] > 0:
                    if i == j or j == k or i == k: break
                    k -= 1 
                while nums[i] + nums[j] + nums[k] < 0:
                    if i == j or j == k or i == k: break
                    j += 1 
                if nums[i] + nums[j] + nums[k] == 0 and i != j != k:
                    sol.add(tuple(sorted([nums[i], nums[j], nums[k]])))
                    j += 1
        
        return list(sol)
# %%
s=Solution
# %%
s.threeSum(a,[-1,0,1,2,-1,-4])
# %%
int(len([-1,0,1,2,-1,-4])/3)
# %%
