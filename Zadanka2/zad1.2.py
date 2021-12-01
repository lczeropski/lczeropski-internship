#%%
class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        self.num = nums
        self.tgt = target
        return [ [i,j] for i in range(0,len(self.num)) for j in range(i,len(self.num)) if self.num[i]+self.num[j]==self.tgt and i!=j][0]
# %%
a=Solution()
# %%
a.twoSum([3,2,4],6)
# %%
a.twoSum([1,2,3,4,5,6,7,8,9,10,11,12,13,14,16],30)