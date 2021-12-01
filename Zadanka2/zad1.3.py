#%%
class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        self.num = nums
        self.tgt = target
        for i in range(0,len(self.num)-1):
            tgt_list = self.num[i+1:]
            if self.tgt - self.num[i] in tgt_list:
                return[i,tgt_list.index(self.tgt - self.num[i])+i+1]


            
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