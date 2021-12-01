#%%
class Solution:
    def containsDuplicate(self, nums: list[int]) -> bool:
        self.num = nums
        if len(set(self.num)) == len(self.num):
            return False
        return True
# %%
a=Solution
# %%
a.containsDuplicate(a,[1,2,3,1])
# %%
a.containsDuplicate(a,[3,1])
# %%
