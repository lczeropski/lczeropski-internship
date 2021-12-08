#%%
from typing import List


class Solution:
    def __init__(self, nums):
        self.n = nums
    def containsDuplicate(self) -> bool:
        if len(set(self.n)) == len(self.n):
            return False
        return True
# %%
a=Solution([1,2,3,1])
b=Solution([3,1])
# %%
a.containsDuplicate()
# %%
b.containsDuplicate()
# %%
