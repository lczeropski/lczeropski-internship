#%%
class Solution:
    def __init__(self, string) -> None:
        self.s = string
    def countSubstrings(self) -> int:
        res = n = len(self.s) 
        for i in range(n):
            r = l = i
            while r + 1 < n and self.s[l] == self.s[r+1]:
                r += 1
                res += 1
            while r + 1 < n and l > 0 and self.s[l-1] == self.s[r+1]:
                r += 1
                l -= 1
                res += 1
        return res
# %%
