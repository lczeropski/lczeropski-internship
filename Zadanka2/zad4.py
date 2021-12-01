#%%
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return {i:s.count(i) for i in s} == {i: t.count(i) for i in t}
# %%
a=Solution
# %%
a.isAnagram(a,s = "anagram", t = "nagaram")
# %%
