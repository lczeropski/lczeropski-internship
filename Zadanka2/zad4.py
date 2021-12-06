#%%
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return {i:s.count(i) for i in s} == {i: t.count(i) for i in t}
# %%
a=Solution
# %%
a.isAnagram(a,s = "anagram", t = "nagaram")
# %%
class Solution: #szybsze rozwiazanie
    def isAnagram(self, s: str, t: str) -> bool:
        a, b = len(s), len(t)       
        if a != b:
            return False
        for i in range(a):
            if s[i] in t:
                t = t.replace(s[i],'',1)
        if not t:
            return True
        return False