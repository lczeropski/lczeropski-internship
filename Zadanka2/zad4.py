#%%
class Solution:
    def __init__(self, string):
        self.s = string
    def isAnagram(self, t: str) -> bool:
        return {i:self.s.count(i) for i in self.s} == {i: t.count(i) for i in t}
# %%
a=Solution("anagram")
b=Solution("word")
c=Solution("notanagram")
# %%

#%%
a.isAnagram("nagaram")
#%%
b.isAnagram("drow")
#%%
c.isAnagram("asdacsa")
# %%
class Solution: #szybsze rozwiazanie
    def __init__(self, string):
        self.s = string
    def isAnagram(self,t: str) -> bool:
        a, b = len(self.s), len(t)       
        if a != b:
            return False
        for i in range(a):
            if self.s[i] in t:
                t = t.replace(self.s[i],'',1)
        if not t:
            return True
        return False
