#%%
class Solution:
    def __init__(self,string) -> None:
        self.s = string
    def characterReplacement(self, k: int) -> int:
        Start = 0
        maxRepeat = 0
        maxLength = 0
        
        chars = {}
        
        for End in range(len(self.s)):
            rightChar = self.s[End]
            if rightChar not in chars:
                chars[rightChar] = 0
            chars[rightChar] += 1
            
            maxRepeat = max(maxRepeat, chars[rightChar])
            
            if (End-Start+1 - maxRepeat) > k:
                leftChar = self.s[Start]
                chars[leftChar] -= 1
                Start += 1
            
            
            maxLength = max(maxLength, End-Start+1)
        return maxLength
# %%
a = Solution("IMNJJTRMJEGMSOLSCCQICIHLQIOGBJAEHQOCRAJQMBIBATGLJDTBNCPIFRDLRIJHRABBJGQAOLIKRLHDRIGERENNMJSDSSMESSTR")
b = Solution('ABBA')
# %%
a.characterReplacement(2)
# %%
b.characterReplacement(2)

# %%
