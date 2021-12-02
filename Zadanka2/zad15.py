#%%
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        Start = 0
        maxRepeat = 0
        maxLength = 0
        
        chars = {}
        
        for End in range(len(s)):
            rightChar = s[End]
            if rightChar not in chars:
                chars[rightChar] = 0
            chars[rightChar] += 1
            
            maxRepeat = max(maxRepeat, chars[rightChar])
            
            if (End-Start+1 - maxRepeat) > k:
                leftChar = s[Start]
                chars[leftChar] -= 1
                Start += 1
            
            
            maxLength = max(maxLength, End-Start+1)
        return maxLength
# %%
a=Solution
# %%
a.characterReplacement(a,"IMNJJTRMJEGMSOLSCCQICIHLQIOGBJAEHQOCRAJQMBIBATGLJDTBNCPIFRDLRIJHRABBJGQAOLIKRLHDRIGERENNMJSDSSMESSTR",2)
# %%
a.characterReplacement(a,'ABBB',2)

# %%
