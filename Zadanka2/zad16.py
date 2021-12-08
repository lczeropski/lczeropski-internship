#%%
class Solution:
    def __init__(self,string) -> None:
        self.s = string
    def lengthOfLongestSubstring(self) -> int:
        start = 0
        max_len = 0
        temp = ''
        for i, c in enumerate(self.s):
            if c not in temp:
                temp +=c
            else :
                temp_len = len(temp)
                max_len = max(max_len,temp_len)
                if c == temp[-1]:
                    start = i
                else:
                    start = temp.index(c) + start +1
                temp = s[start:i+1]
        temp_len = len(temp)
        max_len = max(max_len,temp_len)
        return (max_len)
        
                
            
            
# %%
s=Solution("abcabcbb")
# %%
s.lengthOfLongestSubstring()
# %%
#DONE:enumerate = index , value
a = 'abcdefghijk'
for i,v in enumerate(a):
    print(i,v)
# %%
