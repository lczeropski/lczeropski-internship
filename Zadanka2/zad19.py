class Solution:
    def countSubstrings(self, s: str) -> int:
        res = n = len(s) 
        for i in range(n):
            r = l = i
            while r + 1 < n and s[l] == s[r+1]:
                r += 1
                res += 1
            while r + 1 < n and l > 0 and s[l-1] == s[r+1]:
                r += 1
                l -= 1
                res += 1
        return res