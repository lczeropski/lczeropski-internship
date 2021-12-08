#%%
class Solution:
    def __init__(self, prices):
        self.p = prices
    def maxProfit(self):
        max_profit = 0
        min_price = max(self.p) 
        for i in range(0,len(self.p)):
            if self.p[i] < min_price:
                min_price = self.p[i]
            elif self.p[i] - min_price > max_profit:
                max_profit = self.p[i] - min_price
        return max_profit



# %%
a = Solution([7,1,5,3,6,4])
b = Solution([7,6,4,3,1])
c = Solution([2,4,1])
d = Solution([1,2])
e = Solution([2,1,2,1,0,1,2])
# %%
a.maxProfit()
# %%
b.maxProfit()
# %%
c.maxProfit()
# %%
d.maxProfit()
# %%
e.maxProfit()
# %%
