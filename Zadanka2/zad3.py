#%%
class Solution:
    def maxProfit(self, prices):
        self.pri = prices
        max_profit = 0
        min_price = max(self.pri) 
        for i in range(0,len(self.pri)):
            if self.pri[i] < min_price:
                min_price = self.pri[i]
            elif self.pri[i] - min_price > max_profit:
                max_profit = self.pri[i] - min_price
        return max_profit



# %%
a = Solution

# %%
li=[7,1,5,3,6,4]
# %%
a.maxProfit(a,[7,1,5,3,6,4])
# %%
a.maxProfit(a,[7,6,4,3,1])
# %%
a.maxProfit(a,[2,4,1])
# %%
a.maxProfit(a,[1,2])

# %%
a.maxProfit(a,[3,3])
# %%

a.maxProfit(a,[2,1,2,1,0,1,2])
# %%