#%%
def three_odd_numbers(li):
    for i in range(0,len(li)-3):
        if (li[i]+li[i+1]+li[i+2])%2!=0:            
            return True
    return False
# %%
three_odd_numbers([1,2,3,3,2]) # False
# %%
three_odd_numbers([0,-2,4,1,9,12,4,1,0]) # True
# %%
three_odd_numbers([1,2,3,4,5]) # True
# %%
