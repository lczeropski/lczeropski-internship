#%%
def find_greater_numbers(li):
    num = 0
    for i in range(len(li)-1,-1,-1):
        for j in range(0,i):
            if li[i]>li[j]:
                num +=1
    return num
# %%
find_greater_numbers([5,4,3,2,1]) # 0
# %%
find_greater_numbers([6,1,2,7]) # 4
# %%
