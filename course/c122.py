#%%
def find_the_duplicate(li):
    for num in li:
        if num in li[li.index(num)+1:]:
            return num
    return None
#%%
find_the_duplicate([1,2,1,4,3,12]) # 1
# %%
find_the_duplicate([6,1,9,5,3,4,9]) # 9
# %%
find_the_duplicate([2,1,3,4]) # None