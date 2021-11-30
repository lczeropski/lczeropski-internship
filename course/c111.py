#%%
def sum_pairs(li,num):
    for i in range(0,len(li)-2) :
        for j in range(1,len(li)-1):
            if li[i]+li[j]== num :
                return [li[i],li[j]]
    return []    


