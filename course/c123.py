#%%
def sum_up_diagonals(li):
    n = len(li)
    result = [li[j][i] for j in range(0,n) for i in range(0,n) if j==i] 
    result2 = [li[j][-i] for j in range(0,n) for i in range(1,n+1) if j==i-1] 
    return sum(result) + sum (result2)    
        
# %%
list4 = [
  [ 1, 2, 3, 4 ],
  [ 5, 6, 7, 8 ],
  [ 9, 10, 11, 12 ],
  [ 13, 14, 15, 16 ]
]
# %%
sum_up_diagonals(list4)
# %%
list3 = [
  [ 4, 1, 0 ],
  [ -1, -1, 0],
  [ 0, 0, 9]
]
# %%
sum_up_diagonals(list3)
# %%
list2 = [
  [ 1, 2, 3 ],
  [ 4, 5, 6 ],
  [ 7, 8, 9 ]
]
# %%
sum_up_diagonals(list2)
# %%
