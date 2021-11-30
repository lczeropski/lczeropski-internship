#%%
def valid_parentheses(para):
    l = 0
    p = 0
    for i in para:
        if i == '(':
            l += 1
        elif i == ')':
            p += 1 
    return l==p and para[0]=='(' and para[len(para)-1]==')'
    
    
# %%
valid_parentheses('()()()()())()(') # False
# %%
valid_parentheses("(())((()())())") # True 
# %%
