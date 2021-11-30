#%%
def two_list_dictionary(first, sec):
    di = {}.fromkeys(first)
    for k in di.keys():
        if sec:
            di[k] = sec[0]
            sec.pop(0)
        else:
            di[k] = None 
    return di
#%%
two_list_dictionary(['a', 'b', 'c', 'd'], [1, 2, 3]) # {'a': 1, 'b': 2, 'c': 3, 'd': None}
# %%
two_list_dictionary(['x', 'y', 'z']  , [1,2]) # {'x': 1, 'y': 2, 'z': None}
# %%
two_list_dictionary(['a', 'b', 'c']  , [1, 2, 3, 4]) # {'a': 1, 'b': 2, 'c': 3}


#%%
#########
def two_list_dictionary(keys, values):
    collection = {}
 
    for idx, val in enumerate(keys):
        if idx < len(values):
            collection[keys[idx]] = values[idx]
        else:
            collection[keys[idx]] = None
 
    return collection
#########
#%%
