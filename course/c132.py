#%%
def running_average():
    running_average.acc = 0
    running_average.n = 0
    def inner(num):
        running_average.acc += num
        running_average.n += 1
        return running_average.acc / running_average.n
    return inner
# %%
rAvg = running_average()
rAvg(10) # 10.0
rAvg(11) # 10.5
rAvg(12) # 11
# %%
rAvg2 = running_average()
rAvg2(1) # 1
rAvg2(3) # 2