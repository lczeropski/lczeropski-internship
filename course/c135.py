#%%
def next_prime():
    num = 2
    all_primes = set()
    while True:
        for prime in all_primes:
            if num % prime == 0:
                break
        else:
            all_primes.add(num)
            yield num
        num += num % 2 + 1
            
            
# %%
primes = next_prime()
[next(primes) for i in range(1)]
# %%
