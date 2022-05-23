def main(min,max,avoid=[]):
    '''
    Function to find factors of random number between two intergers min and max.
    can also be given an optional list of numbers to avoid.
    '''
    import random
    
    if max>100000:
        print("Max is to big I'm not a supercompter you know!!!")
        return    
    if min<0:
        print("Min must not be negative")
        return
    if list(range(min,max+1))==avoid:
        print("you are avoiding every number between min and max you twit")
        return

    Lucky = random.randint(min,max)
    while Lucky in avoid:
        Lucky = random.randint(min,max)
    factors = factorization(Lucky)
    nfactors=len(factors) 
    

    print(F"You asked for a random number between {min} and {max}")
    print(F"Today's Lucky number is {Lucky}")
    if nfactors == 0:
        print(F"This Number is Special in that it has no factors (not even 1)")
        return
    if nfactors == 1:
        print(F"This Number is Special it has only one factor which is itself.")
        return
    if nfactors <= 2:
        print(F"This Number is prime")
        return
    print(F"This Number has {nfactors} factors which are {factors}")
    return 

def factorization(n):
    factors = set()
    for N in range(1, n + 1):
        if n % N == 0:  # remainder is zero
            factors.add(N)
    return factors
