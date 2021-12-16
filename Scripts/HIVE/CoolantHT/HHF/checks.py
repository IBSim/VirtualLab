# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 16:11:24 2016

@author: dhancock
"""

__all__ = ['check_validated']

def check_validated(checks,strictness='verbose'):
    """
    Explanation:
        checks against a set of validation limits in the form:
        
        checks = ('name of check', variable to check, lower limit, upper limit)
        
        strictnes must be 'verbose' strict 'lax'
    """
    import inspect
    
    if strictness.casefold() == 'verbose':
        for checkname,item,lower,upper in checks:
            if (lower <= item <= upper) is False:
                print('{:}: {:} should be between {:} and {:} (not {:})'\
                        .format(inspect.stack()[1][3],
                                checkname,
                                lower,
                                upper,
                                item))
    elif strictness.casefold() == 'strict':
        for name,item,lower,upper in checks:
            assert (lower <= item <= upper) is True,\
                 '{:}: {:} must be between {:} and {:} (not {:})'\
                     .format(inspect.stack()[1][3],
                             checkname,
                             lower,
                             upper,
                             item)
    elif strictness.casefold() == 'none':
        pass
    else:
        print(strictness,'is not a valid strictness')
    