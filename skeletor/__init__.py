"""
Package overview.
"""

__version__ = '0.1.0'
__author__ = 'Jack Featherstone'
__credits__ = 'Okinawa Institute of Science and Technology; Certain parts based on MarcSchotman\'s code' 

# The below code was mostly copied from scipy

import importlib as _importlib

submodules = [
    'spatial',
    'utils',
    'skeleton',
    'data'
]

__all__ = submodules


def __dir__():
    return __all__


def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f'skeletor.{name}')
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(
                f"Module 'skeletor' has no attribute '{name}'"
            )
