#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

__minimum_numpy_version__ = '1.10.0'
__minimum_tensorflow_version__ = '1.14.0'

setup_requires = ['numpy>=' + __minimum_numpy_version__, 
'tensorflow>='+__minimum_tensorflow_version__]

setup(name='rayintegral_kernels',
      version='0.0.1',
      description='Implements ray integral kernels.',
      author=['Josh Albert', 'Martijn Oei'],
      author_email=['albert@strw.leidenuniv.nl', 'oei@strw.leidenuniv.nl'],
    setup_requires=setup_requires,  
    tests_require=[
        'pytest>=2.8',
    ],
    package_data= {'rayintegral_kernels':['data/*']},
   package_dir = {'':'./'},
   packages=find_packages('./')
     )

