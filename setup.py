#!/usr/bin/env python

from distutils.core import setup
from QDYNTransmonLib import __version__

setup(name='QDYNTransmonLib',
      version=__version__,
      description='Transmon Hamiltonian',
      author='Michael Goerz',
      author_email='goerz@physik.uni-kassel.de',
      license='GPL',
      packages=['QDYNTransmonLib', 'QDYNTransmonLib.ham'],
      scripts=[]
     )
