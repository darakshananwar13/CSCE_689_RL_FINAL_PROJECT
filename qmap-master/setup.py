from setuptools import setup, find_packages
import sys


setup(name='qmap',
      packages=[package for package in find_packages() if package.startswith('qmap')],
      )
