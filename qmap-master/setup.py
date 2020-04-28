from setuptools import setup, find_packages
import sys


setup(name='qmap',
      packages=[package for package in find_packages() if package.startswith('qmap')],
      description="qmap",
      author="Darakshan and Sameer",
      url='https://github.com/darakshananwar13/CSCE_689_RL_FINAL_PROJECT',
      author_email="darakshan@tamu.edu",
      version="0.1")
