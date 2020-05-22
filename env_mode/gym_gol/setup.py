from setuptools import setup

setup(name='gym_gol',
author="Simone Di Cataldo",
version='0.0.1',
packages=["gym_gol", "gym_gol.envs"],
install_requires=['gym', 'numpy', 'matplotlib']
)
