from setuptools import setup, find_packages
from os.path import join, dirname

setup(
    name='sarcsdet',
    version='1.0',
    author='Ekaterina Polyaeva',
    author_email='katepolyaeva@gmail.com',
    packages=find_packages(),
    long_description=open(join(dirname(__file__), 'README.txt')).read(),
)

