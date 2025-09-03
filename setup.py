from distutils.core import setup
from setuptools import find_packages

setup(
    name='deep_6dof_tracking',
    version='0.1dev',
    packages=find_packages(),
    author='Mathieu Garon',
    author_email='mathieugaron91@gmail.com',
    requires=['numpy', 'tqdm', 'pyyaml', ]
)