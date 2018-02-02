from setuptools import setup, find_packages

setup(
    name='ShutTUM',
    version='1.0',
    author='Thore Goll',
    author_email='thore.goll@tum.de',
    description='A utility API to easily interact with the ShutTUM dataset',
    url='https://github.com/gollth/ShutTUM',
    packages=['ShutTUM'],
    long_description=open('README.md').read(),
    install_requires=[
        'opencv-python>=3.2.0',
        'numpy>=1.12',
        'transforms3d>=0.3.1',
        'PyYAML>=3.12'
    ]
)