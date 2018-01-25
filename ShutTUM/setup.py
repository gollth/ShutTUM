from setuptools import setup

setup(
    name='ShutTUM',
    version='0.1',
    author='Thore Goll',
    author_email='thore.goll@tum.de',
    description='A utility API to easily interact with the ShutTUM dataset',
    #long_description=read('README.md')
    url='https://github.com/gollth/ShutTUM',
    packages=['ShutTUM'],
    install_requires=[
        'cv2',
        'numpy'
        'transforms3d',
        'yaml',
        'zipfile'
    ]
)