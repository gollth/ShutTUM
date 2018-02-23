from setuptools import setup, find_packages

setup(
    name='ShutTUM',
    version='1.1',
    author='Thore Goll',
    author_email='thoregoll@googlemail.de',
    description='A utility API to easily interact with the ShutTUM dataset',
    url='https://github.com/gollth/ShutTUM',
    license='MIT',
    packages=['ShutTUM'],
    keywords='dataset rolling shutter visual odometry slam',
    long_description=open('README.md').read(),
    install_requires=[
        'opencv-python>=3.2.0',
        'numpy>=1.12',
        'transforms3d>=0.3.1',
        'PyYAML>=3.12'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Localization',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
      ]
)