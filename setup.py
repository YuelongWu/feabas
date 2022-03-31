from setuptools import setup, find_packages

VERSION = '0.0'

setup(
    name='fem-aligner',
    version=VERSION,
    description='A python library for stitching & alignment of connectome datasets using finite-element method.',
    author='Yuelong Wu',
    author_email='yuelong.wu.2017@gmail.com',
    packages=find_packages(),
    license='MIT',
    install_requires=[
        'h5py',
        'matplotlib',
        'opencv-python',
        'pyyaml',
        'rtree',
        'scipy',
        'shapely',
        'triangle'
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3'
    ]
)