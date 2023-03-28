from setuptools import setup, find_packages

VERSION = '0.1.1'

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='feabas',
    version=VERSION,
    description='A python library for stitching & alignment of connectome datasets using finite-element method.',
    long_description=readme(),
    long_description_content_type="text/markdown",
    author='Yuelong Wu',
    author_email='yuelong.wu.2017@gmail.com',
    packages=find_packages(),
    license='MIT',
    install_requires=[
        'h5py',
        'matplotlib',
        'opencv-python',
        'scikit-image',
        'pyyaml',
        'rtree',
        'scipy',
        'shapely>=2.0.0',
        'triangle'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3'
    ]
)