from setuptools import setup, find_packages

VERSION = '2.0.1'

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
    python_requires='>=3.8',
    packages=find_packages(),
    include_package_data=True,
    license='MIT',
    install_requires=[
        'google-cloud-storage',
        'h5py',
        'matplotlib',
        'numpy',
        'opencv-python',
        'psutil',
        'pyyaml',
        'rtree',
        'scikit-image',
        'scipy',
        'shapely>=2.0.0',
        'tensorstore',
        'triangle',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3'
    ]
)