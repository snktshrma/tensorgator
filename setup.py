from setuptools import setup, find_packages

setup(
    name='tensorgator',
    version='0.1.0',
    description='Satellite propagation and coverage visualization toolkit',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'numba',
    ],
    python_requires='>=3.7',
    include_package_data=True,
    url='https://github.com/yourusername/tensorgator',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
