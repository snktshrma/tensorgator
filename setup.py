from setuptools import setup, find_packages

setup(
    name='tensorgator',
    version='0.1.1',
    description='Satellite propagation and coverage visualization toolkit',
    author='ApoPeri',
    author_email='ApoPeri@protonmail.com',
    packages=['tensorgator', 'tensorgator.examples'],
    package_dir={'tensorgator': ''},
    package_data={'tensorgator.examples': ['*.ipynb', '*.png']},
    install_requires=[
        'numpy',
        'matplotlib',
        'numba',
    ],
    python_requires='>=3.7',
    include_package_data=True,
    url='https://github.com/ApoPeri/TensorGator',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
