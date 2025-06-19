from setuptools import setup, find_packages

setup(
    name='diabetes_regression_comparison',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'scikit-learn',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
    ],
) 