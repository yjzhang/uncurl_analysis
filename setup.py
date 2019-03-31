from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


extensions = [
    Extension('uncurl_analysis.sparse_gene_extraction',
             ['uncurl_analysis/sparse_gene_extraction.pyx'],
             extra_compile_args=['-O3', '-ffast-math']),
    Extension('uncurl_analysis.sparse_bulk_data',
             ['uncurl_analysis/sparse_bulk_data.pyx'],
             extra_compile_args=['-O3', '-ffast-math']),
    Extension('uncurl_analysis.poisson_tests',
             ['uncurl_analysis/poisson_tests.pyx'],
             extra_compile_args=['-O3', '-ffast-math']),
]

install_requires = [
        'uncurl-seq',
        'cython',
        'numpy',
        'scipy',
        'scikit-learn',
        'requests',
        'umap-learn',
        'tables',
        'pandas'
]

setup(
    name='uncurl-analysis',
    version='0.1.1',
    author='Yue Zhang',
    author_email='yjzhang@cs.washington.edu',
    url='https://github.com/yjzhang/uncurl_analysis',
    license='MIT',
    include_dirs=[numpy.get_include()],
    ext_modules = cythonize(extensions),
    install_requires=install_requires,
    packages=find_packages("."),
    test_suite='nose.collector',
    tests_require=['nose', 'flaky'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
    ],

)
