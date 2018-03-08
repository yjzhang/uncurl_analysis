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
]


setup(
    name='uncurl-analysis',
    version='0.0.1',
    author='Yue Zhang',
    author_email='yjzhang@cs.washington.edu',
    url='https://github.com/yjzhang/uncurl_analysis',
    license='MIT',
    include_dirs=[numpy.get_include()],
    ext_modules = cythonize(extensions),
    packages=find_packages("."),
    test_suite='nose.collector',
    tests_require=['nose', 'flaky'],
)
