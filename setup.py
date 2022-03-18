from glob import glob
from setuptools import setup

# Workaround to ensure pybind11 gets installed (without using toml)
try:
    from pybind11.setup_helpers import Pybind11Extension
except ImportError:
    from setuptools import Extension as Pybind11Extension

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
ext_modules = [
    Pybind11Extension("gofit",
        sorted(glob("src/*.cpp")),
        define_macros = [('VERSION_INFO', __version__)], # pass version to interface
        include_dirs=["/usr/include/eigen3"],
        cxx_std=17,
        ),
]

setup(
    name="gofit",
    version=__version__,
    description="GOFit: Global Optimization for Fitting problems",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author="Jaroslav Fowkes",
    author_email="jaroslav.fowkes@stfc.ac.uk",
    url="https://github.com/ralna/gofit",
    license='New BSD',
    keywords = "mathematics optimization",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Framework :: IPython',
        'Framework :: Jupyter',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        ],
    setup_requires=["pybind11"],
    ext_modules=ext_modules,
    python_requires=">=3.8",
    zip_safe=False,
)
