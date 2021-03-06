import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='parsecpy',
      version='1.1.0',
      description='Parsec Benchmark interface tool',
      url='https://github.com/alexfurtunatoifrn/parsecpy',
      author='Alex Furtunato',
      author_email='alexfurtunato@gmail.com',
      license='MIT',
      packages=['parsecpy'],
      install_requires=[
          'pbr>=1.8',
          'psutil',
          'cpufreq',
          'numpy',
          'pandas',
          'xarray',
          'matplotlib',
          'scikit-learn',
          'bitstring',
          'scipy',
          'ghalton==0.6.1'
      ],
      long_description=read('README.rst'),
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Topic :: Software Development :: Build Tools",
          "Topic :: Software Development :: Libraries :: Python Modules",
          "Topic :: Scientific/Engineering",
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python :: 3",
      ],
      entry_points={
          'console_scripts': [
              'parsecpy_createinputs = parsecpy.createinputs:main',
              'parsecpy_processlogs = parsecpy.processlogs:main',
              'parsecpy_runprocess = parsecpy.runprocess:main',
              'parsecpy_runmodel = parsecpy.runmodel:main',
              'parsecpy_runmodel_errors = parsecpy.runmodel_errors:main',
          ],
      },
      keywords='parsec benchmark tool',
      zip_safe=False)
