import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='parsecpy',
      version='0.9.6',
      description='Parsec Benchmark interface tool',
      url='https://github.com/alexfurtunatoifrn/parsecpy',
      author='Alex Furtunato',
      author_email='alexfurtunato@gmail.com',
      license='MIT',
      packages=['parsecpy'],
      install_requires=[
          'pbr>=1.8',
          'psutil',
          'numpy',
          'pandas',
          'matplotlib>=2.0.2',
          'scikit-learn>=0.19.1',
          'bitstring',
          'scipy'
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
              'parsecpy_runmodel_pso = parsecpy.runmodel_pso:main',
              'parsecpy_runmodel_validation_pso = parsecpy.runmodel_validation_pso:main',
              'parsecpy_runmodel_csa = parsecpy.runmodel_csa:main',
          ],
      },
      keywords='parsec benchmark tool',
      zip_safe=False)
