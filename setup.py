import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='parsecpy',
      version='0.2',
      description='Parsec Benchmark tool',
      url='https://github.com/alexfurtunatoifrn/parsecpy',
      author='Alex Furtunato',
      author_email='alexfurtunato@gmail.com',
      license='MIT',
      packages=['parsecpy'],
      install_requires=[
            'pandas',
            'matplotlib>=2.0.2',
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
          ],
      },
      keywords='parsec benchmark tool',
      zip_safe=False)
