from setuptools import setup

setup(name='parseecpy',
      version='0.1',
      description='Parsec tool interface',
      url='https://github.com/alexfurtunatoifrn/parsecpy',
      author='Alex Furtunato',
      author_email='alexfurtunato@gmail.com',
      license='MIT',
      packages=['parsecpy'],
      install_requires=[
            'pandas',
            'matplotlib',
      ],
      scripts=['bin/parsecpy_runprocess', 'bin/parsecpy_processlogsfolder',
               'bin/parsecpy_createinputsizes'],
      zip_safe=False)