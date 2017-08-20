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
            'matplotlib>=2.0.2',
            ],
      scripts=['bin/parsecpy_runprocess', 'bin/parsecpy_processlogs',
               'bin/parsecpy_createinputs'],
      zip_safe=False)