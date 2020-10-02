from setuptools import setup

setup(name='nngeometry',
      version='0.1',
      description='Manipulate geometry matrices in Pytorch',
      url='https://github.com/OtUmm7ojOrv/nngeometry',
      author='anonymous',
      author_email='anonymous@anonymous.com',
      license='MIT',
      packages=['nngeometry',
                'nngeometry.generator',
                'nngeometry.object'],
      install_requires=['torch>=1.0.0'],
      zip_safe=False)
