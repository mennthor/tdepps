from setuptools import setup

setup(name='tdepps',
      version='0.1',
      description='Time Dependent Point Source Analysis',
      author='Thorben Menne',
      author_email='thorben.menne@tu-dortmund.de',
      url='github.com/mennthor/tdepps',
      packages=['tdepps'],
      install_requires=['numpy', 'scipy', 'scikit-learn', 'docrep'],
      )

