from setuptools import setup

req = ['numpy', 'scipy', 'scikit-learn', 'sphinx', 'docrep', 'anapymods3']
dep_links = ['http://github.com/user/python_modules3/tarball/' +
             'master#egg=anapymods3-0.1']


setup(name='tdepps',
      version='0.1',
      description='Time Dependent Point Source Analysis',
      author='Thorben Menne',
      author_email='thorben.menne@tu-dortmund.de',
      url='github.com/mennthor/tdepps',
      packages=['tdepps'],
      install_requires=req,
      dependency_links=dep_links,
      )
