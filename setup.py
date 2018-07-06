from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='nmmn',
    version='0.8.4',
    description='Miscellaneous methods for astronomy, dealing with arrays, statistical distributions and computing goodness-of-fit',
    long_description=readme,
    author='Rodrigo Nemmen',
    author_email='rodrigo.nemmen@iag.usp.br',
    url='https://github.com/rsnemmen/nmmn',
    download_url = 'https://github.com/peterldowns/mypackage/archive/0.8.4.tar.gz', # I'll explain this in a second
    license=license,
    keywords = ['science', 'statistics', 'signal-processing', 'numerical-methods', 'astronomy', 'numerical-simulations', 'astrophysics', 'mhd', 'grmhd'], # arbitrary keywords
    packages=find_packages(exclude=('tests', 'docs'))
)

