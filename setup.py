from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

#with open('LICENSE') as f:
#    license = f.read()

setup(
    name='nmmn',
    version='1.1',
    description='Miscellaneous methods for astronomy, dealing with arrays, statistical distributions and computing goodness-of-fit',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Rodrigo Nemmen',
    author_email='rodrigo.nemmen@iag.usp.br',
    url='https://github.com/rsnemmen/nmmn',
    download_url = 'https://github.com/rsnemmen/nmmn/archive/1.1.tar.gz', # I'll explain this in a second
    license="MIT License",
    keywords = ['science', 'statistics', 'signal-processing', 'numerical-methods', 'astronomy', 'numerical-simulations', 'astrophysics', 'mhd', 'grmhd'], # arbitrary keywords
    packages=find_packages(exclude=('tests', 'docs'))
    )