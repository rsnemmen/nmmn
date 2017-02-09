from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='nmmn',
    version='0.2',
    description='Miscellaneous methods for astronomy, dealing with arrays, statistical distributions and computing goodness-of-fit',
    long_description=readme,
    author='Rodrigo Nemmen',
    author_email='rodrigo.nemmen@iag.usp.br',
    url='https://github.com/rsnemmen/nemmen',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)