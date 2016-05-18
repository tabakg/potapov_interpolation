from setuptools import setup

setup(name='Potapov_interpolation',
    version='0.1',
    description='Treating feedback with delays in quantum systems',
    url='https://github.com/tabakg/potapov_interpolation/',
    author='Gil Tabak',
    author_email='tabak.gil@gmail.com',
    license='GNU',
    packages=['Potapov_Code'],
    install_requires=[
        'matplotlib',
        'sympy',
        'numpy',
    ],
    extras_require=[
        'QNET==1.4.1',
    ],
    dependency_links = [
        'https://github.com/mabuchilab/QNET.git#egg=QNET-1.4.1',
    ],
      zip_safe=False)
