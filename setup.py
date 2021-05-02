from setuptools import setup

setup(
    name='srcfilesnetwork',
    version='0.1',
    py_modules=['srcfilesnetwork'],
    install_requires=[
        'Click',
        'networkx==2.5',
        'pyvis==0.1.9'
    ],
    entry_points='''
        [console_scripts]
        srcfilesnetwork=srcfilesnetwork:plotnetwork 
    ''',
)
