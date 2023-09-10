from setuptools import setup, find_packages

setup(
    name="test",
    extras_require=dict(tests=['pytest']),
    packages=find_packages(where='tests'),
    package_dir={"": "./"}
)