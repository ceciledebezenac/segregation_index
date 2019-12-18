
from setuptools import setup, find_packages

long_description='README.md'

setup(
    name = "segregation_distortion",
    version = "0.1.0",
    packages = find_packages(exclude=['*test']),
    scripts = ['divergence/segregation_distortion'],
    install_requires = ['argparse','numpy','seaborn','pandas','geopandas','matplotlib','math','random']
    description = "Analytics for multiscale and multigroup segregation measures.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    maintainer = "Cecile de Bezenac",
    maintainer_email = 'gycdb@leeds.ac.uk'
    download_url='git@github.com:ceciledebezenac/segregation_index.git'
    license = 'MIT',
    keywords = ['segregation','entropy','spatial statistics', 'demography'],
    python_requires = '>3.4'
)
    
if __name__ == '__main__':
    setup_package()
