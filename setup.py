from setuptools import setup, find_packages


from distutils.command.build_py import build_py

import os

#with open('README.md', 'r', encoding='utf8') as document:
    #long_description = document.read()

with open('divergence/__init__.py', 'r') as f:
    exec(f.readline())


# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')


def _get_requirements_from_files(groups_files):
    groups_reqlist = {}

    for k, v in groups_files.items():
        with open(v, 'r') as f:
            pkg_list = f.read().splitlines()
        groups_reqlist[k] = pkg_list

    return groups_reqlist


def setup_package():
    # get all file endings and copy whole file names without a file suffix
    # assumes nested directories are only down one level
    _groups_files = {
        'base': 'requirements.txt',
        'tests': 'requirements_tests.txt',
    }

    modules = _get_requirements_from_files(_groups_files)
    install_modules = modules.pop('base')

    setup(
        name = 'divergence',
        version = '0.1.0',
        description = "Analytics for multiscale and multigroup segregation measures.",
        #long_description = long_description,
		long_description_content_type = "text/markdown",
        maintainer = "Cecile de Bezenac",
        maintainer_email = 'gycdb@leeds.ac.uk',
        #url=,
        #download_url='https://pypi.org/project/segregation/',
        license = 'MIT',
        #py_modules = ['segregation'],
        packages = find_packages(exclude=['*test']),
        # setup_requires = ["pytest-runner"],
        # tests_require = ["pytest"],
        keywords = ['segregation','entropy','spatial statistics', 'demography'],
        install_requires = install_modules,
        cmdclass = {'build_py': build_py},
        python_requires = '>3.4',
        entry_points={'console_scripts': ['hunt = package.command:process']}
)
    
if __name__ == '__main__':
    setup_package()