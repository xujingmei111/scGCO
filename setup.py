from setuptools import setup, find_packages
setup(
        name='scGCO',
        version='1.1.0',
        description='Spatial Gene',
        url='https://github.com/Wanglab/scGCO',
        packages=find_packages(),
        include_package_data=True,
        install_requires=['parmap','numpy','matplotlib','scipy','sklearn','Cython','pygco',
                        'tqdm','networkx','shapely'],
        author='Peng Wang',
        author_email='wangpeng@picb.ac.cn',
        license='MIT'

)