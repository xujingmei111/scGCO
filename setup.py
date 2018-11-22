from setuptools import setup, find_packages
setup(
        name='scGCO',
        version='0.0.1',
        description='Spatial Gene',
        url='https://github.com/WangPeng-Lab/scGCO',
        packages=find_packages(),
        include_package_data=True,
        install_requires=['parmap','numpy','matplotlib','scipy','sklearn','Cython','pygco',
                        'tqdm','networkx','shapely'],
        author='Peng Wang',
        author_email='wangpeng@picb.ac.cn',
        license='MIT'

)
