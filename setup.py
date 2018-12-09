from setuptools import setup, find_packages
setup(
        name='scGCO',
        version='0.0.1',
        description='single-cell graph cuts optimization',
        url='https://github.com/WangPeng-Lab/scGCO',
        packages=find_packages(),
        include_package_data=True,
        install_requires=['pandas','numpy','matplotlib >=2.0.0, <3.0.0','scipy','sklearn','parmap','Cython','pygco',
                        'tqdm','networkx','shapely','statsmodels'],
        author='Peng Wang',
        author_email='wangpeng@picb.ac.cn',
        license='MIT'

)
