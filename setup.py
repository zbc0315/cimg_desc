from setuptools import find_packages, setup


def read_file(file):
    with open(file, "rt") as f:
        return f.read()


setup(
    name='cimg_desc',
    version='0.0.1',
    description='Chemical-Informed Molecular Graph Descriptor',
    keywords=(["Retrosynthesis", "Machine Learning"]),
    packages=find_packages(exclude=[]),
    package_data={'cimg_desc': ['*.pt']},
    author='Baicheng Zhang',
    author_email='zhangbc0315@outlook.com',
    license='Apache License v2',
    url='https://github.com/zbc0315/cimg_desc',
    install_requires=[i for i in read_file("requirements.txt").strip().splitlines() if i != ''],
    zip_safe=False,
    python_requires='==3.7',
)
