from setuptools import find_packages, setup


def read_file(file):
    with open(file, "rt") as f:
        return f.read()


setup(
    name='cimg_desc',
    description='Chemical-Informed Molecular Graph Descriptor',
    keywords=(["Retrosynthesis", "Machine Learning"]),
    packages=find_packages(exclude=[]),
    package_data={'cimg_desc': ['*.pt']},
    author='Baicheng Zhang',
    author_email='zhangbc0315@outlook.com',
    license='Apache License v2',
    url='',
    install_requires=[i for i in read_file("requirements.txt").strip().splitlines() if i != ''],
    zip_safe=False,
)
