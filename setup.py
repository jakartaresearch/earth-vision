from setuptools import setup, find_packages

with open('README.md', encoding='utf8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='earth-vision',
    packages=find_packages(),
    version='0.0.1',
    license='MIT',
    description='Python library for solving computer vision tasks specifically for satellite imagery',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Jakarta Research Team',
    author_email='researchjair@gmail.com',
    url='https://github.com/jakartaresearch/earth-vision',
    download_url=f'https://github.com/jakartaresearch/earth-vision/archive/v{version}.tar.gz',
    keywords=['computer-vision', 'pytorch',
              'machine-learning', 'satellite-imagery'],
    install_requires=required,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Researchers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
