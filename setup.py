from setuptools import find_packages, setup

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setup(
    name             = 'MAMMAL',
    packages         = find_packages(),
    version          = '0.0.1',
    description      = 'Python package used to create aeromagnetic anomaly maps for Magnetic Navigation (MagNav)',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    author           = 'Power_Broker',
    author_email     = 'gitstuff2@gmail.com',
    url              = 'https://github.com/PowerBroker2/MAMMAL',
    download_url     = 'https://github.com/PowerBroker2/MAMMAL/archive/0.0.1.tar.gz',
    keywords         = ['navigation', 'mapping', 'geospatial', 'magnetometer', 'survey', 'remote-sensing', 'magnet', 'magnav'],
    classifiers      = [],
    install_requires = ['numpy==1.20.3',
                        'pandas==1.3.4',
                        'scipy==1.7.1',
                        'matplotlib==3.4.3',
                        'tqdm==4.62.3',
                        #'gdal==3.4.3', # Must be installed via conda
                        'xarray==2022.3.0',
                        'rioxarray==0.10.3',
                        'scikit-learn==0.24.2',
                        'ppigrf==1.0.0',
                        'geoscraper==0.0.1',
                        'simplekml==1.3.3']
)
