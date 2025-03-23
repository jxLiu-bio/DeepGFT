from setuptools import Command, find_packages, setup

__lib_name__ = "DeepGFT"
__lib_version__ = "1.0.0"
__description__ = "clustering spatial transcriptomics data using deep learning and graph Fourier transform"
__url__ = "https://github.com/jxLiu-bio/DeepGFT"
__license__ = "MIT"

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    license = __license__,
    packages = ['DeepGFT'],
    install_requires = ["requests"],
    zip_safe = False,
    include_package_data = True
)
