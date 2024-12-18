from setuptools import find_packages, setup

setup(
    name="point_sam",
    version="1.0",
    author="Yuchen Zhou",
    author_email="zyc200187@gmail.com",
    install_requires=[],
    packages=find_packages(),
    extras_require={
        "all": ["matplotlib", "pycocotools", "opencv-python", "timm"],
        "dev": ["flake8", "isort", "black", "mypy"],
    },
)
