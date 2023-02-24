from setuptools import setup
from setuptools import find_packages

setup(
    name="mvhg",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
    ],
    extras_require={
        "pt": ["torch"],
        "tf": ["tensorflow"],
        "tf_gpu": ["tensorflow-gpu"],
    },
)
