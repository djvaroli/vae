from setuptools import find_namespace_packages, setup

dev = ["black==22.1.0", "isort==5.10.1", "pytest==7.0.0"]

setup(
    name="vae",
    version="0.1.0",
    author="Daniel John Varoli",
    description="Variational Auto Encoder",
    url="https://github.com/djvaroli/vae",
    packages=find_namespace_packages(include=["vae.*"], where="."),
    package_dir={".": "vae"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=["tensorflow>=2.7.0"],
)
