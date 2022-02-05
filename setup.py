import setuptools

dev = [
    "black==22.1.0",
    "isort==5.10.1"
]

setuptools.setup(
    name="vae",
    version="0.1.0",
    author="Daniel John Varoli",
    description="Variational Auto Encoder",
    url="https://github.com/djvaroli/vae",
    package_dir={"": "vae"},
    install_requires=[
        "tensorflow==2.7.0"
    ]
)