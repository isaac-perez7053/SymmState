from setuptools import setup, find_packages

setup(
    name="symmstate",  # Name of your package
    version="1.0.0",  # Package version
    description="FLPZ: A program ",
    author="Isaac Perez",  # Your name or team name
    author_email="iperez@g.hmc.edu",  # Your email
    url="https://github.com/yourusername/flpz",  # URL to your project repository (optional)
    packages=find_packages(),  # Automatically finds all submodules and packages
    include_package_data=True,  # Includes non-Python files specified in MANIFEST.in
    python_requires=">=3.7",  # Specify the minimum Python version
    install_requires=["numpy", "pymatgen", "matplotlib", "scipy"],
    entry_points={
        "console_scripts": [
            "flpz=flpz:main",  # Command `flpz` will invoke the `main` function in `flpz/__init__.py`
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
