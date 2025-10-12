from setuptools import setup, find_packages

setup(
    name="matcher",
    version="0.1.0",
    description="Research skeleton for image matching",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.23",
        "opencv-python>=4.7",
        "torch>=1.12",
        "matplotlib>=3.5",
        "pillow>=9.0",
    ],
    python_requires=">=3.8",
)
