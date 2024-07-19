from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="semicart",
    version="0.0.1",
    author="Aydin Abedinia",
    author_email="abedinia.aydin@gmail.com",
    description="Building semi-supervised decision trees with semi-cart algorithm",
    long_description="SemiCart is an algorithm based on CART that uses the weights of test data to improve prediction accuracy. This algorithm employs calculation methods such as Nearest Neighbor and metrics like Euclidean and Mahalanobis distances to determine the weights, more infor: https://link.springer.com/article/10.1007/s13042-024-02161-z",
    long_description_content_type="text/markdown",
    url="https://github.com/WeightedBasedAI/semicart",
    packages=find_packages(),
    install_requires=[
        'numpy==2.0.0',
        'scikit-learn==1.5.1',
        'scipy==1.14.0',
        'joblib==1.4.2',
        'db-weights==0.0.7',
        'tqdm==4.66.4'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
