from setuptools import setup

exec(open("boicl/version.py").read())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="boicl",
    version=__version__,
    description="BayesOPT with In-Context Learning",
    author="Andrew White",
    author_email="andrew.white@rochester.edu",
    url="https://github.com/ur-whitelab/BO-ICL",
    license="MIT",
    packages=["boicl"],
    package_data={
        "boicl": [
            "data/moc/*.csv",
            "data/moc/*.gz",
            "data/moc/*.json",
            "prompts/*.txt",
            "prompts/*.json",
            "prompts/*.md",
            "toolkit_main.js",
        ]
    },
    install_requires=[
        "numpy",
        "langchain",
        "langchain_community",
        "langchain_openai",
        "langchain_anthropic",
        "openai",
        "faiss-cpu",
        "scipy",
        "pandas",
        "tiktoken",
        "openpyxl",
        "python-dotenv",
    ],
    extras_require={"gpr": ["scikit-learn", "torch", "botorch", "gpytorch"]},
    test_suite="tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
