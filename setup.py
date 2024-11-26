import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='skeletor',  
     version='0.2.0',
     author="Jack Featherstone",
     author_email="jack.featherstone@oist.jp",
     license='MIT',
     url='https://jfeatherstone.github.io/skeletor',
     description="Point cloud skeletonization library.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3.11",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
