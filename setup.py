from setuptools import setup
from setuptools import find_packages

try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()


setup(name='soundification',
      version='1.0.0',
      description='Multi-Channel Audio Classifier',
      long_description=read_md('README.md'),
      author='Ravi Singh',
      author_email='mailbox.ravisingh@gmail.com',
      url='https://github.com/ravising-h/soundification',
      download_url='https://github.com/ravising-h/Soundification/releases/download/1.0.0/Soundification.1.0.0.tar',
      license='MIT',
      install_requires=['pytorch', 'librosa', 'sys','py3nvml'],
      packages=find_packages()
) 
