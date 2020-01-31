from setuptools import setup, find_packages

setup(
    name='pose_viterbi',
    version='0.0.1',
    # url='https://github.com/mypackage.git',
    author='Johannes Haux',
    author_email='jo.mobile.2011@gmail.com',
    description='Custom implementation inspired by the viterbi algorithm to '
                'find matching pose sequences.',
    packages=find_packages(),    
    install_requires=['numpy'],
)
