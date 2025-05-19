from setuptools import setup, find_packages

setup(
    name='ros_lm',
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/ros_lm']),
        ('share/ros_lm', ['package.xml']),
    ],
    install_requires=['setuptools', 'transformers', 'torch'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='jemonra@uma.es',
    description='ROS 2 Package for Interacting with Open Large [Vision-]Language Models',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'server = ros_lm.ros_lm_server:main',
            'sample_client = ros_lm.ros_lm_sample_client:main'
        ],
    },
)
