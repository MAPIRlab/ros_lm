from setuptools import find_packages, setup

package_name = 'ros_lm'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jesús Moncada Ramírez',
    maintainer_email='jemonra@uma.es',
    description='ROS 2 Package for Interacting with Open Large [Vision-]Language Models',
    license='Apache 2.0 License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'test = ros_lm.my_python_node:main'
        ],
    },
)
