from setuptools import find_packages, setup

package_name = 'scan_detection_fusion'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aarav',
    maintainer_email='aarav@todo.todo',
    description='LiDAR + camera fusion node — produces named object map positions',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fuser_node = scan_detection_fusion.fuser_node:main',
        ],
    },
)