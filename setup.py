from setuptools import find_packages, setup

package_name = 'ur3_injection_controller'

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
    maintainer='gab',
    maintainer_email='gabriel.fumagalli@studenti.unitn.it',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ur3_injection = ur3_injection_controller.ur3_injection:main',
            'eye_tracking = ur3_injection_controller.eye_tracking:main',
            'eye_sphere_tracker = ur3_injection_controller.eye_sphere_tracker:main',
            'ur3_eye_motion = ur3_injection_controller.ur3_eye_motion:main',
            'gaze_replayer = ur3_injection_controller.gaze_replayer:main',
            'injection_path_planner = ur3_injection_controller.injection_path_planner:main'
        ],
    },
)
