from setuptools import find_packages, setup

package_name = 'roboship_core'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/roboship_core/launch', ['roboship_core/LAUNCH_FILE/launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='roboship',
    maintainer_email='roboship@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'status = roboship_core.old_scripts.status_broadcaster:main',
	        'mocap_relay = roboship_core.mocap_scripts.mocap_relay:main',
            'calm_water_nav = roboship_core.navigation_scripts.calm_water_nav:main',
            'manual_control = roboship_core.navigation_scripts.manual_control:main',
            'fake_mocap = roboship_core.navigation_scripts.fake_mocap:main',
            'boundary_box = roboship_core.calibration_scripts.boundary_box_detection:main',
            'motor_calibration = roboship_core.calibration_scripts.motor_calibration:main',
            'mission_control = roboship_core.cool_demo_scripts.mission_control_server:main',
            'master_logger = roboship_core.logger_scripts.master_logger:main',
            'mocap_decay_logger = roboship_core.logger_scripts.mocap_decay_logger:main',
            'plot_power_log = roboship_core.logger_plotter_scripts.plot_power_log:main',
            'post_run_process = roboship_core.logger_plotter_scripts.post_run_process:main',
            'LOS = roboship_core.navigation_scripts.LOS:main',
            'ILOS = roboship_core.navigation_scripts.ILOS:main',
            'pure_pursuit = roboship_core.navigation_scripts.pure_pursuit:main',
        ],
    },
)
