from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # ── 1. Motion capture tracking (Qualisys → /poses) ──
    mocap_launch_path = os.path.join(
        get_package_share_directory('motion_capture_tracking'),
        'launch',
        'launch.py'
    )
    mocap_tracking = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(mocap_launch_path),
    )

    # ── 2. Mocap relay (/poses → /mavros/vision_pose/pose) ──
    mocap_relay = Node(
        package='roboship_core',
        executable='mocap_relay',
        name='mocap_relay',
        output='screen',
    )

    # ── 3. Publish global origin once (Imperial College London) ──
    set_origin = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(cmd=[
                'ros2', 'topic', 'pub', '--once',
                '/mavros/global_position/set_gp_origin',
                'geographic_msgs/msg/GeoPointStamped',
                '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: "map"}, '
                'position: {latitude: 51.498356, longitude: -0.176894, altitude: 0.0}}'
            ])
        ]
    )

    # ── Assemble ──
    return LaunchDescription([
        mocap_tracking,
        mocap_relay,
        set_origin,
    ])