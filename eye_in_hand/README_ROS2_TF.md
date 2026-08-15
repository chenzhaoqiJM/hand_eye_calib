# ROS 2 TF 自动采集

`calibrate_ros2_tf.py` 基于 `calibrate.py`，按 Enter 时保存一张 D405 图像，
并同时读取以下最新 TF：

```text
base_footprint_link <- Right_Arm_Link8
```

该方向表示 `Right_Arm_Link8` 在 `base_footprint_link` 中的位姿，与
`calibrate_from_data.py` 所需的法兰到基座位姿一致。输出仍是
`frame_NNN.png`、`pose_NNN.json` 和 `poses.txt`，可直接使用原求解器。

## 运行

先打开已加载 ROS 2 环境的终端。Windows ROS 2 可使用相应的
`local_setup.bat`；如果采集脚本实际运行在 Ubuntu，则先 `source` ROS 环境。
安装本项目 Python 依赖后运行：

```bash
python calibrate_ros2_tf.py --tag-size-mm 77.6 --min-samples 20
```

脚本默认使用：

```text
ROS_DOMAIN_ID=25
target/base frame: base_footprint_link
source/flange frame: Right_Arm_Link8
```

所以通常不需要 SSH 到 `k3@192.168.11.116`。只要采集电脑和 K3 在同一局域网，
ROS 2 发行版/DDS 配置兼容，并且两端防火墙允许 DDS 流量，脚本就能直接发现 K3
发布的 `/tf`。启动时会先等待并打印 TF，确认成功后才启动采集。

如果 TF 发现失败，先在运行脚本的同一台电脑、同一个 ROS 环境测试：

```bash
ROS_DOMAIN_ID=25 ros2 run tf2_ros tf2_echo base_footprint_link Right_Arm_Link8
```

Windows PowerShell 的等价设置方式是：

```powershell
$env:ROS_DOMAIN_ID = "25"
ros2 run tf2_ros tf2_echo base_footprint_link Right_Arm_Link8
```

脚本内部也会在初始化 ROS 节点前设置 `ROS_DOMAIN_ID`，可用
`--ros-domain-id` 修改。其他常用参数：

```bash
python calibrate_ros2_tf.py --no-solve
python calibrate_ros2_tf.py --output-dir data/my_ros2_session
python calibrate_ros2_tf.py --base-frame base_footprint_link --flange-frame Right_Arm_Link8
python calibrate_ros2_tf.py --max-tf-age 1.0
python calibrate_ros2_tf.py --no-preview
```

采集时让机械臂完全停止后再按 Enter。默认拒绝时间戳超过 2.5 秒的动态 TF；
每个 `pose_NNN.json` 还会保存 TF 时间戳、四元数和 4x4 矩阵，方便检查。
