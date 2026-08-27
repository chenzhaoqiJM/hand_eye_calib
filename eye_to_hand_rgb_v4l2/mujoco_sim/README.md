# MuJoCo 眼在手外 RGB 手眼标定验证

本目录提供一个仿真的 `eye_to_hand_rgb_v4l2` 采集、标定和误差验证程序。机械臂使用 GitHub 上的 [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) 中的 Franka Emika Panda MJCF 模型；首次运行会自动下载 `franka_emika_panda` 到 `third_party/`。

程序会在 Panda 法兰/手爪末端刚性安装一个 AprilTag 标定板，在工作空间中放置一个固定 RGB 相机，并通过浏览器提供实时图像、标签检测有效性、关节滑块、随机姿态、采集和求解按钮。

网页里的默认求解方式是 `--solve-source truth`：采集时仍要求图像中的 AprilTag 检测有效，但手眼求解使用 MuJoCo 保存的真值 `T_camera_target`，用于验证手眼数学链路和误差计算。若要专门测试“渲染图像 -> AprilTag PnP -> 手眼”的完整视觉链路，可用 `--solve-source image` 或父目录离线求解器。

## 安装依赖

```bash
cd /media/chenzhaoqi/data/tmp/hand_eye_calib/eye_to_hand_rgb_v4l2/mujoco_sim
python -m pip install -r requirements.txt
```

如果在无显示服务器上运行 MuJoCo，可先设置：

```bash
export MUJOCO_GL=egl
```

## 启动采集界面

```bash
python sim_eye_to_hand.py --host 127.0.0.1 --port 8088
```

启动后打开：

```text
http://127.0.0.1:8088
```

网页左侧是实时渲染图像，右侧会显示：

- 当前 AprilTag 是否有效；
- PnP 重投影 RMS；
- 标签像素面积；
- 已采集样本数量；
- 7 个关节目标滑块；
- `随机姿态`、`采集当前帧`、`求解标定` 按钮。

建议采集 15-30 组姿态。每次先点 `随机姿态` 或手动调整关节，等画面中标签完整且状态为 `有效` 后再点 `采集当前帧`。姿态要覆盖不同位置和不同旋转角度，否则手眼约束会退化。

## 输出数据

默认数据保存到：

```text
eye_to_hand_rgb_v4l2/mujoco_sim/data/YYYY-mm-dd_HHMMSS/
```

每个样本包含：

```text
frame_000.png          # 固定相机看到的 RGB 图像
pose_000.json          # MuJoCo 真值 T_base_flange
target_000.json        # MuJoCo 真值 T_camera_target
intrinsics.json        # 仿真相机内参
session.json           # 会话元数据
```

求解后会生成：

```text
T_base_camera.json
ground_truth_T_base_camera.json
```

`T_base_camera.json` 是手眼标定得到的结果；`ground_truth_T_base_camera.json` 是 MuJoCo 固定相机的真值。网页点击 `求解标定` 后会额外写入 `ground_truth_error`，包含平移误差毫米值和旋转误差角度值。

## 离线重新求解

采集完成后，也可以复用真实相机目录里的离线求解器：

```bash
cd /media/chenzhaoqi/data/tmp/hand_eye_calib/eye_to_hand_rgb_v4l2
python calibrate_from_data.py mujoco_sim/data/你的会话目录 --method PARK
```

或者直接在网页里点击 `求解标定`。

## 常用参数

```bash
python sim_eye_to_hand.py \
  --width 1280 --height 720 \
  --tag-size-mm 90 \
  --min-tag-area-px 700 \
  --max-reprojection-px 2.0 \
  --method PARK \
  --solve-source truth
```

- `--tag-size-mm`：仿真标定板边长，单位 mm。
- `--min-tag-area-px`：标签面积低于该值时判定为无效。
- `--max-reprojection-px`：PnP 重投影 RMS 高于该值时判定为无效。
- `--output-dir`：指定采集输出目录。
- `--host 0.0.0.0`：允许局域网其他设备访问网页。该页面没有认证，只建议在可信网络使用。
- `--solve-source truth`：使用 MuJoCo 真值 `T_camera_target` 求解，适合验证手眼链路和真值误差。
- `--solve-source image`：使用采集图像中的 AprilTag PnP 结果求解，适合测试视觉检测链路；仿真渲染存在采样、遮挡和姿态歧义，误差通常会比真值模式大。

## 坐标系

标定目标与真实相机版保持一致：

```text
T_base_camera：OpenCV 相机坐标系 -> MuJoCo 世界/机器人基座坐标系
```

采集时保存的机器人位姿是：

```text
T_base_flange
```

其中 `flange` 在本仿真里使用 Panda 的 `hand` 刚体坐标系。验证约束仍然是：

```text
T_flange_target = inv(T_base_flange) @ T_base_camera @ T_camera_target
```

因为标定板刚性安装在法兰上，所有样本计算出的 `T_flange_target` 应保持一致。

## 第三方模型来源

- 模型仓库：[google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- 使用模型：`franka_emika_panda`
- 许可：请查看首次下载后的 `third_party/franka_emika_panda/LICENSE`

本目录不会把第三方 mesh 直接提交进当前项目；运行时自动下载，便于保持来源清晰。
