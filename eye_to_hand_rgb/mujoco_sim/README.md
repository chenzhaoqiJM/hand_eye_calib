# MuJoCo 眼在手外 RGB 手眼标定验证

本目录提供棋盘格眼在手外标定的 MuJoCo 仿真程序。仿真中，棋盘格刚性安装在 Panda 法兰上，RGB 相机固定在工作空间中；浏览器界面提供实时图像、棋盘格角点和质量指标、关节控制、采集及求解功能。

默认 `--solve-source truth` 使用 MuJoCo 保存的 `T_camera_target` 真值求解，适合先验证手眼变换链路。使用 `--solve-source image` 可测试完整视觉流程：渲染图像 → 棋盘格检测 → PnP → 手眼求解。

机械臂使用 GitHub 上的 [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) 中的 Franka Emika Panda MJCF 模型；首次运行会自动下载 `franka_emika_panda` 到 `third_party/`。

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

网页里的 `求解标定` 会根据启动参数选择求解来源：

```bash
python sim_eye_to_hand.py --host 127.0.0.1 --port 8088 --solve-source truth   # 使用 MuJoCo 真值目标位姿
python sim_eye_to_hand.py --host 127.0.0.1 --port 8088 --solve-source image   # 使用棋盘格图像 PnP 位姿
```

父目录的 `calibrate_from_data.py` 已支持棋盘格数据，也可以对本目录生成的数据进行离线求解：

```bash
python ../calibrate_from_data.py \
  data/YYYY-mm-dd_HHMMSS \
  --method PARK
```

程序会优先从数据目录的 `session.json` 读取棋盘格列数、行数和方格尺寸。

启动后打开：

```text
http://127.0.0.1:8088
```

网页左侧是实时渲染图像，检测到棋盘格时会绘制内角点；右侧会显示：

- 当前棋盘格是否检测到及是否有效；
- PnP 重投影 RMS（像素）；
- 棋盘格覆盖率；
- 已采集样本数量；
- 7 个关节目标滑块；
- `随机姿态`、`采集当前帧`、`求解标定` 按钮。

只有当前画面满足有效性条件时才能采集。建议采集 20～30 组姿态。每次先点 `随机姿态` 或手动调整关节，等棋盘格完整、状态为 `有效` 且重投影 RMS 较低后再点 `采集当前帧`。姿态要覆盖不同位置和不同旋转角度，否则手眼约束会退化。

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


## 常用参数

```bash
python sim_eye_to_hand.py \
  --width 640 --height 480 \
  --chessboard-columns 7 \
  --chessboard-rows 6 \
  --square-size-mm 40 \
  --min-board-coverage 0.015 \
  --max-reprojection-px 2.0 \
  --method PARK \
  --solve-source truth
```

- `--chessboard-columns`：棋盘格横向内角点数量。
- `--chessboard-rows`：棋盘格纵向内角点数量。
- `--square-size-mm`：棋盘格单个方格边长，单位 mm。
- `--min-board-coverage`：棋盘格内角点包围区域覆盖率低于该值时判定为无效。
- `--max-reprojection-px`：PnP 重投影 RMS 高于该值时判定为无效。
- `--output-dir`：指定采集输出目录。
- `--host 0.0.0.0`：允许局域网其他设备访问网页。该页面没有认证，只建议在可信网络使用。
- `--solve-source truth`：使用 MuJoCo 真值 `T_camera_target` 求解，适合验证手眼链路和真值误差。
- `--solve-source image`：使用采集图像中的棋盘格 PnP 结果求解，适合测试视觉检测链路；仿真渲染存在采样、遮挡和姿态覆盖问题，误差通常会比真值模式大。

采集时若状态显示 `chessboard_not_found`，请确认内角点参数与棋盘格一致，并调整关节使棋盘格完整出现在相机画面中。`--min-board-coverage` 和 `--max-reprojection-px` 分别控制最小覆盖率和最大允许重投影误差。

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
