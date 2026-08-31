# 手眼标定与平面映射工具集

本项目提供一套可独立组合的相机与机器人标定工具，覆盖标定板生成、单目相机内参标定、眼在手上/眼在手外标定，以及像素到工作平面的二维映射。采集过程支持浏览器预览，标定数据可保存后离线求解和验证。

## 功能概览

| 目录 | 功能 | 主要输出 |
| --- | --- | --- |
| [`monocular_rgb_calibration/`](monocular_rgb_calibration/) | 使用棋盘格标定 Linux V4L2 单目 RGB 相机内参，并生成可打印棋盘格 | `intrinsics.json` |
| [`apriltag_board_generator/`](apriltag_board_generator/) | 生成具有准确物理尺寸的 AprilTag 36h11 标定板 PDF | `*.pdf` |
| [`eye_in_hand/`](eye_in_hand/) | 眼在手上标定；支持 RealSense D4xx、手动输入法兰位姿或从 ROS 2 TF 自动读取位姿 | `T_flange_camera.json` |
| [`eye_to_hand_rgb/`](eye_to_hand_rgb/) | 眼在手外标定；支持 V4L2 RGB 相机和 RealSense D4xx、棋盘格或 AprilTag | `T_base_camera.json` |
| [`planar_homography/`](planar_homography/) | 从单张图像或实时视频计算像素坐标到棋盘平面的单应性映射 | `pixel_to_plane_homography.json` |

矩阵命名采用 `T_target_source` 约定，即把 `source` 坐标系中的点转换到 `target` 坐标系。例如：

```text
p_base = T_base_camera @ p_camera
p_flange = T_flange_camera @ p_camera
```

## 如何选择流程

- 相机安装在机械臂末端：使用 [`eye_in_hand/`](eye_in_hand/README.md)，求取相机坐标系到法兰坐标系的变换。若机器人通过 ROS 2 发布法兰 TF，可使用 [ROS 2 TF 自动采集流程](eye_in_hand/README_ROS2_TF.md)。
- 相机固定在机器人外部：使用 [`eye_to_hand_rgb/`](eye_to_hand_rgb/README.md)，求取相机坐标系到机器人基座坐标系的变换。RealSense 用户另见 [RealSense 使用说明](eye_to_hand_rgb/README_REALSENSE.md)。
- 只需把图像坐标映射到固定工作平面：使用 [`planar_homography/`](planar_homography/README.md)。该结果只提供平面内的二维坐标，不会额外估计高度。

## 推荐流程

1. 准备标定板。可使用 [`monocular_rgb_calibration/generate_chessboard_pdf.py`](monocular_rgb_calibration/generate_chessboard_pdf.py) 生成棋盘格，或使用 [`apriltag_board_generator/generate_tag_pdf.py`](apriltag_board_generator/generate_tag_pdf.py) 生成 AprilTag。
2. 打印时选择 **100% / 实际大小**，关闭页面缩放；打印后测量方格边长或 AprilTag 最外侧黑框边长，并将标定板固定到平整、刚性的背板上。
3. V4L2 RGB 相机先使用 [`monocular_rgb_calibration/`](monocular_rgb_calibration/README.md) 标定内参。RealSense 流程可从设备读取当前彩色流内参。
4. 根据相机安装方式执行眼在手上或眼在手外标定。建议采集 20～30 组位置和旋转方向均有明显变化的样本。
5. 使用未参与求解的姿态或已知空间点验证结果，再将外参用于实际任务。
6. 如需固定平面的像素坐标映射，再执行 [`planar_homography/`](planar_homography/README.md)。相机、镜头或工作平面位置变化后必须重新标定。

## 安装

各模块依赖不同，进入所需目录后单独安装：

```bash
cd monocular_rgb_calibration
python -m pip install -r requirements.txt
```

建议为项目创建独立的 Python 虚拟环境。AprilTag 检测与手眼标定需要 `opencv-contrib-python`；RealSense 流程还需要 `pyrealsense2`，具体安装命令请查看对应模块文档。

## 使用前检查

- 相机分辨率必须与内参文件中的 `width`、`height` 完全一致；更换分辨率、焦距、变焦、对焦模式或镜头后应重新标定内参。
- 棋盘格参数使用的是**内角点数量**，不是黑白方格数量；所有物理尺寸参数必须与打印后的实测值一致。
- 标定板和相机在采集期间必须刚性固定，画面应清晰、完整、无明显反光或运动模糊。
- 机器人位姿方向、长度单位和角度单位必须与命令行参数一致。各模块默认值和坐标系约定以其 README 为准。
- 浏览器预览服务默认可能监听局域网接口且不提供认证，只应在可信网络中使用。
- 输出文件中的 `validated: false` 表示结果尚未通过独立现场验证，不应直接视为生产可用。

## 文档入口

- [单目 RGB 相机内参标定](monocular_rgb_calibration/README.md)
- [AprilTag 标定板生成](apriltag_board_generator/README.md)
- [眼在手上标定](eye_in_hand/README.md)
- [眼在手上：ROS 2 TF 自动采集](eye_in_hand/README_ROS2_TF.md)
- [眼在手外 V4L2 RGB 标定](eye_to_hand_rgb/README.md)
- [眼在手外 RealSense 标定](eye_to_hand_rgb/README_REALSENSE.md)
- [像素到二维平面映射](planar_homography/README.md)

本项目采用 [LICENSE](LICENSE) 中声明的许可证。
