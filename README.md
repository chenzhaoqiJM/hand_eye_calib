# 手眼标定与平面映射工具集

本项目提供从相机内参标定、标定板生成，到手眼标定和像素坐标映射的完整工具。各目录可独立使用，详细参数请查看对应目录的 `README.md`。

## 目录说明

| 目录 | 用途 |
| --- | --- |
| `monocular_rgb_calibration/` | 使用棋盘格标定单目 USB RGB 相机内参 |
| `apriltag_board_generator/` | 生成 AprilTag 36h11 标定板 PDF |
| `eye_in_hand/` | 眼在手上：计算相机到机械臂法兰的变换 |
| `eye_to_hand_rgb_v4l2/` | 眼在手外：计算相机到机器人基座的变换 |
| `planar_homography/` | 计算图像像素到二维工作平面的单应性映射 |

## 推荐使用顺序

1. 在 `monocular_rgb_calibration/` 中完成相机内参标定。
2. 根据需求生成并打印棋盘格或 AprilTag 标定板，打印时使用 100% 比例并核对实际尺寸。
3. 根据相机安装方式选择 `eye_in_hand/` 或 `eye_to_hand_rgb_v4l2/` 完成手眼标定。
4. 如需将像素坐标转换为工作平面坐标，使用 `planar_homography/`。

## 基本要求

- Python 依赖按各目录中的 `requirements.txt` 分别安装。
- 相机分辨率、内参文件与实际采集配置必须一致。
- 标定板应保持平整、刚性固定并完整可见。
- 标定结果投入使用前，应通过独立样本进行现场验证。

本项目采用 `LICENSE` 中声明的许可证。
