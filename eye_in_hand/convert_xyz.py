import json
import numpy as np

with open("T_flange_camera.json", encoding="utf-8") as f:
    T_flange_camera = np.array(json.load(f)["matrix_4x4"])

p_camera = np.array([0.0302, 0.0369, 0.1877, 1.0])  # 米
print(p_camera.shape)
p_flange = T_flange_camera @ p_camera

print(p_flange.shape)

print("法兰坐标:", *[f"{value:.4f}" for value in p_flange[:3]])
