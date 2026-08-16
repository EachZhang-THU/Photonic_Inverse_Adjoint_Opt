"""机器相关环境与项目路径：所有 import 路径副作用集中在此。"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_PATH / "src"
DATASET_PATH = PROJECT_PATH / "dataset"
LSF_PATH = PROJECT_PATH / "lsf"
RES_PATH = PROJECT_PATH / "opt_results"
DRC_PATH = PROJECT_PATH / "drc"

# 允许通过环境变量覆盖，便于换机器/服务器时无需改代码
LUMERICAL_API_PATH = os.environ.get(
    "LUMERICAL_API_PATH", r"D:\Program Files\Lumerical\v241\api\python")
KLAYOUT_PATH = os.environ.get(
    "KLAYOUT_PATH", r"D:\Program Files\Klayout\klayout_app.exe")


def configure() -> None:
    """注入 Lumerical API 与 src 目录到 sys.path（必须在 import lumapi 前调用）。"""
    for path in (str(LUMERICAL_API_PATH), str(SRC_PATH)):
        if path not in sys.path:
            sys.path.append(path)


def create_new_sim_directory(base_path, new_path):
    """在 base_path 下创建 new_path_N，返回新目录（与原行为一致）。"""
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"Created base directory: {base_path}")

    sim_number = 1
    while True:
        new_directory = os.path.join(base_path, f"{new_path}_{sim_number}")
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
            print(f"New directory created: {new_directory}")
            return new_directory
        sim_number += 1


@dataclass
class Workspace:
    """一次运行的输出目录。"""

    save_path: str
    sub_path: str


def create_workspace(output_dir: str = "sim_100nm") -> Workspace:
    save_path = create_new_sim_directory(RES_PATH, output_dir)
    sub_path = os.path.join(save_path, "sim_file")
    os.makedirs(sub_path, exist_ok=True)
    return Workspace(save_path, sub_path)
