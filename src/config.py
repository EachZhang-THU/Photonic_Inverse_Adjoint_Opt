"""参数模型：静态配置（可安全替换/复用）与运行时状态分离。"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src import environment as env


@dataclass
class RegionConfig:
    size_x: float = 5e-6
    size_y: float = 6e-6
    size_z: float = 220e-9
    pixel_size: float = 100e-9
    z_spacing: float = 110e-9
    sim_spacing: float = 20e-9
    dz_ratio: Optional[float] = None    # analog 时 dz = dx / dz_ratio，且 V_cell = dx * dy


@dataclass
class FDTDConfig:
    fdtd_dimension: str = "3D"
    fdtd_gui: bool = True
    device: str = "GPU"
    express_mode: int = 1


@dataclass
class OptimizerConfig:
    max_epoch: int = 1000
    learning_rate: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    params_min: float = 0.0
    params_max: float = 1.0


@dataclass
class QuantConfig:
    eta: float = 0.5
    beta0: float = 1.0
    beta_max: float = 500.0
    test_length: int = 10
    threshold: float = 0.1


@dataclass
class MaterialConfig:
    eps_min: float = 1.44 ** 2
    eps_max: float = 3.47 ** 2


@dataclass
class ObjectSpec:
    filename: str
    fom_name: List[str]
    mode_exp_monitor_name: List[str]
    forward_source_name: List[str]
    adjoint_source_name: List[str]
    wavelength: float
    forward_source_phase: List[float]
    forward_source_amp: List[float]
    target_fom: List[float]
    weight: List[float]


@dataclass
class DatasetConfig:
    path: str = "iris_training_dataset.npz"
    batch_size: int = 12


@dataclass
class DRCConfig:
    klayout_path: str = field(default_factory=lambda: env.KLAYOUT_PATH)
    gds_name: str = "gds_result"
    drc_script: str = "drc_45nm.lydrc"
    drc_threshold: float = 0.05
    param_delta: float = 0.2
    filter_R_max: int = 8
    filter_R_min: int = 3
    filter_R0: int = 3


@dataclass
class TaskConfig:
    name: str
    region: RegionConfig
    fdtd: FDTDConfig
    optimizer: OptimizerConfig
    quant: QuantConfig
    material: MaterialConfig
    objects: List[ObjectSpec]
    output_dir: str = "sim_100nm"
    dataset: Optional[DatasetConfig] = None
    drc: Optional[DRCConfig] = None
    resume: Optional[str] = None


class Region:
    """设计区域与仿真网格（由 RegionConfig 派生）。"""

    def __init__(self, cfg: RegionConfig):
        analog = cfg.dz_ratio is not None

        self.size_x = cfg.size_x
        self.size_y = cfg.size_y
        self.size_z = cfg.size_z
        self.pixel_size = cfg.pixel_size

        self.z_points = int(self.size_z / cfg.z_spacing) + 1
        self.z_pos = np.linspace(-self.size_z / 2, self.size_z / 2, self.z_points)

        if analog:
            self.x_points = int(self.size_x / self.pixel_size) + 1
            self.y_points = int(self.size_y / self.pixel_size) + 1
            self.x_pos = np.linspace(-self.size_x / 2, self.size_x / 2, self.x_points)
            self.y_pos = np.linspace(-self.size_y / 2, self.size_y / 2, self.y_points)
            self.dx = self.x_pos[1] - self.x_pos[0]
            self.dy = self.y_pos[1] - self.y_pos[0]
            self.dz = self.dx / cfg.dz_ratio
            self.V_cell = self.dx * self.dy
        else:
            self.x_points = int(self.size_x / self.pixel_size)
            self.y_points = int(self.size_y / self.pixel_size)
            self.dx = self.pixel_size
            self.dy = self.pixel_size
            self.x_pos = np.linspace(-self.size_x / 2 + self.dx / 2,
                                     self.size_x / 2 - self.dx / 2, self.x_points)
            self.y_pos = np.linspace(-self.size_y / 2 + self.dy / 2,
                                     self.size_y / 2 - self.dy / 2, self.y_points)
            self.dz = self.z_pos[1] - self.z_pos[0]
            self.V_cell = self.dx * self.dy * self.dz

        self.sim_x_points = int(self.size_x / cfg.sim_spacing) + 1
        self.sim_y_points = int(self.size_y / cfg.sim_spacing) + 1
        self.sim_x_pos = np.linspace(-self.size_x / 2, self.size_x / 2, self.sim_x_points)
        self.sim_y_pos = np.linspace(-self.size_y / 2, self.size_y / 2, self.sim_y_points)


class Object:
    """一次优化中的 Lumerical 对象：静态描述 + 运行时字段。"""

    def __init__(self, spec: ObjectSpec):
        self.model = []
        self.filename = spec.filename
        self.fom_name = spec.fom_name
        self.fom = np.zeros((len(spec.fom_name)))
        self.factor = np.zeros((len(spec.fom_name)), dtype=np.complex64)
        self.mode_exp_monitor_name = spec.mode_exp_monitor_name
        self.forward_source_name = spec.forward_source_name
        self.adjoint_source_name = spec.adjoint_source_name
        self.forward_source_phase = spec.forward_source_phase
        self.forward_source_amp = spec.forward_source_amp
        self.target_fom = spec.target_fom
        self.weight = spec.weight
        self.wavelength = spec.wavelength
        self.E_for = None
        self.E_adj = None
        self.E_fom = None
        self.E_fom_desire_mode = None


@dataclass
class OptimState:
    """优化器运行时状态（可断点保存/恢复），不属于静态配置。"""

    eps_opt: np.ndarray
    index_opt: np.ndarray
    params: np.ndarray
    m: np.ndarray
    v: np.ndarray
    iteration: int
    beta: float
    filter_R: int = 3
    params_all: Optional[np.ndarray] = None
    params_conv: Optional[np.ndarray] = None
    delta_R: int = 0
    filter_kernel: Optional[np.ndarray] = None
    filter_kernel_inverse: Optional[np.ndarray] = None

    @classmethod
    def create(cls, region, cfg: TaskConfig,
               npz_path: Optional[str] = None) -> "OptimState":
        if npz_path is not None:
            guess = np.load(npz_path)
            eps_opt = guess["eps_save"]
            params = guess["params"]
            m = guess["m"]
            v = guess["v"]
            iteration = guess["all_iteration"]
            beta = float(guess["beta"])
            index_opt = np.real(np.sqrt(eps_opt))
        else:
            eps_initial = (0.5 * np.ones((region.y_points, region.x_points))
                           * (cfg.material.eps_max - cfg.material.eps_min)
                           + cfg.material.eps_min)
            params = ((eps_initial - cfg.material.eps_min)
                      / (cfg.material.eps_max - cfg.material.eps_min))
            eps_opt = eps_initial
            index_opt = np.sqrt(eps_initial)
            m = np.zeros_like(eps_opt)
            v = np.zeros_like(eps_opt)
            iteration = 0
            beta = cfg.quant.beta0

        return cls(
            eps_opt=eps_opt,
            index_opt=index_opt,
            params=params,
            m=m,
            v=v,
            iteration=iteration,
            beta=beta,
            filter_R=cfg.drc.filter_R0 if cfg.drc else 3,
        )


@dataclass
class History:
    """一次运行的曲线/收敛历史，与静态配置分离。"""

    all_fom: List[np.ndarray] = field(default_factory=list)
    all_beta: List[float] = field(default_factory=list)
    non_convergence: int = 0


def build_region(cfg: RegionConfig) -> Region:
    return Region(cfg)


def build_object(spec: ObjectSpec) -> Object:
    return Object(spec)
