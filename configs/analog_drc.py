"""analog_drc 任务参数。"""

from src.config import (DRCConfig, FDTDConfig, MaterialConfig, ObjectSpec,
                        OptimizerConfig, QuantConfig, RegionConfig, TaskConfig)


def make_config() -> TaskConfig:
    return TaskConfig(
        name="analog_drc",
        region=RegionConfig(size_x=4e-6 , pixel_size=20e-9, dz_ratio=5.0),
        fdtd=FDTDConfig(fdtd_dimension="2D", fdtd_gui=True,
                        device="CPU", express_mode=0),
        optimizer=OptimizerConfig(max_epoch=1000),
        quant=QuantConfig(beta_max=200.0),
        material=MaterialConfig(),
        objects=[ObjectSpec(
            filename="obj_1_m.lsf",
            fom_name=["fom_1", "fom_2", "fom_3"],
            mode_exp_monitor_name=["fom_exp_1", "fom_exp_2", "fom_exp_3"],
            forward_source_name=["forward_source_1"],
            adjoint_source_name=["adjoint_source_1", "adjoint_source_2",
                                 "adjoint_source_3"],
            wavelength=1550e-9,
            forward_source_phase=[0, 0, 0],
            forward_source_amp=[1, 0, 0],
            target_fom=[0, 0, 1],
            weight=[1, 1, 1],
        )],
        drc=DRCConfig(),
        output_dir="sim_100nm",
    )
