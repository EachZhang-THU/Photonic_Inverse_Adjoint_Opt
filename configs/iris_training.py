"""iris_training 任务参数。"""

from src.config import (DatasetConfig, FDTDConfig, MaterialConfig, ObjectSpec,
                        OptimizerConfig, QuantConfig, RegionConfig, TaskConfig)


def make_config() -> TaskConfig:
    return TaskConfig(
        name="iris_training",
        region=RegionConfig(),
        fdtd=FDTDConfig(fdtd_dimension="2D", fdtd_gui=True,
                        device="CPU", express_mode=0),
        optimizer=OptimizerConfig(max_epoch=1000),
        quant=QuantConfig(),
        material=MaterialConfig(),
        objects=[ObjectSpec(
            filename="obj_1_3d_coherent_fom.lsf",
            fom_name=["fom_1", "fom_2", "fom_3"],
            mode_exp_monitor_name=["fom_exp_1", "fom_exp_2", "fom_exp_3"],
            forward_source_name=["forward_source_1", "forward_source_2",
                                 "forward_source_3", "forward_source_4"],
            adjoint_source_name=["adjoint_source_1", "adjoint_source_2",
                                 "adjoint_source_3"],
            wavelength=1550e-9,
            forward_source_phase=[0, 0, 0, 0],
            forward_source_amp=[1, 0, 0, 0],
            target_fom=[0, 1, 0],
            weight=[1, 1, 1],
        )],
        dataset=DatasetConfig(),
        output_dir="sim_100nm",
    )
