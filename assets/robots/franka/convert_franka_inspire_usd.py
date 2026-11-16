"""把 franka_inspire.urdf 转成 USD（保留 PhysX mimic 关节）。

为什么不用 IsaacLab 自带的 scripts/tools/convert_urdf.py:
    该脚本没有暴露 mimic 选项，导入器默认不创建 PhysxMimicJointAPI，
    手指远端关节 (*_2) 和拇指 thumb_3/thumb_4 会变成无驱动的自由关节。
    注意: IsaacLab 的 UrdfConverterCfg.convert_mimic_joints_to_normal_joints
    会被直接传给 Isaac Sim 的 set_parse_mimic()，而 parse_mimic=True 的含义是
    "创建 PhysX Mimic Joint"，所以这里要设为 True（名字和实际含义相反）。

用法:
    conda activate grasp
    cd /home/windsky/project/Grasp2
    python assets/robots/franka/build_franka_inspire_urdf.py      # 先生成 URDF
    python assets/robots/franka/convert_franka_inspire_usd.py --headless
"""

import argparse
import os

from isaaclab.app import AppLauncher

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Convert franka_inspire.urdf to USD.")
parser.add_argument("--urdf", default=os.path.join(HERE, "franka_inspire", "franka_inspire.urdf"))
parser.add_argument("--usd", default=os.path.join(HERE, "franka_inspire", "franka_inspire.usd"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

from pxr import PhysxSchema, Usd  # noqa: E402

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402


def main():
    cfg = UrdfConverterCfg(
        asset_path=os.path.abspath(args_cli.urdf),
        usd_dir=os.path.dirname(os.path.abspath(args_cli.usd)),
        usd_file_name=os.path.basename(args_cli.usd),
        fix_base=True,
        merge_fixed_joints=False,  # 保留 rh_*_tip / rh_*_pad 等 link，方便做指尖奖励和接触传感器
        convert_mimic_joints_to_normal_joints=True,  # => parse_mimic=True => 创建 PhysxMimicJointAPI
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),  # 增益由 ArticulationCfg 的 actuators 设置
            target_type="position",
        ),
    )
    converter = UrdfConverter(cfg)
    print(f"Generated USD: {converter.usd_path}")
    fix_mimic_offsets(converter.usd_path, cfg.asset_path)


def fix_mimic_offsets(usd_path: str, urdf_path: str):
    """修正 Isaac Sim 5.1 URDF 导入器写错的 mimic offset。

    PhysX mimic 约束: q + gearing * q_ref + offset = 0，旋转关节的 offset 单位是「度」。
    URDF: q = multiplier * q_ref + offset_urdf(弧度)
    => gearing = -multiplier（导入器已正确写入）
       offset  = -offset_urdf 换算成度（导入器直接写了 offset_urdf 弧度数值，符号和单位都不对）
    实测（未修正）: index_2 = 1.1169 * index_1 + 0.0026，即 offset 只有 0.15°，等于没生效。
    """
    import math
    import xml.etree.ElementTree as ET

    urdf_offsets = {}
    for joint in ET.parse(urdf_path).getroot().findall("joint"):
        mimic = joint.find("mimic")
        if mimic is not None:
            urdf_offsets[joint.get("name")] = float(mimic.get("offset", "0"))

    stage = Usd.Stage.Open(usd_path)
    for prim in stage.Traverse():
        for schema in prim.GetAppliedSchemas():
            if not schema.startswith("PhysxMimicJointAPI:"):
                continue
            axis = schema.split(":", 1)[1]
            api = PhysxSchema.PhysxMimicJointAPI(prim, axis)
            offset_attr = api.GetOffsetAttr()
            # 该属性定义在被引用的子图层（configuration/*_physics.usd）里，直接改那个图层中的属性 spec
            spec = offset_attr.GetPropertyStack()[0]
            spec.default = -math.degrees(urdf_offsets.get(prim.GetName(), 0.0))
            spec.layer.Save()
            stage.Reload()
            ref = api.GetReferenceJointRel().GetTargets()
            print(
                f"  mimic {prim.GetName():20s} -> {ref[0].name if ref else '?':20s}"
                f" gearing={api.GetGearingAttr().Get():+.4f} offset={offset_attr.Get():+.4f} deg"
            )


if __name__ == "__main__":
    main()
    # Isaac Sim 5.1 下 simulation_app.close() 偶尔会卡住，转换完成后直接退出
    os._exit(0)
