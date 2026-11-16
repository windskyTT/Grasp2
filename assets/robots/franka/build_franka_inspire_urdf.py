"""生成 Franka Panda 机械臂 + Inspire RH56DFX 右手 的合成 URDF。

用法（在 conda grasp 环境中，需要 pip install xacro）:
    python assets/robots/franka/build_franka_inspire_urdf.py

输出: assets/robots/franka/franka_inspire/franka_inspire.urdf

处理内容:
1. 用 xacro 展开 rh56dfx_macro.xacro（side=right, prefix=rh_）。
2. 读取 Isaac Sim 自带的 panda_arm.urdf（不带二指夹爪）。
3. 用固定关节 panda_hand_mount 把 rh_hand_root 接到 panda_link8 上
   （绕 Z 轴 -45°，与原 Panda 夹爪安装方式一致）。
4. 把灵巧手的两个腕部关节 rh_wrist_yaw_joint / rh_hand_base_joint 改为 fixed
   （RH56DFX 真机只有 6 个手指自由度，腕部关节由 Franka 提供）。
5. 把 package:// 网格路径改写为相对于输出 URDF 的相对路径。
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET

import xacro

HERE = os.path.dirname(os.path.abspath(__file__))
HAND_PKG = os.path.join(HERE, "rh56dfx_description")
ARM_PKG = os.path.join(HERE, "franka_description")

HAND_PREFIX = "rh_"
HAND_SIDE = "right"
FIXED_HAND_JOINTS = [f"{HAND_PREFIX}wrist_yaw_joint", f"{HAND_PREFIX}hand_base_joint"]
MOUNT_RPY = (0.0, 0.0, -math.pi / 4)  # 和 panda_hand_joint 一致
MOUNT_XYZ = (0.0, 0.0, 0.0)


def expand_hand_xacro() -> ET.Element:
    """展开灵巧手 xacro，返回 <robot> 根节点。"""
    wrapper = f"""<?xml version="1.0"?>
<robot name="rh56dfx" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:include filename="{os.path.join(HAND_PKG, 'urdf', 'rh56dfx_macro.xacro')}"/>
  <xacro:rh56dfx_hand side="{HAND_SIDE}" prefix="{HAND_PREFIX}"/>
</robot>
"""
    doc = xacro.parse(wrapper)
    xacro.process_doc(doc)
    return ET.fromstring(doc.toxml())


def rewrite_mesh_paths(root: ET.Element, out_dir: str):
    packages = {"rh56dfx_description": HAND_PKG, "franka_description": ARM_PKG}
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        if not fn.startswith("package://"):
            continue
        pkg, rel = fn[len("package://"):].split("/", 1)
        abs_path = os.path.join(packages[pkg], rel)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"网格文件不存在: {abs_path}")
        mesh.set("filename", os.path.relpath(abs_path, out_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(HERE, "franka_inspire", "franka_inspire.urdf"))
    args = parser.parse_args()
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    arm = ET.parse(os.path.join(ARM_PKG, "panda_arm.urdf")).getroot()
    hand = expand_hand_xacro()

    robot = ET.Element("robot", {"name": "franka_inspire"})
    for child in list(arm):
        robot.append(child)
    for child in list(hand):
        robot.append(child)

    # 安装关节
    mount = ET.SubElement(robot, "joint", {"name": "panda_hand_mount", "type": "fixed"})
    ET.SubElement(mount, "parent", {"link": "panda_link8"})
    ET.SubElement(mount, "child", {"link": f"{HAND_PREFIX}hand_root"})
    ET.SubElement(
        mount,
        "origin",
        {"xyz": " ".join(f"{v:g}" for v in MOUNT_XYZ), "rpy": " ".join(f"{v:.12g}" for v in MOUNT_RPY)},
    )

    # 腕部关节固定
    for joint in robot.iter("joint"):
        if joint.get("name") in FIXED_HAND_JOINTS:
            joint.set("type", "fixed")
            for tag in ("axis", "limit", "dynamics", "mimic"):
                for el in joint.findall(tag):
                    joint.remove(el)

    rewrite_mesh_paths(robot, out_dir)

    # 简单检查: 名称不重复，每个 link 只有一个父关节
    links = [l.get("name") for l in robot.findall("link")]
    joints = [j.get("name") for j in robot.findall("joint")]
    assert len(links) == len(set(links)), "link 名称重复"
    assert len(joints) == len(set(joints)), "joint 名称重复"
    children = [j.find("child").get("link") for j in robot.findall("joint")]
    assert len(children) == len(set(children)), "某个 link 有多个父关节"
    roots = set(links) - set(children)
    assert roots == {"panda_link0"}, f"根 link 不唯一: {roots}"

    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(args.out, encoding="utf-8", xml_declaration=True)

    movable = [(j.get("name"), j.get("type"), j.find("mimic") is not None) for j in robot.findall("joint") if j.get("type") != "fixed"]
    print(f"已写出: {args.out}")
    print(f"links: {len(links)}, joints: {len(joints)}, 可动关节: {len(movable)}")
    for name, jtype, is_mimic in movable:
        print(f"  {name:28s} {jtype:10s} {'(mimic)' if is_mimic else ''}")


if __name__ == "__main__":
    main()
