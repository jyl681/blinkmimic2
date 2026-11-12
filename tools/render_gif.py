"""Render recorded evaluation rollouts (tools/record_clips.py) as a looping grid GIF / MP4 with MuJoCo (EGL).

Each tile plays one clip: the simulated robot solid, the reference motion a translucent blue ghost anchored to the robot's
xy (as the tracking reward compares them). The camera follows the robot at a fixed distance and height; its azimuth is
set from the robot's heading at the tile's first frame.

    MUJOCO_GL=egl python tools/render_gif.py traj.npz out.gif --cols 2 --seconds 6 \
        --clip flip_360_002__A305:0.5:backflip --clip high_jump_R_002__A441:1.0:"high jump"

--clip NAME[:START_S[:LABEL]]; every tile shows --seconds from its start time (shorter clips hold their last frame).
NAME may be OTHER.npz@NAME to take that tile from another recording.
"""

import argparse
import os
import re
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from blinkmimic2.robot import get_robot

FPS_SIM = 50


def scene_xml(robot, ghost_alpha: float) -> str:
    xml = open(robot.mjcf_path).read()
    mjcf_dir = os.path.dirname(os.path.abspath(robot.mjcf_path))
    xml = re.sub(r'meshdir="([^"]*)"', lambda m: 'meshdir="' + os.path.join(mjcf_dir, m.group(1)) + '/"', xml)
    body = xml[xml.index(f'<body name="{robot.tracked_bodies[0]}"') : xml.index("</worldbody>")]
    ghost = re.sub(r' name="([^"]+)"', r' name="ref_\1"', body)
    ghost = re.sub(r' (contype|conaffinity|group)="[^"]*"', "", ghost)
    ghost = ghost.replace("<geom ", '<geom contype="0" conaffinity="0" group="1" ')
    scene = """
    <light pos="0 0 6" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="0" conaffinity="0"/>
    """
    xml = xml.replace("</worldbody>", ghost + scene + "</worldbody>")
    assets = """
    <texture name="grid" type="2d" builtin="checker" rgb1="0.965 0.965 0.972" rgb2="0.925 0.925 0.935" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.03" emission="0.75"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.96 0.96 0.97" width="256" height="256"/>
    """
    xml = xml.replace("<asset>", "<asset>" + assets, 1)
    if "<visual>" not in xml:
        xml = xml.replace("<asset>", '<visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/></visual>\n  <asset>', 1)
    return xml


def mat_to_quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1).astype(np.float64))
    return q


def yaw_of_quat(q):  # wxyz
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("out", help=".gif or .mp4")
    ap.add_argument("--clip", action="append", required=True, help="NAME[:START_S[:LABEL]]")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=320, help="tile width")
    ap.add_argument("--height", type=int, default=400, help="tile height")
    ap.add_argument("--distance", type=float, default=3.0)
    ap.add_argument("--elevation", type=float, default=-10.0)
    ap.add_argument("--az_rel", type=float, default=140.0)
    ap.add_argument("--lookat_z", type=float, default=0.7)
    ap.add_argument("--ghost_alpha", type=float, default=0.3)
    ap.add_argument("--robot", default="mjlab_g1")
    ap.add_argument("--colors", type=int, default=96, help="GIF palette size")
    a = ap.parse_args()

    d = dict(np.load(a.traj))
    names = [str(n) for n in d["names"]]
    robot = get_robot(a.robot)
    nq = 7 + robot.num_joints
    model = mujoco.MjModel.from_xml_string(scene_xml(robot, a.ghost_alpha))
    for g in range(model.ngeom):  # reference meshes translucent blue without material, reference collision primitives hidden
        b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        if b.startswith("ref_"):
            model.geom_matid[g] = -1
            model.geom_rgba[g] = (0.35, 0.65, 1.0, a.ghost_alpha if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else 0.0)
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.55
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, a.height, a.width)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    stride = FPS_SIM // a.fps if FPS_SIM % a.fps == 0 else None
    assert stride, f"--fps must divide {FPS_SIM}"
    n_frames = int(round(a.seconds * a.fps))
    tiles = []
    d0, names0 = d, names
    for spec in a.clip:
        name, *rest = spec.split(":")
        d, names = d0, names0
        if "@" in name:  # a clip from another recording: OTHER.npz@NAME
            npz, name = name.split("@")
            d = dict(np.load(npz))
            names = [str(n) for n in d["names"]]
        t0 = float(rest[0]) if rest else 0.0
        label = rest[1] if len(rest) > 1 else name
        c = names.index(name)
        L = int(d["clip_len"][c])
        alive = d["alive"][:, c]
        dead = np.flatnonzero(~alive[:L])
        last = int(dead[0]) - 1 if len(dead) else min(L, alive.shape[0]) - 1
        s0 = int(round(t0 * FPS_SIM))
        steps = [min(s0 + k * stride, last) for k in range(n_frames)]
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation = a.distance, a.elevation
        cam.azimuth = np.degrees(yaw_of_quat(d["root_quat"][steps[0], c])) + a.az_rel
        frames = []
        for t in steps:
            f = int(d["f"][t, c])
            data.qpos[0:3] = d["root_p"][t, c]
            data.qpos[3:7] = d["root_quat"][t, c]
            data.qpos[7:nq] = d["q"][t, c]
            data.qpos[nq : nq + 3] = d["ref_root_p"][f]
            data.qpos[nq : nq + 2] = d["root_p"][t, c][:2]
            data.qpos[nq + 3 : nq + 7] = mat_to_quat(d["ref_root_R"][f])
            data.qpos[nq + 7 : 2 * nq] = d["ref_q"][f]
            mujoco.mj_kinematics(model, data)
            cam.lookat[:] = [data.qpos[0], data.qpos[1], a.lookat_z]
            renderer.update_scene(data, camera=cam)
            im = Image.fromarray(renderer.render())
            ImageDraw.Draw(im).text((10, a.height - 24), label, fill=(70, 70, 80), font=font)
            frames.append(im)
        tiles.append(frames)
        print(f"{name}: frames {steps[0]}..{steps[-1]} of {L}", flush=True)

    rows = -(-len(tiles) // a.cols)
    out = []
    for k in range(n_frames):
        canvas = Image.new("RGB", (a.cols * a.width, rows * a.height), "white")
        for i, fr in enumerate(tiles):
            canvas.paste(fr[k], ((i % a.cols) * a.width, (i // a.cols) * a.height))
        out.append(canvas)
    if a.out.endswith(".gif"):
        # one palette for all frames, from a strip of frames spread over the clip (robot grey and ghost blue both kept)
        picks = out[:: max(1, len(out) // 8)]
        strip = Image.new("RGB", (picks[0].width, picks[0].height * len(picks)))
        for i, im in enumerate(picks):
            strip.paste(im, (0, i * im.height))
        pal = strip.quantize(colors=a.colors, method=Image.Quantize.MEDIANCUT)
        q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in out]
        q[0].save(a.out, save_all=True, append_images=q[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
    else:
        W, H = out[0].size
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                              "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out],
                             stdin=subprocess.PIPE)  # fmt: skip
        for im in out:
            p.stdin.write(im.tobytes())
        p.stdin.close()
        p.wait()
    print(f"{a.out}: {len(out)} frames, {os.path.getsize(a.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
