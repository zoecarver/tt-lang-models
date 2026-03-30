"""Render micelle trajectory to mp4 using Ovito Python API + ffmpeg."""
from ovito.io import import_file
from ovito.vis import Viewport, TachyonRenderer
import sys
import subprocess
import os
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/micelle_traj.xyz"
outmp4 = sys.argv[2] if len(sys.argv) > 2 else "/tmp/micelle.mp4"

pipeline = import_file(path, columns=["Particle Type", "Position.X", "Position.Y", "Position.Z"])

# Color and size by type — green tails, red heads, small dark water
def assign_colors_and_radii(frame, data):
    types = data.particles["Particle Type"]
    n = len(types)
    colors = data.particles_.create_property("Color", data=np.tile([0.5, 0.5, 0.5], (n, 1)))
    radii = data.particles_.create_property("Radius", data=np.full(n, 0.5))
    type_colors = {
        "HEAD": [0.9, 0.15, 0.15],
        "TAIL": [0.2, 0.75, 0.2],
        "WATER": [0.35, 0.4, 0.5],
        "NA": [1.0, 0.8, 0.0],
        "CL": [0.0, 1.0, 0.3],
        "CA": [0.8, 0.0, 0.8],
    }
    type_radii = {
        "HEAD": 0.55,
        "TAIL": 0.4,
        "WATER": 0.15,
    }
    for i, t in enumerate(types):
        name = data.particles.particle_types.type_by_id(t).name
        if name in type_colors:
            colors[i] = type_colors[name]
        if name in type_radii:
            radii[i] = type_radii[name]

pipeline.modifiers.append(assign_colors_and_radii)

# Keep water visible (small dark beads provide context for clustering)

# Compute to get box info
data = pipeline.compute()
n_frames = pipeline.source.num_frames
print(f"Loaded {n_frames} frames from {path}")

pipeline.add_to_scene()

# Camera looking at box center
cell = data.cell
if cell is not None:
    box_size = max(cell[0, 0], cell[1, 1], cell[2, 2])
else:
    pos = data.particles.positions
    box_size = np.max(pos) - np.min(pos)
center = box_size / 2.0

vp = Viewport(type=Viewport.Type.Perspective)
vp.camera_pos = (center, center, box_size * 2.2)
vp.camera_dir = (0, 0, -1)
vp.fov = 0.65

# Render frames to temp dir
tmpdir = "/tmp/micelle_frames"
os.makedirs(tmpdir, exist_ok=True)

for frame in range(n_frames):
    outfile = f"{tmpdir}/frame_{frame:04d}.png"
    vp.render_image(size=(1280, 960), filename=outfile, frame=frame,
                    renderer=TachyonRenderer())
    print(f"  Rendered frame {frame}/{n_frames-1}")

# Encode to mp4
fps = max(1, n_frames // 10)  # ~10 second video
fps = min(fps, 30)
cmd = [
    "ffmpeg", "-y",
    "-framerate", str(fps),
    "-i", f"{tmpdir}/frame_%04d.png",
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
    outmp4,
]
print(f"\nEncoding {n_frames} frames at {fps} fps -> {outmp4}")
subprocess.run(cmd, check=True, capture_output=True)
print(f"Done: {outmp4}")
