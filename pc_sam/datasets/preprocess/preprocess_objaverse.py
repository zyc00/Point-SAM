import trimesh
import numpy as np

scene = trimesh.load(
    "/home/yuchen/workspace/pointcloud-sam/6f0d4f5e-1c3e-5b26-aec0-ca2219d36b2a.glb"
)
geometry = trimesh.base.Trimesh()
geometries = list(scene.geometry.values())
for g in geometries:
    geometry = geometry + g
geometry.export("test.obj")

# ccs = trimesh.graph.connected_component_labels(geometry.face_adjacency)
# print(np.unique(np.array(ccs)))
