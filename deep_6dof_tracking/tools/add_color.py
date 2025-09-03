import trimesh
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Add color to a PLY file.')
parser.add_argument('-i', '--input', type=str, help='Input PLY file')
parser.add_argument('-o', '--output', type=str, help='Output PLY file')
args = parser.parse_args()

# Load the PLY file
mesh = trimesh.load(args.input)

# Check if the mesh has vertices
if not mesh.is_empty:
    # Create a gray color array (128, 128, 128 for RGB)
    gray_color = np.array([128, 128, 128, 255], dtype=np.uint8)  # Including alpha channel

    # Ensure the vertex count matches the color array
    vertex_colors = np.tile(gray_color, (len(mesh.vertices), 1))

    # Assign the gray colors to the vertices
    mesh.visual.vertex_colors = vertex_colors

    # Save the modified mesh to a new PLY file
    mesh.export(args.output)
else:
    print("The mesh is empty.")
