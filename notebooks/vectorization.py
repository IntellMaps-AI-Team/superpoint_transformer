import laspy
import numpy as np
import matplotlib.pyplot as plt
import svgwrite
import os
from sklearn.cluster import DBSCAN


def extract_walls(las_file):
    """Extracts wall points from a LAS file."""
    print("Loading LAS file...")
    las = laspy.read(las_file)
    print("Extracting wall points...")
    mask = las.classification == 2  # Walls in S3DIS dataset
    return las.x[mask], las.y[mask]


def cluster_points(x, y, eps=0.5, min_samples=10):
    """Clusters points to reduce noise and create structured wall segments."""
    print("Clustering points...")
    points = np.column_stack((x, y))
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    print(f"Found {len(set(clustering.labels_)) - 1} clusters (excluding noise)")
    return points, clustering.labels_


def create_svg(x, y, labels, output_file="floorplan.svg"):
    """Creates an SVG floorplan from clustered wall points."""
    print("Creating SVG floorplan...")
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    width, height = max_x - min_x, max_y - min_y

    dwg = svgwrite.Drawing(output_file, size=(width, height))

    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            continue  # Ignore noise
        mask = labels == label
        dwg.add(dwg.polyline(points=[(x[i] - min_x, height - (y[i] - min_y)) for i in range(len(x)) if mask[i]],
                             stroke="black", fill="none", stroke_width=2))

    dwg.save()
    print(f"SVG floorplan saved as {output_file}")


def generate_floorplan(las_file, output_svg="floorplan.svg"):
    print("Starting floorplan generation...")
    x, y = extract_walls(las_file)
    print(f"Extracted {len(x)} wall points.")
    points, labels = cluster_points(x, y)
    print("Generating SVG...")
    create_svg(points[:, 0], points[:, 1], labels, output_svg)
    print("Floorplan generation completed.")
    visualize_walls(points[:, 0], points[:, 1], labels)


def visualize_walls(x, y, labels):
    """Generates a PNG visualization for verification."""
    print("Creating visualization PNG...")
    plt.figure(figsize=(10, 10))
    unique_labels = set(labels)
    for label in unique_labels:
        mask = labels == label
        plt.scatter(x[mask], y[mask], s=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Wall Extraction Verification")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.savefig("walls_verification.png", dpi=300)
    plt.close()
    print("Visualization PNG saved as walls_verification.png")


if __name__ == "__main__":
    filepath = '/home/data/assets/poly/NP2_cast1_semantic.las'
    generate_floorplan(filepath)
