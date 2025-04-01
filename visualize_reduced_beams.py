import numpy as np
import open3d as o3d
import argparse
import os


def load_kitti_bin(file_path):
    """Load a KITTI .bin file into a NumPy array."""
    return np.fromfile(file_path, dtype=np.float32).reshape(
        -1, 4
    )  # KITTI format: x, y, z, intensity


def save_kitti_bin(file_path, point_cloud):
    """Save a NumPy array as a KITTI .bin file."""
    point_cloud.astype(np.float32).tofile(file_path)


def remove_half_beams(points, num_bins=64):
    """
    Remove every alternate beam using vertical angle binning.
    For LiDAR point clouds, beams are better approximated by their vertical angle rather than the z coordinate directly.
    """
    # Compute the vertical angle (in radians) for each point
    angles = np.arctan2(points[:, 2], np.sqrt(points[:, 0]**2 + points[:, 1]**2))
    # Create equally spaced bins in the angle space corresponding to the number of beams
    bins = np.linspace(angles.min(), angles.max(), num_bins + 1)

    filtered_points = []
    for i in range(num_bins):
        if i % 2 == 0:  # Retain every alternate beam
            # Use <= for the last bin to include the max angle value
            if i == num_bins - 1:
                mask = (angles >= bins[i]) & (angles <= bins[i + 1])
            else:
                mask = (angles >= bins[i]) & (angles < bins[i + 1])
            filtered_points.append(points[mask])

    return np.vstack(filtered_points) if filtered_points else points


def visualize_and_save_point_clouds(original, modified, output_dir):
    """Visualize original and modified point clouds side by side and save as images."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create point clouds
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(original[:, :3])

    modified_pcd = o3d.geometry.PointCloud()
    modified_pcd.points = o3d.utility.Vector3dVector(modified[:, :3])

    # Assign colors for differentiation
    original_pcd.paint_uniform_color([1, 0, 0])  # Red for original
    modified_pcd.paint_uniform_color([0, 1, 0])  # Green for modified

    # Save original point cloud
    original_filename = os.path.join(output_dir, "original_pointcloud.ply")
    o3d.io.write_point_cloud(original_filename, original_pcd)
    print(f"Original point cloud saved to {original_filename}")
    
    # Save modified point cloud
    modified_filename = os.path.join(output_dir, "modified_pointcloud.ply")
    o3d.io.write_point_cloud(modified_filename, modified_pcd)
    print(f"Modified point cloud saved to {modified_filename}")
    
    # Save combined point cloud (side by side)
    # Compute translation offset based on the bounding box width of the original point cloud
    bbox = original_pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()  # [width, height, depth]
    translation = np.array([extent[0] * 1.5, 0, 0])  # Shift modified point cloud to the right
    modified_pcd_shifted = o3d.geometry.PointCloud(modified_pcd)
    modified_pcd_shifted.translate(translation)
    
    # Create combined point cloud
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(
        np.vstack([
            np.asarray(original_pcd.points),
            np.asarray(modified_pcd_shifted.points)
        ])
    )
    
    # Set colors for combined point cloud
    colors = np.vstack([
        np.full((len(original_pcd.points), 3), [1, 0, 0]),  # Red for original
        np.full((len(modified_pcd_shifted.points), 3), [0, 1, 0])  # Green for modified
    ])
    combined_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save combined point cloud
    combined_filename = os.path.join(output_dir, "comparison_pointcloud.ply")
    o3d.io.write_point_cloud(combined_filename, combined_pcd)
    print(f"Combined point cloud saved to {combined_filename}")
    
    print(f"Point cloud files saved to {output_dir}")
    print("Note: To visualize these .ply files, you can use Open3D's visualization tool or any 3D viewer")
    print("      You may need to run: o3d.visualization.draw_geometries([pcd]) with your point cloud")
    
    return [original_filename, modified_filename, combined_filename]


def process_kitti_bin(input_file, output_file, output_dir="figures"):
    """Process KITTI .bin file to remove half of the beams and visualize."""
    points = load_kitti_bin(input_file)
    filtered_points = remove_half_beams(points, num_bins=64)
    
    print(f"Original point cloud: {len(points)} points")
    print(f"Filtered point cloud: {len(filtered_points)} points")
    print(f"Reduction ratio: {len(filtered_points)/len(points):.2f}")

    save_kitti_bin(output_file, filtered_points)
    print(f"Processed file saved to: {output_file}")

    visualization_files = visualize_and_save_point_clouds(points, filtered_points, output_dir)
    print(f"Visualization files saved to: {', '.join(visualization_files)}")
    
    # Try to generate PNG images if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        print("Matplotlib imported successfully, generating visualizations...")
        
        # Original pointcloud visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sample points to avoid overplotting
        sample_size = min(5000, len(points))
        indices = np.random.choice(len(points), sample_size, replace=False)
        sampled_points = points[indices, :3]
        
        ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], 
                  c='r', s=1, alpha=0.5)
        ax.set_title('Original Point Cloud (Sampled)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=30, azim=45)
        plt.savefig(os.path.join(output_dir, "original_pointcloud.png"), dpi=300, bbox_inches='tight')
        
        # Modified pointcloud visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        sample_size = min(5000, len(filtered_points))
        indices = np.random.choice(len(filtered_points), sample_size, replace=False)
        sampled_points = filtered_points[indices, :3]
        
        ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], 
                  c='g', s=1, alpha=0.5)
        ax.set_title('Half-Beam Point Cloud (Sampled)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=30, azim=45)
        plt.savefig(os.path.join(output_dir, "modified_pointcloud.png"), dpi=300, bbox_inches='tight')
        
        # Side-by-side comparison (top view)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original (top view)
        sample_size = min(10000, len(points))
        indices = np.random.choice(len(points), sample_size, replace=False)
        sampled_points = points[indices, :3]
        
        ax1.scatter(sampled_points[:, 0], sampled_points[:, 1], c='r', s=0.5, alpha=0.5)
        ax1.set_title('Original Point Cloud - Top View')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_aspect('equal')
        ax1.grid(True)
        
        # Modified (top view)
        sample_size = min(10000, len(filtered_points))
        indices = np.random.choice(len(filtered_points), sample_size, replace=False)
        sampled_points = filtered_points[indices, :3]
        
        ax2.scatter(sampled_points[:, 0], sampled_points[:, 1], c='g', s=0.5, alpha=0.5)
        ax2.set_title('Half-Beam Point Cloud - Top View')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_aspect('equal')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison_top_view.png"), dpi=300, bbox_inches='tight')
        
        print(f"Additional matplotlib visualizations saved to {output_dir}")
        
    except ImportError:
        print("Matplotlib not available for additional visualizations. PLY files are still generated.")
    except Exception as e:
        print(f"Error generating matplotlib visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove half the beams from a KITTI .bin point cloud and visualize."
    )
    parser.add_argument("input", help="Path to the input KITTI .bin file")
    parser.add_argument("output", help="Path to the output KITTI .bin file")
    parser.add_argument("--output_dir", default="figures", help="Directory to save visualization images")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
    else:
        process_kitti_bin(args.input, args.output, args.output_dir)
