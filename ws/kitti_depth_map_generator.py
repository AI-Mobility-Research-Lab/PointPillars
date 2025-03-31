import numpy as np
import cv2
import os
from tqdm import tqdm

################################
# 1) Parse KITTI Calibration
################################
def read_calib_file(calib_path):
    """
    Reads a KITTI calibration file and returns dict of numpy arrays:
      {
        'P2': (3x4),
        'R0_rect': (4x4),
        'Tr_velo_to_cam': (4x4)
      }
    """
    data = {}
    with open(calib_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) < 2:
                continue
            key, value = line.split(':', 1)
            value = value.strip()
            # e.g. P2: 7.215377e+02 0.000000e+00 ...
            data[key] = np.array([float(x) for x in value.split(' ')])

    # Reshape relevant parts
    # Projection matrix P2 (3x4)
    P2 = data['P2'].reshape(3, 4)

    # Rectification R0_rect (3x3), but often stored as 3x3 in text. We'll make it 4x4
    R0 = data['R0_rect'].reshape(3, 3)
    R0_rect_4x4 = np.eye(4)
    R0_rect_4x4[:3, :3] = R0

    # Tr_velo_to_cam (3x4), we'll embed into 4x4
    Tr_velo_to_cam = data['Tr_velo_to_cam'].reshape(3, 4)
    Tr_velo_to_cam_4x4 = np.eye(4)
    Tr_velo_to_cam_4x4[:3, :4] = Tr_velo_to_cam

    return {
        'P2': P2,
        'R0_rect': R0_rect_4x4,
        'Tr_velo_to_cam': Tr_velo_to_cam_4x4
    }

################################
# 2) Load and project a point cloud
################################
def project_velo_to_image(points_velo, calib):
    """
    Projects velodyne points (N x 4) into the camera-2 image plane.
    Returns (u, v, z) for each point (in pixel coords plus depth).
    points_velo: array of shape (N, 4) [x, y, z, reflectance]
    """
    P2 = calib['P2']             # shape (3, 4)
    R0_rect = calib['R0_rect']   # shape (4, 4)
    Tr_velo_to_cam = calib['Tr_velo_to_cam']  # shape (4, 4)

    # 1) Convert points from velodyne to homogeneous coords
    pts_velo_hom = np.concatenate([points_velo[:, :3], np.ones((points_velo.shape[0], 1))], axis=1)  # (N,4)

    # 2) Apply rigid transform: Velo -> Cam
    pts_cam = (Tr_velo_to_cam @ pts_velo_hom.T).T  # shape (N,4)
    
    # 3) Apply rectification: Cam -> Rectified
    pts_rect = (R0_rect @ pts_cam.T).T  # (N,4)

    # 4) Project to image plane: (3x4) * (4xN) = (3xN)
    pts_img = (P2 @ pts_rect.T).T  # (N,3)

    # Scale to get pixel coords
    pts_img[:, 0] /= pts_img[:, 2]  # x / z
    pts_img[:, 1] /= pts_img[:, 2]  # y / z

    # Keep (u, v, depth)
    u = pts_img[:, 0]
    v = pts_img[:, 1]
    z = pts_rect[:, 2]  # depth in camera coordinates

    return u, v, z

################################
# 3) Create a depth image
################################
def create_depth_image(u, v, z, image_shape):
    """
    Creates a float depth map (same size as the camera image).
    - (u,v) are pixel coords
    - z is depth in camera coords
    - image_shape: (height, width)
    Returns depth_img (height, width) with each pixel storing depth (or 0 if no point).
    """
    h, w = image_shape
    depth_img = np.zeros((h, w), dtype=np.float32)

    # Round u,v to nearest pixel
    u_rounded = np.round(u).astype(int)
    v_rounded = np.round(v).astype(int)

    # Only use points that fall within the image
    valid_mask = (
        (u_rounded >= 0) & (u_rounded < w) &
        (v_rounded >= 0) & (v_rounded < h) &
        (z > 0)
    )

    for i in range(len(z)):
        if valid_mask[i]:
            col = u_rounded[i]
            row = v_rounded[i]
            depth_img[row, col] = z[i]

    return depth_img

################################
# 4) Main
################################
def main():
    # Define base directory and ensure paths are absolute
    base_dir = os.path.abspath("dataset/kitti/training")
    velodyne_dir = os.path.join(base_dir, "velodyne")
    
    # Check if directory exists
    if not os.path.exists(velodyne_dir):
        raise FileNotFoundError(f"Velodyne directory not found at: {velodyne_dir}")
        
    frame_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith('.bin')])
    total_frames = len(frame_files)

    # Create output directory if it doesn't exist
    os.makedirs("ws/depth", exist_ok=True)

    # Use tqdm for progress bar
    for frame_file in tqdm(frame_files, desc="Processing frames"):
        frame_idx = frame_file.split('.')[0]  # Remove .bin extension
        
        calib_path = os.path.join(base_dir, "calib", f"{frame_idx}.txt")
        velo_path = os.path.join(base_dir, "velodyne", f"{frame_idx}.bin")
        image_path = os.path.join(base_dir, "image_2", f"{frame_idx}.png")

        # Skip if depth image already exists (allows for resuming)
        depth_output_path = f"ws/depth/depth_{frame_idx}.png"
        if os.path.exists(depth_output_path):
            continue

        try:
            # 1) Read calibration
            calib = read_calib_file(calib_path)

            # 2) Load Velodyne points
            points_velo = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)

            # 3) Load corresponding camera image
            rgb_image = cv2.imread(image_path)
            if rgb_image is None:
                print(f"\nError: Could not load image {image_path}")
                continue
            h, w, _ = rgb_image.shape

            # 4) Project point cloud to image plane
            u, v, z = project_velo_to_image(points_velo, calib)

            # 5) Create a depth image
            depth_img = create_depth_image(u, v, z, (h, w))

            # 6) Colorize and save the depth image
            depth_disp = depth_img.copy()
            max_depth = np.max(depth_disp)
            if max_depth > 0:
                depth_disp = (depth_disp / max_depth * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_disp, cv2.COLORMAP_JET)
            
            cv2.imwrite(depth_output_path, depth_colored)

        except Exception as e:
            print(f"\nError processing frame {frame_idx}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
