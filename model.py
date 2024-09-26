import cv2
import torch
import urllib.request
import open3d as o3d
import numpy as np
from PIL import Image
from removebg import remove_background
from messh_gen import generate_mesh
import warnings

# Ignore all warnings
def reconstruct(file):

    warnings.filterwarnings("ignore")

    # import matplotlib.pyplot as plt

    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # img = cv2.imread(file)

    bg_removed_np = remove_background(file)
    img = cv2.cvtColor(bg_removed_np, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    depth_normalized = cv2.normalize(output, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite("depth.png", depth_normalized)

    # Resize the depth image to match the RGB image dimensions
    depth_resized = cv2.resize(depth_normalized, (img.shape[1], img.shape[0]))
    # Use the predicted depth to create a point cloud
    depth_image = o3d.geometry.Image((depth_resized * 1000).astype(np.uint16))  # Convert depth to 16-bit format
    # Create RGBD image
    rgb_image = o3d.geometry.Image(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image)
    # Define camera intrinsics
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )
    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsics)

    rgb_image = img
    depth_image = depth_normalized

    # Convert RGB image to RGBD image (OpenCV loads images as BGR by default)
    # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    # Assume some camera intrinsics (fx, fy, cx, cy)
    fx, fy = 525.0, 525.0  # Focal length
    cx, cy = rgb_image.shape[1] // 2, rgb_image.shape[0] // 2  # Principal point

    # Generate 3D points
    rows, cols = depth_image.shape
    points = []
    colors = []

    for v in range(rows):
        for u in range(cols):
            z = depth_image[v, u] / 1000.0  # Convert depth to meters
            if z == 0: continue  # Skip invalid depth
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])
            colors.append(rgb_image[v, u] / 255.0)  # Normalize to [0, 1]

    points = np.array(points)
    colors = np.array(colors)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    mesh = generate_mesh(pcd)
    return mesh