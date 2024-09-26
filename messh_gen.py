import open3d as o3d

def generate_mesh(pcd):
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    print("Poisson reconstruction completed")

    # Crop the mesh to remove noise (optional)
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    # Simplify the mesh
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=100000)
    print("Mesh simplification completed")

    # Apply texture to the mesh
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    vertex_colors = []
    for vertex in mesh.vertices:
        [_, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 1)
        vertex_colors.append(pcd.colors[idx[0]])
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    # o3d.io.write_triangle_mesh("mesh_with_texture.ply", mesh)
    # print("Mesh with texture saved")
    return mesh
    # Save the mesh
   