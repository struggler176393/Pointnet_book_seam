import open3d as o3d

class Open3dVisualizer():

	def __init__(self):

		self.point_cloud = o3d.geometry.PointCloud()
		self.o3d_started = False

		self.vis = o3d.visualization.VisualizerWithKeyCallback()
		self.vis.create_window()

	def __call__(self, points, colors):

		self.update(points, colors)

		return False

	def update(self, points, colors):
		coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.15, origin = [0,0,0])
		self.point_cloud.points = points
		self.point_cloud.colors = colors
		# self.point_cloud.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
		# self.vis.clear_geometries()
		# Add geometries if it is the first time
		if not self.o3d_started:
			self.vis.add_geometry(self.point_cloud)
			self.vis.add_geometry(coord_mesh)
			self.o3d_started = True

		else:
			self.vis.update_geometry(self.point_cloud)
			self.vis.update_geometry(coord_mesh)

		self.vis.poll_events()
		self.vis.update_renderer()