import threading

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image #, PointCloud2, PointField
# import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

import pyrealsense2 as rs
import numpy as np
import cv2

class RealsenseRos2(Node):
    def __init__(self):
        super().__init__('realsense_ros2')
        self.color_topic_name = "/color_images"
        self.depth_topic_name = "/depth_images"
        self.pointcloud_topic_name = "/pointcloud"
        self.pointcloud_frame = "map"
        self.br = CvBridge()
        self.color_publisher = self.create_publisher(Image, self.color_topic_name , 10)
        self.depth_publisher = self.create_publisher(Image, self.depth_topic_name , 10)
        # self.pc_publisher = self.create_publisher(PointCloud2, self.pointcloud_topic_name , 10)
        self.partial_header = Header()
        self.partial_header.frame_id = self.pointcloud_frame

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))
        print(self.device_product_line)
    
        ctx = rs.context()
        serials = []
        devices = ctx.query_devices()
        for dev in devices:
            dev.hardware_reset()
        
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        self.color_intrinsic = None
        self.depth_scale = None
        self.color_intrinsic = None
        self.depth_intrinsic = None
    
    # get the camera intrinsics attribute
    def _get_intrinsics_from_stream(self, stream_profile):
        intrinsics = stream_profile.as_video_stream_profile().get_intrinsics()
        return np.array([[intrinsics.fx, 0, intrinsics.ppx],
                         [0, intrinsics.fy, intrinsics.ppy],
                         [0, 0, 1]])

    def start(self):
        self.pipeline.start(self.config)
        self.color_intrinsic = self.get_color_intrinsics()
        self.depth_scale = self.get_depth_scale()
        self.depth_intrinsic = self.get_depth_intrinsics()
        # Run once to get rid of initial bad sensor frames
        self.get_and_publish_image(publish=False)
        self.get_logger.info("Camera initialized.")
        self.get_logger.info("Depth scale: %f" % self.depth_scale)

    def get_depth_scale(self):
        """ Get the depth scale of the depth camera """
        profile = self.pipeline.get_active_profile()
        depth_sensor = profile.get_device().first_depth_sensor()
        return depth_sensor.get_depth_scale()
    
    def get_depth_intrinsics(self):
        """ Get the intrinsics of the aligned depth camera 
        :return: numpy array of shape (3, 3) with the intrinsic parameters
        """
        stream_profile = self.pipeline.get_active_profile().get_stream(rs.stream.depth)
        return self._get_intrinsics_from_stream(stream_profile)

    def get_color_intrinsics(self):
        """ Get the color intrinsics of the camera """
        stream_profile = self.pipeline.get_active_profile().get_stream(rs.stream.color)
        return self._get_intrinsics_from_stream(stream_profile)
    
    def stop(self):
        self.pipeline.stop()

    def get_and_publish_image(self, publish: bool =True):
        align_to = rs.stream.color
        align = rs.align(align_to)
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return
        
        # Processing blocks
        pc = rs.pointcloud()
        decimate = rs.decimation_filter()
        decimate.set_option(rs.option.filter_magnitude, 1)
        depth_frame = decimate.process(depth_frame)

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        # Pointcloud data to arrays
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)
        print(verts)
        # pointcloud_msg = PointCloud2()
        # pointcloud_msg.width = verts.shape[0]
        # pointcloud_msg.height = 1
        # pointcloud_msg.header = self.partial_header
        # pointcloud_msg.header.stamp = self.get_clock().now()
        # pointcloud_msg_points = [PointField(verts[i][0], verts[i][1], verts[i][2]) for i in range(verts.shape[0])]
        # pointcloud_msg.

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        resized_color_image = color_image

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        
        if publish:
            self.color_publisher.publish(self.br.cv2_to_imgmsg(resized_color_image, "bgr8"))
            self.depth_publisher.publish(self.br.cv2_to_imgmsg(depth_colormap, "16UC1"))
            # self.pc_publisher


def main(args=None):

    rclpy.init(args=args)

    realsense_ros2 = RealsenseRos2()

    thread = threading.Thread(
    target=rclpy.spin, args=(realsense_ros2, ), daemon=True)
    thread.start()

    realsense_ros2.start()

    while rclpy.ok():
        realsense_ros2.get_and_publish_image()
    
    realsense_ros2.stop()
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    realsense_ros2.destroy_node()
    rclpy.shutdown()   


if __name__ == '__main__':
    main()