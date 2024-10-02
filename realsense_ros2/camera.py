import threading

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

import pyrealsense2 as rs
import numpy as np
import cv2

class RealsenseRos2(Node):
    def __init__(self):
        super().__init__('realsense_ros2')
        self.color_topic_name = "/color_images"
        self.depth_topic_name = "/depth_images"
        self.br = CvBridge()
        self.color_publisher = self.create_publisher(Image, self.color_topic_name , 10)
        self.depth_publisher = self.create_publisher(Image, self.depth_topic_name , 10)

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        self.pipeline_profile = config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))
    
        found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            err_str = "The demo requires Depth camera with Color sensor"
            self.get_logger().error(err_str)
            raise RuntimeError(err_str)
        
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.intrinsic = None
        self.depth_scale = None
    
    def start(self):
        self.pipeline.start(self.config)
        self.intrinsic = self.get_color_intrinsics()
        self.depth_scale = self.get_depth_scale()
        # Run once to get rid of initial bad sensor frames
        self.get_and_publish_image(publish=False)
        self.get_logger.info("Camera initialized.")
        self.get_logger.info("Depth scale: %f" % self.depth_scale)
    
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
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        resized_color_image = color_image

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        
        if publish:
            self.color_publisher.publish(self.br.cv2_to_imgmsg(color_colormap_dim))
            self.color_publisher.publish(self.br.cv2_to_imgmsg(depth_colormap_dim))


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