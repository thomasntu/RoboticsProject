#!/usr/bin/env python
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from datetime import datetime

sys.path.append('/home/robot/colcon_ws/install/tm_msgs/lib/python3.6/site-packages')
from sensor_msgs.msg import Image
from tm_msgs.msg import *
from tm_msgs.srv import *


class ImageSub(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.subscription = self.create_subscription(Image,
                                                     'techman_image', self.image_callback, 10)

    def image_callback(self, data):
        self.get_logger().info('Received image')

        # Write the image to the images folder
        img = np.array(data.data).reshape(data.height, data.width, 3)
        cv2.imwrite(f'images/stereo/IMG{datetime.strftime(datetime.now(), "%H%M%S")}.png', img)

def main(args=None):
    rclpy.init(args=args)
    node = ImageSub('image_sub')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
