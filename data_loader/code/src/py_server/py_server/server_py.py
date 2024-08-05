import rclpy
import cv2
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from PIL import Image as PIL_Image
from nemo import test
import numpy as np
from picture_upload.srv import ImageCv


# ROS2
class ServicePY(Node):

    def __init__(self):
        super().__init__("service_py")
        self.nemo = test.NemoInfer()
        # 3-1.创建服务端；
        self.srv = self.create_service(
            ImageCv, "image_test", self.add_two_ints_callback
        )
        self.get_logger().info("服务端启动！")

    # 处理请求数据并响应结果
    def add_two_ints_callback(self, request, response):
        # 判断客户端图片是否为空
        if request.client_image.width <= 0 or request.client_image.height <= 0:
            print("the client_image is empty")

        bridge = CvBridge()
        try:
            cv_img = bridge.imgmsg_to_cv2(request.client_image, "bgr8")
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        except CvBridgeError as e:
            print(e)

        image = PIL_Image.fromarray(cv_img)  # 转为PIL的Image
        result = self.nemo.infer(orig_image=image)  # 推理

        img_ser = bridge.cv2_to_imgmsg(result, encoding="bgr8")

        response.server_image = img_ser
        return response


def main():
    # ROS2
    rclpy.init()  # 初始化 ROS2 客户端；
    service = ServicePY()

    rclpy.spin(service)

    # 5.释放资源。
    rclpy.shutdown()
