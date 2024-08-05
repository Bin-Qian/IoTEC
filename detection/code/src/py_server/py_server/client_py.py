import sys
import rclpy
import cv2
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
from picture_upload.srv import ImageCv
# 3.定义节点类；
class MinimalClient(Node):
  
    def __init__(self):
        super().__init__('minimal_client_py')
       
        # 3-1.创建客户端；
        self.cli = self.create_client(ImageCv, 'image_test')       
        self.req = ImageCv.Request()
        # 3-2.等待服务连接；
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('服务连接中，请稍候...')
         
 
    # 3-3.组织请求数据并发送；
    def send_request(self):
        self.req.is_req=True
        self.future = self.cli.call_async(self.req)
 
 
 
def main():
    # 2.初始化 ROS2 客户端；
    rclpy.init()
    # 4.创建对象并调用其功能；
    minimal_client = MinimalClient()
    minimal_client.send_request()
    # 处理响应
    rclpy.spin_until_future_complete(minimal_client,minimal_client.future)
    try:
        response = minimal_client.future.result()
 
    except Exception as e:
        minimal_client.get_logger().info(
            '服务请求失败：%r' % (e,))
       
    else:
        # bridge = CvBridge()
        # cv_img = bridge.imgmsg_to_cv2(response.server_image, "bgr8")
        # cv2.imshow("YOLOv8 Tracking", cv_img)
        minimal_client.get_logger().info('响应成功')
        # cv2.waitKey(5000)
        # cv2.destroyAllWindows()
    # 5.释放资源。
    rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()