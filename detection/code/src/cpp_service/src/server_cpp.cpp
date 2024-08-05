#include "rclcpp/rclcpp.hpp"
#include "picture_upload/srv/image_cv.hpp"
#include "opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include "std_msgs/msg/string.hpp"
#include <cstdlib>
using picture_upload::srv::ImageCv;
using std::placeholders::_1;
using std::placeholders::_2;
using namespace std::chrono_literals;

// 定义节点类；
class MyService : public rclcpp::Node
{
public:
  MyService(const char *path) : Node("my_service")
  {
    // 创建服务端；
    server = this->create_service<ImageCv>("image_test", std::bind(&MyService::appect, this, _1, _2));
    picture_path = path;
    RCLCPP_INFO(this->get_logger(), "image_test 服务端启动完毕，等待请求提交...");
  }

private:
  rclcpp::Service<ImageCv>::SharedPtr server;
  const char *picture_path;

  // 处理请求数据并响应结果。
  bool appect(const ImageCv::Request::SharedPtr req, const ImageCv::Response::SharedPtr res)
  {
    // cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(req->client_image, sensor_msgs::image_encodings::BGR8); // 获取客户端的图片
    // cv::Mat img1 = cv_ptr->image;
    if (!req->is_req)
      return false;

    cv::Mat img = cv::imread(picture_path, 1); // 读取传给客户端的照片
    if (img.empty())
    {
      RCLCPP_INFO(this->get_logger(), "picture_server: read picture failed, please check the path and restart the server");
      return false;
    }
    sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img).toImageMsg();
    // cv::resize(img1, img1, cv::Size(img.size().width, img.size().height)); // 将客户端图片大小变为本地图片大小
    // cv::imshow("server", img1);
    // cv::waitKey(25000);
    res->server_image = *msg; // 发送服务器的图片
    // cv::destroyAllWindows();3
    return true;
  }
};

int main(int argc, char const *argv[])
{
  // 初始化 ROS2 客户端；
  rclcpp::init(argc, argv);

  // 获取全局变量中设定的路径
  const char *path = std::getenv("MY_PICTURE_PATH");
  if (path == nullptr)
  {
    path = "/default"; // 如果没有设置环境变量，则使用默认路径
    std::cout << "picture_server: path is empty, please check the path and restart the server" << std::endl;
  }

  // 调用spin函数，并传入节点对象指针；
  auto server = std::make_shared<MyService>(path);
  rclcpp::spin(server);

  // 释放资源。
  rclcpp::shutdown();
  return 0;
}
