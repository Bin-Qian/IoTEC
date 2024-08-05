#include "cv_bridge/cv_bridge.hpp"
#include "rclcpp/rclcpp.hpp"
#include "picture_upload/srv/image_cv.hpp"
#include "opencv2/opencv.hpp"
using picture_upload::srv::ImageCv;
using namespace std::chrono_literals;

// 定义节点类；
class MyClient : public rclcpp::Node
{
public:
    MyClient() : Node("my_client")
    {
        // 创建客户端；
        client = this->create_client<ImageCv>("image_test");
        RCLCPP_INFO(this->get_logger(), "客户端创建，等待连接服务端！");
    }
    // 等待服务连接；
    bool connect_server()
    {
        while (!client->wait_for_service(1s))
        {
            if (!rclcpp::ok())
            {
                RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "强制退出！");
                return false;
            }

            RCLCPP_INFO(this->get_logger(), "服务连接中，请稍候...");
        }
        return true;
    }
    // 组织请求数据并发送；
    rclcpp::Client<ImageCv>::FutureAndRequestId send_request(sensor_msgs::msg::Image image, bool isReq)
    {
        auto request = std::make_shared<ImageCv::Request>();
        request->client_image = image;
        request->is_req = isReq;
        return client->async_send_request(request);
    }

private:
    rclcpp::Client<ImageCv>::SharedPtr client;
};

int main(int argc, char **argv)
{
    // 2.初始化 ROS2 客户端；
    rclcpp::init(argc, argv);
    // 4.创建对象指针并调用其功能；
    auto client = std::make_shared<MyClient>();
    bool flag = client->connect_server();
    if (!flag)
    {
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "服务连接失败！");
        return 0;
    }

    cv::Mat img = cv::imread("/project/sample_data/sample_val_frames/5th_hour_of_the_Cold_Springs_Fire_August_14th_2015_FR-825.jpg", 1);
    if (img.empty()) {
        RCLCPP_INFO(client->get_logger(), "读取图片为空");
        return 1;
        // 处理图片为空的情况
    }

    sensor_msgs::msg::Image::SharedPtr img1 = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img).toImageMsg();
    auto response = client->send_request(*img1, true);

    // 处理响应
    if (rclcpp::spin_until_future_complete(client, response) == rclcpp::FutureReturnCode::SUCCESS)
    {
        RCLCPP_INFO(client->get_logger(), "请求正常处理");

        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(response.get()->server_image, sensor_msgs::image_encodings::BGR8);
        cv::Mat s_img = cv_ptr->image;
        if (!s_img.empty())
        {
            std::cout << "成功接收图片" << std::endl;
            cv::imwrite("/project/output/result.jpg", s_img);
            std::cout << "成功保存图片" << std::endl;
        }
        else
        {
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "服务连接失败！");
        }
    }
    else
    {
        RCLCPP_INFO(client->get_logger(), "请求异常");
    }

    // 5.释放资源。
    rclcpp::shutdown();
    return 0;
}
