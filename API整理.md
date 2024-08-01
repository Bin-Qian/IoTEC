模型检测类：
aiDetect.cpp
模型选择、模型推理、模型路径选择、模型更新

UI界面整合类：
baseblock.cpp、

相机控制类别：
cam.cpp
void CAM_USB::Cam_init(void) 相机参数初始化、

证书授权类：
Licence_board.cpp
1.string Licence_board::Get_CPU_ID()
描述：获取 CPU 的序列号并保存到配置文件中。
参数：无
返回：CPU 序列号（string）

WiFi连接类：
Board_net.cpp
1. 连接 WiFi
void Board_net::Wifi_connect()
描述：连接到指定的 WiFi 网络。
参数：无
返回：无
2. 获取 IP 地址
bool Board_net::GetIPAddress(bool ip_switch)
描述：获取指定网络接口的 IP 地址。
参数：- ip_switch：选择网络接口（true 为 WiFi，false 为 LAN）
返回：获取结果（bool）
3. 检查网络连接
bool Board_net::Is_connect()
描述：检查是否成功连接到网络。
参数：无
返回：连接状态（bool）
4. 保存参数
void Board_net::Para_Save()
描述：保存 WiFi 名称和密码到配置文件。
参数：无
返回：无

管道通信类：
class MY_FIFO
1. 构造函数
MY_FIFO(QObject *parent)
描述: MY_FIFO 类的构造函数。
参数:
  - QObject *parent: 父对象。
2. 公共方法
int AI_open_fifo(void)
描述: 打开 FIFO 进行通信。
返回值:
  - 成功时返回 1。
  - 如果打开用于读取的 stj_fifo_2 失败，返回 -1。
  - 如果打开用于写入的 stj_fifo_1 失败，返回 -2。
3. int AI_read_fifo(u_int8_t *buffer, u_int16_t len)
描述: 从 FIFO 读取数据。
参数:
  - u_int8_t *buffer: 存储读取数据的缓冲区。
  - u_int16_t len: 要读取的数据长度。
返回值:
  - 读取的字节数。
  - 如果读取失败，返回 -1。
  - 如果读取被中断，返回 -2。
4. int AI_write_fifo(u_int8_t *buffer, u_int16_t len)
描述: 向 FIFO 写入数据。
参数:
  - u_int8_t *buffer: 包含要写入数据的缓冲区。
  - u_int16_t len: 要写入的数据长度。
返回值:
  - 写入的字节数。
  - 如果写入失败，返回 -1。
5. u_int8_t data_buffer_sum(u_int8_t* dataBuf, u_int16_t len)
描述: 计算数据缓冲区的校验和。
参数:
  - u_int8_t* dataBuf: 数据缓冲区。
  - u_int16_t len: 数据缓冲区的长度。
返回值: 数据缓冲区的校验和。
6. void Fifo_Receive_Handler(void)
描述: 处理从 FIFO 接收的数据。
参数：无
返回值：无
7. void flag_solve(void)
描述: 根据标志处理接收到的数据。
参数：无
返回值：无
8. void Fifo_Send_Handler(void)
描述: 处理数据发送。
参数：无
返回值：无

图像显示类：
ImageShow 
1. 构造函数
ImageShow::ImageShow(QObject *parent = nullptr)
- 描述: 初始化 ImageShow 对象。
- 参数:
  - QObject *parent: 可选参数，指向父对象的指针，默认为 nullptr。
- 返回: 无

2. 公共方法
QImage ImageShow::matToQImage(const cv::Mat &src)
- 描述: 将 OpenCV 的 Mat 格式图像转换为 Qt 的 QImage 格式图像。
- 参数:
  - const cv::Mat &src: 输入的 OpenCV Mat 图像。
- 返回: 转换后的 QImage 图像。

3. void ImageShow::working()
- 描述: 持续获取图像并将其转换为 QImage，然后通过信号 imageReady 发送到 UI。
- 参数: 无
- 返回: 无

4. void ImageShow::imageReady(const QImage &display_img)
- 描述: 图像准备就绪信号。该信号在新的图像准备好并转换为 QImage 后被发射。
- 参数:
  - const QImage &display_img: 转换后的 QImage 图像。
- 返回: 无

私有成员变量
- bool work_flag: 工作标志，用于控制 working 方法中的循环。
- USBCam usb_cam: USB 摄像头对象，用于获取图像帧。

依赖库
- OpenCV: 用于图像处理和获取图像帧。
- Qt: 用于 GUI 和图像显示。






























