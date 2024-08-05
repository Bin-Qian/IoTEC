import math
import io
import os
import cv2
import sys
import argparse
from pathlib import Path
from typing import Iterable
from PIL import Image, ImageFont, ImageDraw
import numpy as np

import torch

# import util.misc as utils
from .util import misc as utils

from .models import build_model
from .datasets.face import make_face_transforms

import matplotlib.pyplot as plt
import time


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def get_images(in_path):
    img_files = []
    # 遍历,dirpath正在遍历的路径，dirnames子目录名称，filenames是dirpath下的非目录文件名称
    for dirpath, dirnames, filenames in os.walk(in_path):
        for file in filenames:
            # 从文件名中分离出文件的基本名称和拓展名
            filename, ext = os.path.splitext(file)
            # 扩展名转换为小写，以确保文件扩展名的比较不受大小写影响
            ext = str.lower(ext)
            if ext in [".jpg", ".jpeg", ".gif", ".png", ".pgm"]:
                # 将 dirpath 和 file 连接成完整的文件路径
                img_files.append(os.path.join(dirpath, file))
    return img_files


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--lr_drop", default=200, type=int)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )
    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--num_queries", default=32, type=int, help="Number of query slots"
    )
    parser.add_argument("--pre_norm", action="store_true")

    # * Segmentation
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Train segmentation head if the flag is provided",
    )

    # # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=1,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )
    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument(
        "--eos_coef",
        default=0.1,
        type=float,
        help="Relative classification weight of the no-object class",
    )

    # dataset parameters
    parser.add_argument("--dataset_file", default="face")
    parser.add_argument(
        "--data_path", type=str, default="/project/local_code/sample_data"
    )
    parser.add_argument("--data_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument(
        "--output_dir", help='path where to save the results, default is "output"'
    )
    parser.add_argument(
        "--device", default="cpu", help="device to use for training / testing"
    )
    parser.add_argument(
        "--resume",
        default="/project/src/nemo/nemo/Nemo_moudle.pth",
        help="resume from checkpoint",
    )

    parser.add_argument("--thresh", default=0.5, type=float)

    return parser


def plot_results(pil_img, prob, boxes, save_path):
    COLORS = [
        [0.000, 0.447, 0.741],
        [0.850, 0.325, 0.098],
        [0.929, 0.694, 0.125],
        [0.494, 0.184, 0.556],
        [0.466, 0.674, 0.188],
        [0.301, 0.745, 0.933],
    ]
    CLASSES = ["N/A", "smoke", "fire", "flame", "NightSmoke"]

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()  # 获取当前的轴对象，用于绘图
    # p: 当前目标的概率;(xmin, ymin, xmax, ymax): 当前边界框的坐标;c: 从 COLORS 中选取的颜色。zip(将 prob 列表、boxes 转换为普通的Python列表和 COLORS 列表（重复100次以匹配 boxes 的长度）中的元素打包成一个元组列表
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3
            )
        )  # 创建一个矩形
        cl = p.argmax()  # 找到概率最高的类别索引
        if cl >= len(CLASSES):
            continue  # skip if the class index is out of range
        text = f"{CLASSES[cl]}: {p[cl]:0.2f}"  # 创建一个格式化的字符串
        # 在边界框的左上角绘制文本,dict设置文本框背景为黄色,透明度为0.5
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")  # 关闭坐标轴的显示
    plt.savefig(save_path)
    plt.close()


@torch.no_grad()  # 指定一个代码块，在该代码块中不追踪梯度（即不计算关于张量（tensor）的操作的梯度）。这通常用在推理（inference）阶段或者在训练过程中执行不需要梯度更新的操作。
def infer(images_path, model, postprocessors, device, output_path, args):
    model.eval()
    duration = 0
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for img_sample in images_path:
        # 返回路径的最后一部分，即文件名和扩展名
        filename = os.path.basename(img_sample)
        # format(filename) 是一个字符串格式化操作，它将 filename 变量的值插入到大括号 {} 的位置
        print("processing...{}".format(filename))
        orig_image = Image.open(img_sample)
        w, h = orig_image.size
        # TODO: face? 创建一个变换对象或变换序列
        transform = make_face_transforms("val")
        # 字典，包含两个键 "size" 和 "orig_size",可能用于在模型训练或推理过程中模拟目标（例如边界框或掩码）
        dummy_target = {
            "size": torch.as_tensor([int(h), int(w)]),
            "orig_size": torch.as_tensor([int(h), int(w)]),
        }
        image, targets = transform(orig_image, dummy_target)
        # 在张量 image 的第一个维度上增加一个维度。这通常是为了匹配神经网络的输入要求
        image = image.unsqueeze(0)
        image = image.to(device)

        # 观察特征图（conv_features）、编码器的注意力权重（enc_attn_weights）和解码器的注意力权重（dec_attn_weights）
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        hooks = [
            model.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),  # 对模型的 backbone 组件中的倒数第二层注册一个前向传播钩子。这个钩子将在该层的前向传播完成后被调用，并捕获该层的输出特征图
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),  # 对模型的 transformer 组件中的编码器的最后一层的自注意力（self_attn）机制注册一个前向传播钩子。这个钩子将捕获编码器的注意力权重。
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),  # 对模型的 transformer 组件中的解码器的最后一层的多头注意力（multihead_attn）机制注册一个前向传播钩子。这个钩子将捕获解码器的注意力权重。
        ]

        start_t = time.perf_counter()
        outputs = model(image)
        end_t = time.perf_counter()

        # 将模型输出中的预测 logits 和预测框 (pred_boxes) 从 GPU 移动到 CPU 内存
        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()

        probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
        # 过滤概率，保留概率大于阈值的预测结果
        keep = probas.max(-1).values > args.thresh

        # 将模型输出的预测框重新缩放到原始图像的尺寸
        bboxes_scaled = rescale_bboxes(outputs["pred_boxes"][0, keep], orig_image.size)
        probas1 = probas[keep]

        for hook in hooks:
            hook.remove()

        conv_features = conv_features[0]
        enc_attn_weights = enc_attn_weights[0]
        dec_attn_weights = dec_attn_weights[0].cpu()

        # 获取卷积特征图的高度和宽度
        h, w = conv_features["0"].tensors.shape[-2:]

        # 检查是否有有效的预测框，如果没有，则跳过当前循环
        if len(bboxes_scaled) == 0:
            continue
        print("there is a box")

        img = np.array(orig_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for idx, box in enumerate(bboxes_scaled):
            bbox = box.cpu().data.numpy()
            bbox = bbox.astype(np.int32)
            bbox = np.array(
                [
                    [bbox[0], bbox[1]],
                    [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]],
                    [bbox[0], bbox[3]],
                ]
            )
            bbox = bbox.reshape((4, 2))  # 多余?
            cv2.polylines(img, [bbox], True, (0, 255, 0), 2)

        save_path = os.path.join(output_path, filename)
        plot_results(img, probas[keep], bboxes_scaled, save_path)

        infer_time = end_t - start_t
        duration += infer_time
        print("Processing...{} ({:.3f}s)".format(filename, infer_time))

    avg_duration = duration / len(images_path)
    print("Avg. Time: {:.3f}s".format(avg_duration))


# TODO:
def plot_results_return_CVimg(pil_img, prob, boxes):
    COLORS = [
        [0.000, 0.447, 0.741],
        [0.850, 0.325, 0.098],
        [0.929, 0.694, 0.125],
        [0.494, 0.184, 0.556],
        [0.466, 0.674, 0.188],
        [0.301, 0.745, 0.933],
    ]
    CLASSES = ["N/A", "smoke", "fire", "flame", "NightSmoke"]

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()  # 获取当前的轴对象，用于绘图
    # p: 当前目标的概率;(xmin, ymin, xmax, ymax): 当前边界框的坐标;c: 从 COLORS 中选取的颜色。zip(将 prob 列表、boxes 转换为普通的Python列表和 COLORS 列表（重复100次以匹配 boxes 的长度）中的元素打包成一个元组列表
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                color=c,
                linewidth=3,
            )
        )  # 创建一个矩形
        cl = p.argmax()  # 找到概率最高的类别索引
        if cl >= len(CLASSES):
            continue  # skip if the class index is out of range
        text = f"{CLASSES[cl]}: {p[cl]:0.2f}"  # 创建一个格式化的字符串
        # 在边界框的左上角绘制文本,dict设置文本框背景为黄色,透明度为0.5
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")  # 关闭坐标轴的显示
    fig = plt.gcf()  # 获取当前的 Matplotlib 图形对象
    plt.close()

    # 将PLT转为CV
    buffer_ = io.BytesIO()  # 申请缓存
    fig.savefig(buffer_, bbox_inches="tight", pad_inches=0)  # 后两个参数是为了去白边
    buffer_.seek(0)
    image = Image.open(buffer_)
    pil_image_np = np.asarray(image)
    buffer_.close()  # 释放缓存
    cv_image = cv2.cvtColor(pil_image_np, cv2.COLOR_RGB2BGR)

    # cv2.imwrite("/project/local_code/output/img.jpg", cv_image)

    return cv_image


@torch.no_grad()
def infer_one_image(orig_image, model, postprocessors, device, args):
    model.eval()
    w, h = orig_image.size
    transform = make_face_transforms("val")  # 创建一个变换对象或变换序列
    # 字典，包含两个键 "size" 和 "orig_size",可能用于在模型训练或推理过程中模拟目标（例如边界框或掩码）
    dummy_target = {
        "size": torch.as_tensor([int(h), int(w)]),
        "orig_size": torch.as_tensor([int(h), int(w)]),
    }
    image, targets = transform(orig_image, dummy_target)
    # 在张量 image 的第一个维度上增加一个维度。这通常是为了匹配神经网络的输入要求
    image = image.unsqueeze(0)
    image = image.to(device)

    start_t = time.perf_counter()
    outputs = model(image)
    end_t = time.perf_counter()

    # 将模型输出中的预测 logits 和预测框 (pred_boxes) 从 GPU 移动到 CPU 内存
    outputs["pred_logits"] = outputs["pred_logits"].cpu()
    outputs["pred_boxes"] = outputs["pred_boxes"].cpu()

    probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
    # 过滤概率，保留概率大于阈值的预测结果
    keep = probas.max(-1).values > args.thresh

    # 将模型输出的预测框重新缩放到原始图像的尺寸
    bboxes_scaled = rescale_bboxes(outputs["pred_boxes"][0, keep], orig_image.size)

    # 检查是否有有效的预测框
    if len(bboxes_scaled) == 0:
        print("result: no fire or smoke")
        return orig_image

    print("result: fire or smoke exist")

    img = np.array(orig_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # for idx, box in enumerate(bboxes_scaled):
    #     bbox = box.cpu().data.numpy()
    #     bbox = bbox.astype(np.int32)
    #     bbox = np.array(
    #         [
    #             [bbox[0], bbox[1]],
    #             [bbox[2], bbox[1]],
    #             [bbox[2], bbox[3]],
    #             [bbox[0], bbox[3]],
    #         ]
    #     )
    #     bbox = bbox.reshape((4, 2))
    #     cv2.polylines(img, [bbox], True, (0, 255, 0), 2)

    # save_path = os.path.join(output_path, filename)
    img_with_boxes = plot_results_return_CVimg(img, probas[keep], bboxes_scaled)

    infer_time = end_t - start_t
    print("Processing...image ({:.3f}s)".format(infer_time))
    return img_with_boxes


class NemoInfer:
    def __init__(self):
        # 设置模型所需参数
        parser = argparse.ArgumentParser(
            "DETR training and evaluation script", parents=[get_args_parser()]
        )
        self.args = parser.parse_args()  # 存储参数
        self.model, _, self.postprocessors = build_model(self.args)
        self.device = torch.device(self.args.device)  # 用于存储设备信息
        # 指定模型
        if self.args.resume:
            checkpoint = torch.load(self.args.resume, map_location="cpu")
            # Override the number of classes and queries
            num_classes = 7
            num_queries = 32
            self.model.class_embed = torch.nn.Linear(
                self.model.class_embed.in_features, num_classes
            )
            self.model.query_embed = torch.nn.Embedding(
                num_queries, self.model.query_embed.embedding_dim
            )
            self.model.load_state_dict(checkpoint["model"])

        self.model.to(self.device)

    # 进行推理,返回是cv2的图片
    def infer(self, orig_image):
        return infer_one_image(
            orig_image, self.model, self.postprocessors, self.device, self.args
        )


def main():
    nemo = NemoInfer()
    img = Image.open(
        "/project/local_code/sample_data/sample_val_frames/5th_hour_of_the_Cold_Springs_Fire_August_14th_2015_FR-825.jpg"
    )
    result = nemo.infer(orig_image=img)

    pil_image_np = np.asarray(result)
    cv_image = cv2.cvtColor(pil_image_np, cv2.COLOR_RGBA2BGR)

    cv2.imwrite("/project/local_code/output/img.jpg", cv_image)
    return


if __name__ == "__main__":
    print("!!!!!!!!!!!!!!!!!")
    main()
