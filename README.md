# HappyAI

## 项目说明
身处信息技术变革时代的我们，有必要去亲身体验、感受人工智能技术，才能更好地适应这个时代。

目前大多数AI项目依赖复杂的环境部署、高性能的GPU计算硬件。本项目旨在集成各类易用的AI基础能力，并基于此开发各类简单有趣的AI应用，力求部署简单，方便体验，让更多的人感受到AI的乐趣。

本项目所有项目都可在CPU性能中上的电脑上实时运行，不需要高性能显卡(GPU)。

## 硬件条件
* 性能中上的CPU（无需GPU）
* 摄像头

## 系统环境
* Windows、MacOS、Linux均可
* Python3
* git

## 项目安装
* git clone https://github.com/YipengChen/HappyAI
* pip install -r requirements.txt

## 基础AI能力
* 人脸检测
* 人脸关键点检测
* 手部关键点检测
* 人体关键点检测
* 人体综合检测（脸部关键点+手部关键点+人体关键点）
* 20类、80类常用物体检测
* 卡通风格转换
* 人像分割

## 体验demo
|                          摄像头测试                          |                           人脸检测                           |                        人脸关键点检测                        |                        手部关键点检测                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![image](https://github.com/YipengChen/HappyAI/blob/main/docs/gifs/camera_read.gif) | ![image](https://github.com/YipengChen/HappyAI/blob/main/docs/gifs/face_detection.gif) | ![image](https://github.com/YipengChen/HappyAI/blob/main/docs/gifs/face_mesh.gif) | ![image](https://github.com/YipengChen/HappyAI/blob/main/docs/gifs/hand_detection.gif) |

|                        人体关键点检测                        |                         人体综合检测                         |                       20类常用物体检测                       |                       80类常用物体检测                       |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![image](https://github.com/YipengChen/HappyAI/blob/main/docs/gifs/pose_detection.gif) | ![image](https://github.com/YipengChen/HappyAI/blob/main/docs/gifs/holistic_detection.gif) | ![image](https://github.com/YipengChen/HappyAI/blob/main/docs/gifs/object_detection_20class.gif) | ![image](https://github.com/YipengChen/HappyAI/blob/main/docs/gifs/object_detection_80class.gif) |

|                           风格转换                           |                           嘴唇美妆                           |                           放大眼睛                           |                           闭眼检测                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![image](https://github.com/YipengChen/HappyAI/blob/main/docs/gifs/style_transfer.gif) | ![image](https://github.com/YipengChen/HappyAI/blob/main/docs/gifs/beautify_lip.gif) | ![image](https://github.com/YipengChen/HappyAI/blob/main/docs/gifs/eyes_enlarged.gif) | ![image](https://github.com/YipengChen/HappyAI/blob/main/docs/gifs/eyes_closed_detection.gif) |

|                           张嘴检测                           |                           虚拟口罩                           |                         自拍人像分割                         |                           虚拟面具                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![image](https://github.com/YipengChen/HappyAI/blob/main/docs/gifs/mouth_opened_detection.gif) | ![image](https://github.com/YipengChen/HappyAI/blob/main/docs/gifs/mask_face.gif) | ![image](https://github.com/YipengChen/HappyAI/blob/main/docs/gifs/selfie_segmentation.gif) | ![image](https://github.com/YipengChen/HappyAI/blob/main/docs/gifs/AR_face.gif) |

|                             换脸                             |               人物渲染-东京奥运会比赛项目图标                |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![image](https://github.com/YipengChen/HappyAI/blob/main/docs/gifs/change_face.gif) | ![image](https://github.com/YipengChen/HappyAI/blob/main/docs/gifs/Tokyo2020.gif) |

* 摄像头测试 （python examples/camera_read.py）
* 人脸检测（python examples/face_detection.py）
* 人脸关键点检测（python examples/face_mesh.py）
* 手部关键点检测（python examples/hand_detection.py）
* 人体关键点检测（python examples/pose_detection.py）
* 人体综合检测（python examples/holistic_detection.py）
* 20类常用物体检测（python examples/object_detection_20class.py）
* 80类常用物体检测（python examples/object_detection_80class.py）
* 风格转换（python examples/style_transfer.py）
* 嘴唇美妆（python examples/beautify_lip.py）
* 放大眼睛（python examples/eyes_enlarged.py）
* 闭眼检测 （python examples/eyes_closed_detection.py）
* 张嘴检测（python examples/mouth_opened_detection.py）
* 虚拟口罩（python examples/mask_face.py）
* 自拍人像分割（python examples/selfie_segmentation.py）
* 虚拟面具（python examples/AR_face.py）
* 换脸（python examples/change_face.py）
* 人物渲染-东京奥运会比赛项目图标（python examples/Tokyo2020.py）

## 下一步目标
* 增加手势识别能力及demo
* 增加OCR能力及demo

## 参考
* https://github.com/google/mediapipe
* https://github.com/spmallick/learnopencv
* https://github.com/hpc203
* https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.7/contrib/PP-HumanSeg
* https://github.com/Kazuhito00/Tokyo2020-Pictogram-using-MediaPipe

## 项目记录
- 2022/12/16: 项目说明增加GIFs
- 2022/12/6: 增加人物渲染demo（东京奥运会比赛项目图标）
- 2022/12/2：人像分割增加PP-HumanSeg-v2模型，比起mediapipe分割效果更好