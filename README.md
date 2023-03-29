

# Gobang-AI-program-based-on-AlphaZero 

### 基于AlphaZero的五子棋AI程序

此项目为熟悉pytorch后一个练手的项目，也是实验室项目的一小部分，参考模仿别人的工作做的一个五子棋AI程序，训练3k轮后...嗯...像一个可爱的小学生水平，会懂得堵我，但稍加计谋就被骗过去了 :(

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]




## 目录

- [上手指南](#上手指南)
  - [开发前的配置要求](#开发前的配置要求)
  - [安装步骤](#安装步骤)
- [部署](#部署)
- [文件目录说明](#文件目录说明)
- [版本控制](#版本控制)
- [作者](#作者)
- [鸣谢](#鸣谢)



### 上手指南

###### 开发前的配置要求

1. Pytorch
2. Numpy
2. Anaconda

###### **安装步骤**

```sh
git clone https://github.com/Son4ta/Gobang-AI-program-based-on-AlphaZero.git
```



### 部署

训练请使用

```sh
python train.py
```

对弈请使用

```
python play.py
```

请在代码中更改使用的模型



### 文件目录说明

根目录有已训练好的模型、网络定义、MCTS树程序、DataLoder、全局参数定义以及训练、对弈代码

```
Filetree 
├─alphaZeroNet(网络定义)
├─alphaZeroNet2(网络定义)
├─log(训练数据)
└─__pycache__

```



### 版本控制

该项目使用Git进行版本管理。您可以在repository参看当前可用版本。



### 作者

Son4ta@qq.com



### 版权说明

该项目签署了MIT 授权许可，详情请参阅 [LICENSE.txt](https://github.com/Son4ta/Gobang-AI-program-based-on-AlphaZero/blob/master/LICENSE.txt)



### 鸣谢


- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [Img Shields](https://shields.io)
- [Choose an Open Source License](https://choosealicense.com)
- [GitHub Pages](https://pages.github.com)
- [Animate.css](https://daneden.github.io/animate.css)
- [junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)

<!-- links -->

[your-project-path]:Son4ta/Gobang-AI-program-based-on-AlphaZero
[contributors-shield]: https://img.shields.io/github/contributors/Son4ta/Gobang-AI-program-based-on-AlphaZero.svg?style=flat-square
[contributors-url]: https://github.com/Son4ta/Gobang-AI-program-based-on-AlphaZero/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Son4ta/Gobang-AI-program-based-on-AlphaZero.svg?style=flat-square
[forks-url]: https://github.com/Son4ta/Gobang-AI-program-based-on-AlphaZero/network/members
[stars-shield]: https://img.shields.io/github/stars/Son4ta/Gobang-AI-program-based-on-AlphaZero.svg?style=flat-square
[stars-url]: https://github.com/Son4ta/Gobang-AI-program-based-on-AlphaZero/stargazers
[issues-shield]: https://img.shields.io/github/issues/Son4ta/Gobang-AI-program-based-on-AlphaZero.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/Son4ta/Gobang-AI-program-based-on-AlphaZero.svg
[license-shield]: https://img.shields.io/github/license/Son4ta/Gobang-AI-program-based-on-AlphaZero.svg?style=flat-square
[license-url]: https://github.com/Son4ta/Gobang-AI-program-based-on-AlphaZero/blob/master/LICENSE.txt



