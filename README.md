# deep-learner

> 深度学习练手项目，samples 持续更新。

## 1 克隆仓库

```bash
git clone --depth=1 --recurse-submodules --shallow-submodules https://github.com/RocShi/deep-learner.git && cd deep-learner
```

## 2 配置基础环境

```bash
chmod +x my-dev-env/host/setup_wsl2.sh && ./my-dev-env/host/setup_wsl2.sh
```

## 3 模型开发与模型训练

### 3.1 构建镜像并启动容器

```bash
# 注意，下面使用的是 deep-learner 目录下的 docker-compose.sh
chmod +x docker-compose.sh

# 构建镜像（添加 --no-cache 选项将强制重新构建而非使用缓存）
sudo ./docker-compose.sh build dl-dev-cuda11.8-cudnn8-pytorch1.13.1-ubuntu22.04

# 启动容器
sudo ./docker-compose.sh up -d dl-dev-cuda11.8-cudnn8-pytorch1.13.1-ubuntu22.04
```

### 3.2 进入容器进行模型开发与模型训练

```bash
# 进入容器：自动进入挂载的 ${HOME}/workspace/samples/model 目录
sudo docker exec -it dl_dev_cuda11.8 /bin/bash

# 训练手写数字识别模型：数据及模型保存在 handwritten_digit_recognition 下
cd handwritten_digit_recognition && python train.py
```

## 4 模型部署

### 4.1 构建镜像并启动容器

```bash
# 注意，下面使用的是 deep-learner 目录下的 docker-compose.sh
chmod +x docker-compose.sh

# 构建镜像（添加 --no-cache 选项将强制重新构建而非使用缓存）
sudo ./docker-compose.sh build dl-dev-cuda12.1-cudnn8-tensorrt8.6.1.2-ubuntu22.04

# 启动容器
sudo ./docker-compose.sh up -d dl-dev-cuda12.1-cudnn8-tensorrt8.6.1.2-ubuntu22.04
```

### 4.2 进入容器进行模型部署开发

```bash
# 进入容器：自动进入挂载的 ${HOME}/workspace/samples/deployment 目录
sudo docker exec -it dl_dev_trt8.6.3 /bin/bash

# 进入对应的模型部署子项目，进项开发、编译、调试
```
