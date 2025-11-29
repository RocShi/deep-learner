# deep-learner

> 深度学习练手项目，samples 持续更新。

## 1 克隆仓库

```bash
git clone --recurse-submodules https://github.com/RocShi/deep-learner.git && cd deep-learner
```

## 2 配置基础环境

```bash
chmod +x my-dev-env/host/setup_wsl2.sh && ./my-dev-env/host/setup_wsl2.sh
```

## 3 构建镜像并启动容器

```bash
# 注意，下面使用的是 deep-learner 目录下的 docker-compose.sh
chmod +x docker-compose.sh

# 构建镜像（添加 --no-cache 选项将强制重新构建而非使用缓存）
./docker-compose.sh build dl-dev-cuda11.8-cudnn8-pytorch1.13.1-ubuntu22.04

# 启动容器
./docker-compose.sh up -d dl-dev-cuda11.8-cudnn8-pytorch1.13.1-ubuntu22.04
```

## 4 进入容器并训练模型

```bash
# 进入容器：自动进入挂载的 /workspace/samples 目录
docker exec -it dl_dev_cuda11.8 /bin/bash

# 训练手写数字识别模型：数据及模型保存在 handwritten_digit_recognition 下
cd handwritten_digit_recognition && python train.py
```
