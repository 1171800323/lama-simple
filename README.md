# 简介

LaMa项目推理部分的简化版本

原始代码：https://github.com/saic-mdal/lama


# 推理

需要先配置环境：

```bashrc
$ pip install -r requirements.txt
$ conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
```

然后执行推理：
- device: 推理时所用GPU
- indir: 测试数据集地址
- outdir: 输出地址
- checkpointdir: 模型地址

```bashrc
$ python predict.py --device 0 \
--indir '/home/StyleTextDataset/LaMa/new_tests' \
--outdir '/home/StyleTextDataset/LaMa/new_outputs' \
--checkpointdir '/home/StyleTextDataset/LaMa/pretrained'
```