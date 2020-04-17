# PSENet 文字检测

原作基于python2.7/PyTorch v0.4，这里将它升级到python3 / PyTorch 1.0+

**注意，原作提供的模型都是PyTorch v0.4的，在高版本上不能加载，需要重新训练。**

# 配置

## 1.代码转 python3

全部转换到源文件

```
2to3 -n -W .
```

## 2.依赖库

```
# Polygon
pip install Polygon3
pip install pyclipper
```

# 理解

## 输出尺度

文中提到了 PSENet-1s/4s 两种输出分辨率的模型，但实际代码中，体现分辨率的仅仅在模型最后输出的upscale处， 模型本身的计算没有差别。

不同输出分辨率仅对后面的PSE计算有差别。

```
# x input image
# out -- Si
# upscale out to 1/scale of x
out = self._upsample(out, x, scale=self.scale)
```

# Hacking

## Training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ic15.py
```

## Testing

### ICDAR 2015

```
CUDA_VISIBLE_DEVICES=0 python test_ic15.py --scale 1 --resume [path of model]
```

### CTW1500

~~~~
CUDA_VISIBLE_DEVICES=0 python test_ctw1500.py --scale 1 --kernel_num 7 --ctw_root [path of CTW1500 data] --resume [path of model]
~~~~



## Eval script for ICDAR 2015 and SCUT-CTW1500

```
cd eval
sh eval_ic15.sh
sh eval_ctw1500.sh
```

