# E2E-MLT
代码fork自[E2E-MLT](https://github.com/MichalBusta/E2E-MLT)，完整说明文档见[Readme](README_ori.md)

## Requirements
  - python3.x with
  - opencv-python
  - pytorch
  - torchvision
## 编译

### NMS 模块

**setuptool**

Create file `setup.py` in `nms` directory and type the following script.

```
from setuptools import setup, Extension

functions_module = Extension(
    name='adaptor',
    sources=['adaptor.cpp', 'include/clipper/clipper.cpp'],
    #sources=['adaptor.cpp'],
    include_dirs=[r'.\include',
                  r'E:\install\Anaconda3\include'] # change to your path
)

setup(ext_modules=[functions_module])
```

**Compile**

First make sure c++ compiling toolchain is in PATH.

On windows its better to use VisualStudio's Prompt cmd env.

```
python setup.py build_ext --inplace
```

### Warp CTC

**Warp-ctc**
To make life easier, I just install the module with pip.
~~~
pip install torch-baidu-ctc
~~~

## 下载预训练模型

[e2e-mlt](http://ptak.felk.cvut.cz/public_datasets/SyntText/e2e-mlt.h5), [e2e-mlt-rctw](http://ptak.felk.cvut.cz/public_datasets/SyntText/e2e-mltrctw.h5)

```
wget http://ptak.felk.cvut.cz/public_datasets/SyntText/e2e-mlt.h5
```


## 测试
>下载的模型为 7500 字符集的，而代码中设置的为8400，需要改一下不然跑不起来。将 `models.py` 里面的 8400  替换为 7500， 我已经改好了，直接用。

### GPU 运行

```
python3 demo.py -model=e2e-mlt.h5
```

### CPU 运行
跑起来比较慢，笔记本电脑跑起来非常的卡。
~~~
python3 demo.py -cuda=0 -model=e2e-mlt.h5
~~~

