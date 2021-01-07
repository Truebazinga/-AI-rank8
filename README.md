## [华为云“云上先锋”· AI挑战赛--街景图像分割](https://competition.huaweicloud.com/information/1000041336/introduction)
我们是赛道上**RANK8**的SELIMS队，这里分享一下我们的方案：
### 比赛描述
这个比赛主要是对给定的街道7类的语义分割，该7类分别是cityscapes数据集的7个大类。分别是对街景图像路面、人、车辆、建筑、交通标志、植物、天空等物体进行分割。
### 部署框架
这一部分我们使用的是limzero大佬之前在华为云上部署的mmseg框架，这里放上大佬的github链接：https://github.com/DLLXW/data-science-competition
### 数据集准备
因为华为给的数据集主要是json的标注，然而想要使用mmseg进行训练的话需要将json转换成标注的png。这一步使用mmseg自带的 mmsegmentation/tools/convert_datasets/cityscapes.py即可。
另外如果需要使用原版cityscapes的数据集，需要自己去下一个cityscapesScripts，将 cityscapesScripts/cityscapesscripts/helpers/labels.py 中labels的trainId改成和粗类(即7类)一置。这是因为如果不进行修改直接生成标注png就会变成19类的分类，而我们训练的是7类的粗类。再修改之后，export修改好的cityscapesScripts路径即可，mmseg的转换工具就会自动转换成粗类这里上传我们自己修改的labels.py
### 网络选取及trick
我们使用的是OCRNet，同样也是基于mmseg
- backbone：hrnet40 (预训练模型是在mmdet上找的)，我们发现如果使用hrnet48就会超时，这应该是我们没有调整测试阶段图像尺寸的原因。
- loss: Dice loss (抄的paddleseg,详情见dice_loss.py)，这个提升比较多
- 将图像做(0.5,2)的多尺度变换，水平翻转，还尝试了一下randomrotation,不过用处不大，后来去掉了。
- OHEMsampler
- 加入cityscapes的数据集，将训练集和验证集全部加入训练集中，只保留部分华为比赛的数据集作为验证机。官方提供的是11G的cityscapes数据集，本来还想试一下44G的cityscapes数据集，没时间了，不过也不一定有效
- 我们发现，在最后几次迭代中，提点是比较多的，因此我们会接上最后一个模型的学习率，让这个模型继续finetune 20000次迭代。能有大概0.001的提升，这个也比较靠运气。
## 总结
看了大佬的方案之后发现还是数据是最重要的，我们并没有清洗数据，吃了不少亏。如果能够发现这些问题，应该还能往上提一提。