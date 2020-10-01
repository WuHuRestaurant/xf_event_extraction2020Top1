# XF-Event-Extraction
2020
科大讯飞事件抽取挑战赛

比赛链接：http://challenge.xfyun.cn/topic/info?type=hotspot


结果:
| Name     | Score |  Rank|Team member| 
| :--------|:------|:----|:----------|
|我是蛋糕王 | 0.73859| 1   |https://github.com/WuHuRestaurant<br>https://github.com/aker218|


事件抽取系统，包含触发词（trigger），事件论元（role），事件属性（attribution）的抽取。基于 pytorch 的 pipeline 解决方案。

## 主要思路

将任务分割为**触发词抽取**，**论元抽取**，**属性抽取**。具体而言是论元和属性的抽取结果依赖于**触发词**，因此只有一步误差传播。**因 time loc 并非每个句子中都存在，并且分布较为稀疏，因此将 time & loc 与 sub & obj 的抽取分开（role1 提取 sub & obj；role2 提取 time & loc）**

模型先进行**触发词提取**，由于复赛数据集的特殊性，模型限制抽取的事件仅有一个，**如果抽取出多个触发词，选择 logits 最大的 trigger 作为该句子的触发词**，如果没有抽取触发词，筛选整个句子的 logits，取 argmax 来获取触发词；

然后根据触发词抽取模型抽取的触发词，分别输入到 role1 & role2 & attribution 模型中，进行后序的论元提取和属性分类；四种模型都是基于 Roberta-wwm 进行实验，加入了不同的特征。

最后将识别的结果进行整合，得到提交文件。

## 项目运行主要环境

运行系统：

```python
Ubuntu 18.04.4
```

---

python:

```python
python3.7
```

----

python 运行环境，可以通过以下代码完成依赖包安装：

```python
pip install -r requirements.txt
```

```python
transformers==2.10.0
pytorch_crf==0.7.2
numpy==1.16.4
torch==1.5.1+cu101
tqdm==4.46.1
scikit_learn==0.23.2
torchcrf==1.1.0
```

CUDA:

```python
CUDA Version: 10.2  Driver 440.100 GPU：Tesla V100 (32G) * 2
```

## 项目目录说明

```shell
xf_ee
├── data                                    # 数据文件夹
│   ├── final                               # 复赛数据(处理过的)
│   │   ├── mid_data                        # 中间数据 （词典等）
│   │   ├── preliminary_clean               # 清洗后的初赛数据
│   │   └── raw_data                        # 复赛经过初步清洗后的 raw_data
│   └── preliminary                         # 初赛数据（略）
│
├── out                                     # 存放训练的模型
│   ├── final                               # 复赛各个单模型（trigger/role/attribution）
│   └── stack                               # 十折交叉验证的 attribution 模型
│
├── script/final                            # 放训练 / 评估 / 测试 的脚本
│   ├── train.sh                            
│   ├── dev.sh                     
│   └── test.sh                
│
├── src_final
│   ├── features_analysis                   # 数据分析
│   │   └── images                          # 分析时画得一些图 
│   ├── preprocess                       
│   │   ├── convert_raw_data.py             # 处理转换原始数据
│   │   ├── convert_raw_data_preliminary.py     # 转换初赛数据为复赛格式并处理
│   │   └── processor.py                    # 转换数据为 Bert 模型的输入
│   ├── utils                      
│   │   ├── attack_train_utils.py           # 对抗训练 FGM / PGD
│   │   ├── dataset_utils.py                # torch Dataset
│   │   ├── evaluator.py                    # 模型评估
│   │   ├── functions_utils.py              # 跨文件调用的一些 functions
│   │   ├── model_utils.py                  # 四个任务的 models
│   │   ├── options.py                      # 命令行参数
│   |   └── trainer.py                      # 训练器
|
├── dev.py                                  # 用于模型评估
├── ensemble_predict.py                     # 用百度 ERNIE 模型对 attribution 十折交叉验证
├── predict_preliminary.py                   # 对初赛数据进行清洗
├── readme.md                               # ...
├── test.py                                 # pipeline 预测复赛数据 （包含 ensemble）
└── train.py                                # 模型训练
```

## 使用说明

### 数据转换

数据转换部分只提供代码和已经转换好的数据，具体操作在 **src_final/preprocess **中的 convert_raw_data中，包含对初赛/复赛数据的清洗和转换。

### 训练阶段

```shell
bash ./script/final/train.sh
```

注：**脚本中指定的 BERT_DIR 指BERT所在文件夹，BERT采用的是哈工大的全词覆盖wwm模型，下载地址https://github.com/ymcui/Chinese-BERT-wwm，自行下载并制定对应文件夹，并将 vocab.txt 中的两个 unused 改成 [INV] 和 [BLANK]（详见 processor 代码中的 fine_grade_tokenize）**

**如果设备显存不够，自行调整 train_batch_size，脚本中的 batch_size（32）在上述环境中占用显存为16G**

可更改的公共参数有

```
lr: bert 模块的学习率
other_lr: 除了bert模块外的其他学习率（差分学习率）
weight_decay：...
attack_train： 'pgd' / 'fgm' / '' 对抗训练 fgm 训练速度慢一倍, pgd 慢两倍，但是效果都有提升
swa_start: 滑动权重平均开始的epoch
```

##### trigger提取模型训练 （TASK_TYPE=“trigger”）

可更改的参数有

```python
use_distant_trigger: 是否使用复赛数据构造的远程监督库中的 trigger 信息
```

##### role 提取模型训练 （TASK_TYPE=“role1/role2”）

可更改的参数有

```python
use_trigger_distance: 是否使用句子中的其他词到 trigger 的距离这一个特征
```

**attribution 分类模型训练 （TASK_TYPE=“attribution”）**

未使用其他特征



**MODE=“stack”** 时候对 attribution 任务进行十折交叉验证，换用百度 ERNIE1.0 模型作为预训练模型

### 验证阶段

```shell
bash ./script/final/dev.sh
```

主要的参数有三个：

* TASK_TYPE：需要验证任务的 type
* start/end threshold ：trigger / role1 model 需要进行调整的阈值
* dev_dir: 需要验证的模型的文件夹

### 测试阶段

```shell
bash ./script/final/test.sh
```

利用训练最优的四个单模型进行 pipeline 式的预测 sentences.json 文件，获取最终的 submit 文件，

其中 **submit_{version},json** 为四个单模型的结果， **submit_{version}_ensemble_,json** 为单模型 + attribution 交叉验证后的结果。

四个任务 model 的上级文件夹必须指定，同时文件夹名称应包含模型的参数特征。

* **trigger_ckpt_dir**：              trigger 所在的文件夹
* **role1_ckpt_dir**：                 role1 所在的文件夹
* **role2_ckpt_dir：**                 role2 所在的文件夹
* **attribution_ckpt_dir： **    attribution所在的文件夹

## 测试效果 

|     classification      |    score    |
| :---------------------: | :---------: |
|     submit_v1.json      |   0.73684   |
| submit_v1_ensemble.json | **0.73859** |

---
### 数据增强


在我们的训练过程中，实际使用了组委会提供的初赛(经过清洗和转换)+复赛数据进行训练，在项目内部提供了清洗完毕的初赛数据；具体清洗流程如下所示：

* 只使用复赛数据train得到trigger抽取模型和role1抽取模型(需指定model的上级文件夹)

    **trigger_simple_ckpt_dir**：             单独复赛数据train trigger 所在的文件夹
    **role1_simple_ckpt_dir**：               单独复赛数据train role1 所在的文件夹
* 使用predict_prelimiary.py调用train好的trigger model 和role1 model 预测初赛数据的trigger和sub/ob

```python
python predict_preliminary.py --dev_dir_trigger trigger_simple_ckpt_dir  --dev_dir_role role1_simple_ckpt_dir
```
* 运行src_final/preprocess下的convert_raw_data_preliminary.py
```python
python convert_raw_data_preliminary.py
```
* 运行src_final/preprocess下的convert_raw_data.py 即完成了初赛数据的清洗
```python
python convert_raw_data.py
```



