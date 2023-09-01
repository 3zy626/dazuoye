# 加载相关库
import time
import os
import numpy as np
import json
import paddle
import paddle.nn.functional as F
from paddlenlp.datasets import load_dataset
import paddlenlp

# 为了后续方便使用，我们使用python偏函数（partial）给 convert_example 赋予一些默认参数
from functools import partial

# 读取数据集
data_train = []  # 用于存储训练集
data_dev = []   # 用于存储验证集
data_test =[]   # 用于存储测试集
with open('train.txt', 'r', encoding='utf-8') as f:
    data_train0 = f.readlines()
  
    for line in data_train0:
        #print(line)
        #print(type(line))
        line = json.loads(line)
        data_train.append(line)       
    f.close()
with open('dev.txt', 'r', encoding='utf-8') as f:
    data_dev0 = f.readlines()  
    for line in data_dev0:
        line = json.loads(line)
        data_dev.append(line)       
    f.close()

with open('test.txt', 'r', encoding='utf-8') as f:
    data_test0 = f.readlines()   
    for line in data_test0:
        line = json.loads(line)
        data_test.append(line)
    f.close()


data_train = paddlenlp.datasets.MapDataset(data_train)
data_dev = paddlenlp.datasets.MapDataset(data_dev)
data_test = paddlenlp.datasets.MapDataset(data_test)
# train_ds, dev_ds, test_ds是ernie 3.0用
train_ds, dev_ds, test_ds = data_train,data_dev,data_test
print("训练集数据量：",len(data_train))
print("验证集数据量：",len(data_dev))
print("测试集数据量：",len(data_test))
# 输出训练集的前 3 条样本
for idx, example in enumerate(data_train):
    if idx <= 2:
        print(example)
print(type(data_train))  
for idx, example in enumerate(data_test):
    if idx <= 2:
        print(example)

import random
# 因为是基于预训练模型 ERNIE-Gram 来进行，所以需要首先加载 ERNIE-Gram 的 tokenizer，
# 后续样本转换函数基于 tokenizer 对文本进行切分
tokenizer = paddlenlp.transformers.ErnieGramTokenizer.from_pretrained('ernie-gram-zh')

from paddlenlp.datasets import DatasetBuilder
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab
# 将 1 条明文数据的 sentence1、sentence2 拼接起来，根据预训练模型的 tokenizer 将明文转换为 ID 数据
# 返回 input_ids 和 token_type_ids
def convert_example(example, tokenizer, max_seq_length=512, is_test=False, is_flip=False):

    sentence1, sentence2 = example["sentence1"], example["sentence2"]
    if is_flip:
        random_v = random.uniform(0., 1.)
        if random_v>0.5:
            pass
        else:
            sentence1, sentence2 = sentence2, sentence1

    encoded_inputs = tokenizer(
        text=sentence1, text_pair=sentence2, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    # 在预测或者评估阶段，不返回 label 字段
    else:
        return input_ids, token_type_ids
    
### 对训练集的第 1 条数据进行转换
input_ids, token_type_ids, label = convert_example(data_train[0], tokenizer)
print(input_ids)
print(token_type_ids)
print(label)
# 为了后续方便使用，我们使用python偏函数（partial）给 convert_example 赋予一些默认参数
from functools import partial

# 训练集和验证集的样本转换函数
trans_func = partial(
                convert_example,
                tokenizer=tokenizer,
                max_seq_length=512)
# 测试集样本转换函数
trans_func_test = partial(
                convert_example,
                tokenizer=tokenizer,
                max_seq_length=512,
                is_test=True)


def create_dataloader(dataset,
                      trans_fn=None,
                      mode='train',
                      batch_size=1,
                      tokenizer=None):
    """
    Creats dataloader.
    Args:
        dataset(obj:`paddle.io.Dataset`): Dataset instance.
        mode(obj:`str`, optional, defaults to obj:`train`): If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1): The sample number of a mini-batch.
        pad_token_id(obj:`int`, optional, defaults to 0): The pad token index.
    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """
    if trans_fn:
        dataset = dataset.map(trans_fn)

    assert mode in ['train', 'dev', 'test'], "Error:mode is not in ['train', 'dev', 'test']"
    if mode == 'train':
        # 定义分布式 Sampler: 自动对训练数据进行切分，支持多卡并行训练
        batch_sampler = paddle.io.DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=True)
    else:
        # 针对验证集数据加载，我们使用单卡进行评估，所以采用 paddle.io.BatchSampler 即可
        batch_sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=False)
    
    if mode=='test':
        # 预测数据的组 batch 操作
        # predict 数据只返回 input_ids 和 token_type_ids，因此只需要 2 个 Pad 对象作为 batchify_fn
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment_ids
        ): [data for data in fn(samples)]
    
    else:
        # 我们的训练数据会返回 input_ids, token_type_ids, labels 3 个字段
        # 因此针对这 3 个字段需要分别定义 3 个组 batch 操作
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
            Stack(dtype="int64")  # label
        ): [data for data in fn(samples)]

    dataloader = paddle.io.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        return_list=True,
        collate_fn=batchify_fn)
    return dataloader

batch_size = 128
# 生成训练数据 data_loader
train_data_loader = create_dataloader(dataset=data_train,
                        trans_fn=trans_func,
                        mode='train',
                        batch_size=batch_size,
                        tokenizer=tokenizer)
# 生成验证数据 data_loader
dev_data_loader = create_dataloader(dataset=data_dev,
                        trans_fn=trans_func,
                        mode='dev',
                        batch_size=batch_size,
                        tokenizer=tokenizer)
# 生成预测数据 data_loader
predict_data_loader = create_dataloader(dataset=data_test,
                        trans_fn=trans_func_test,
                        mode='test',
                        batch_size=batch_size,
                        tokenizer=tokenizer)

print("train dataloader length:", len(train_data_loader))
print("dev dataloader length:", len(dev_data_loader))
print("predict dataloader length:", len(predict_data_loader))

import paddle
from paddle import nn
from paddle.nn import functional as F

class PointwiseMatching(nn.Layer):
   
    # 此处的 pretained_model 在本例中会被 ERNIE-Gram 预训练模型初始化
    def __init__(self, pretrained_model, dropout=None, has_pooler=True):
        super().__init__()
        self.ptm = pretrained_model
        self.has_pooler = has_pooler
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # 语义匹配任务: 相似、不相似 2 分类任务
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):

        # 此处的 Input_ids 由两条文本的 token ids 拼接而成
        # token_type_ids 表示两段文本的类型编码
        # 返回的 cls_embedding 就表示这两段文本经过模型的计算之后而得到的语义表示向量
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)
        
        if not self.has_pooler:
            cls_embedding = _[:, 0]

        cls_embedding = self.dropout(cls_embedding)

        # 基于文本对的语义表示向量进行 2 分类任务
        logits = self.classifier(cls_embedding)
        #probs = F.softmax(logits)

        #return probs

        return logits
    
# 我们基于 ERNIE-Gram 模型结构搭建 Point-wise 语义匹配网络
# 所以此处先定义 ERNIE-Gram 的 pretrained_model
pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained('ernie-gram-zh')

# 定义 Point-wise 语义匹配网络
model = PointwiseMatching(pretrained_model)

acc_all_1 = []  # 存放训练时acc，可以自己设置函数查看曲线
accu_all_1 = []   # 存放验证时acc
# 因为训练过程中同时要在验证集进行模型评估，因此我们先定义评估函数
@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader, writer, phase="dev", add_softmax=False):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
        if add_softmax:
            probs = F.softmax(probs)
        loss = criterion(probs, labels)
        losses.append(loss.numpy())
        correct = metric.compute(probs, labels)
        metric.update(correct)
        accu = metric.accumulate()
    
    print("eval {} loss: {:.5}, accu: {:.5}".format(phase,
                                                    np.mean(losses), accu))
    model.train()
    metric.reset()
    
    return losses, accu

from paddlenlp.transformers import LinearDecayWithWarmup

epochs = 10
num_training_steps = len(train_data_loader) * epochs

# 定义 learning_rate_scheduler，负责在训练过程中对 lr 进行调度
#warmup (int or float):
#            If int, it means the number of steps for warmup. If float, it means
#            the proportion of warmup in total training steps.
lr_scheduler = LinearDecayWithWarmup(learning_rate=5E-5, total_steps=num_training_steps, warmup=0.15)

# Generate parameter names needed to perform weight decay.
# All bias and LayerNorm parameters are excluded.
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]

# 定义 Optimizer
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=5e-4,
    apply_decay_param_fun=lambda x: x in decay_params)

# 采用交叉熵 损失函数
criterion = paddle.nn.loss.CrossEntropyLoss()

# 评估的时候采用准确率指标
metric = paddle.metric.Accuracy()

'''def train_model(model, optimizer, epochs, criterion, metric, save_dir, tokenizer, loader_list, patience=20, lr_scheduler=None, writer=None, add_softmax=False):
    global_step = 0
    tic_train = time.time()
    best_acc = 0
    pcount = patience

    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(loader_list[0], start=1):

            input_ids, token_type_ids, labels = batch
            probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
            if add_softmax:
                probs = F.softmax(probs)
            loss = criterion(probs, labels)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

            global_step += 1
        
            # 每间隔 1 step 输出训练指标
            if global_step % 1 == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, lr: %.7f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, acc, optimizer.get_lr(),
                        10 / (time.time() - tic_train)))
                
                tic_train = time.time()
                # 加入train日志显示
                writer.add_scalar(tag="train/loss", step=global_step, value=loss)
                writer.add_scalar(tag="train/acc", step=global_step, value=acc)
                writer.add_scalar(tag="train/lr", step=global_step, value=optimizer.get_lr())

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            # 每间隔 100 step 在验证集和测试集上进行评估
            if global_step % 100 == 0:
                pcount -= 1
                losses, accu = evaluate(model, criterion, metric, loader_list[1], writer, "dev", add_softmax)
                acc_all_1.append(acc)
                accu_all_1.append(accu)
                #加入eval日志显示
                writer.add_scalar(tag="eval/loss", step=global_step, value=np.mean(losses))
                writer.add_scalar(tag="eval/acc", step=global_step, value=accu)
    
                if accu>best_acc:
                    best_acc = accu
                    # 加入保存
                    save_param_path = os.path.join(save_dir, 'best_model_state.pdparams')
                    paddle.save(model.state_dict(), save_param_path)
                    tokenizer.save_pretrained(save_dir)
                    pcount = patience

                if pcount==0:
                    print("Early Stopping!")
                    break
    
        if pcount==0:
            break

    new_save_dir = os.path.join(save_dir, "model_%d" % global_step)
    os.makedirs(new_save_dir)
    save_param_path = os.path.join(new_save_dir, 'model_state.pdparams')
    paddle.save(model.state_dict(), save_param_path)
    tokenizer.save_pretrained(save_dir)

# 模型训练
patience = 3
# 加入日志显示
from visualdl import LogWriter
writer = LogWriter("./log_afqmc")
save_dir = "checkpoint_afqmc"

train_model(model, optimizer, epochs, criterion, metric, save_dir, tokenizer, loader_list=[train_data_loader, dev_data_loader], patience=patience, lr_scheduler=lr_scheduler, writer=writer)
'''

# 定义预测函数
def predict(model, data_loader):
    batch_probs = []
    # 预测阶段打开 eval 模式，模型中的 dropout 等操作会关掉
    model.eval()
    with paddle.no_grad():
        for batch_data in data_loader:
            input_ids, token_type_ids = batch_data
            input_ids = paddle.to_tensor(input_ids)
            token_type_ids = paddle.to_tensor(token_type_ids)           
            # 获取每个样本的预测概率: [batch_size, 2] 的矩阵
            batch_prob = model(
                input_ids=input_ids, token_type_ids=token_type_ids).numpy()
            batch_probs.append(batch_prob)
        batch_probs = np.concatenate(batch_probs, axis=0)
        return batch_probs
    
# 定义预测模型
pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained('ernie-gram-zh')

model = PointwiseMatching(pretrained_model)

# 加载已训练好的模型参数
# 刚才下载的模型解压之后存储路径为 ./ernie_gram_zh_pointwise_matching_model/model_state.pdparams
state_dict = paddle.load("./model_20000/model_state.pdparams")

model.set_dict(state_dict)


# 开始预测
print("predict dataloader length:", len(predict_data_loader))

for idx, batch in enumerate(predict_data_loader):
    if idx < 1:
        print(batch)
# 执行预测函数
y_probs = predict(model, predict_data_loader)

# 根据预测概率获取预测 label
y_preds = np.argmax(y_probs, axis=1)
print(y_preds)
# 将预测结果写入到文件
output_file = "predictions.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for pred in y_preds:
        f.write(str(pred) + "\n")

print("Predictions exported to:", output_file)
print("Accuracy=",metric)
