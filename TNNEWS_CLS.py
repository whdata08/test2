#git测试1
import json
from transformers import BertModel,BertConfig,BertTokenizer
import torch
from torch import nn,optim
from tqdm.auto import tqdm
import matplotlib
import matplotlib.pyplot as plt
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#torch.cuda.set_device(1)
device=torch.device('cpu')#('cuda' if torch.cuda.is_available() else 'cpu')
#读取数据
file_train = "/mnt/wuhan/data/tnews_public/train.json"
file_label="/mnt/wuhan/data/tnews_public/labels.json"
file_val="/mnt/wuhan/data/tnews_public/dev.json"
file_test="/mnt/wuhan/data/tnews_public/test.json"
#labels = [
#    '100', '101', '102', '103', '104', '106', '107', '108', '109', '110', '112',
#    '113', '114', '115', '116'
#]
def load_data(filename):#重要：按行读取数据？，最好不要用来解析json，只适用于json中每条数据占一行的情况
    fr=open(filename,encoding='utf-8')#打开整个文件，每行为原文件的行
    data = fr.readlines()
    data_set=[]
    for i in range(len(data)):
        line=json.loads(data[i].strip())
        #text,label=line['sentence'],line.get('label')
        #data_set.append(text,labels.index(label))这两行可以直接加label对应ids
        data_set.append(line)
    return data_set

TrainSet=load_data(file_train)
LabelSet=load_data(file_label)
ValSet=load_data(file_val)

# 用200条数据实验记得删除
#TrainSet=TrainSet[:500]
#ValSet=ValSet[0:300]

#数据预处理
#对应label与序号，用于计算损失
label_to_ids={}
for i in range(len(LabelSet)):
    label_to_ids[LabelSet[i]['label']]=i
    i=i+1
#在数据中把label对应到其序号
for i in TrainSet:
    i['label_ids']=label_to_ids[i['label']]
for i in ValSet:
    i['label_ids']=label_to_ids[i['label']]




#取出句子和对应类别序号
train_data=[(i['sentence'],i['label_ids']) for i in TrainSet]
test_data=[(i['sentence'],i['label_ids']) for i in ValSet]

#将数据分batch，每个batch为列表中一个元素
def batch_data(inputs,batch_size):
    batched_data=[]
    a=0
    b=batch_size
    while b-batch_size<len(inputs):
        batched_data.append(([i[0] for i in inputs[a:b]],[i[1] for i in inputs[a:b]]))
        a=a+batch_size
        b+=batch_size
    return batched_data
#模型
#configuration= BertConfig.from_pretrained('hfl/chinese-bert-wwm-ext')#这行是不是可以删掉，下面也把configuration删掉
#configuration=BertConfig()
model=BertModel.from_pretrained('hfl/chinese-bert-wwm-ext')
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')

model.to(device)

#分类器，处理模型CLS对应的输出
class Linear1(nn.Module):
    def __init__(self,input_dim,num_class):
        super(Linear1,self).__init__()
        self.linear=nn.Linear(input_dim,num_class)
        self.activation=nn.LogSoftmax(dim=1)
    def forward(self,inputs):
        hidden=self.linear(inputs)
        probs=self.activation(hidden)
        return(probs)
linear=Linear1(768,len(LabelSet))
#linear=torch.nn.DataParallel(linear)
#linear=linear.cuda()
linear.to(device)
#损失，优化
criterion=nn.NLLLoss()
optimizer=optim.Adam(model.parameters(),lr=0.00001)#换0.00001，step_size=4，gamma=0.1
from torch.optim import lr_scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.7)
#超参数
num_epoch=16
batch_size=60
#训练
model.train()#????测试的时候不用model.eval（）？？？
loss_cur = []
for epoch in range(num_epoch):
    total_loss=0
    for batch in tqdm(batch_data(train_data,batch_size),desc=f'Training Epoch{epoch}',leave=True):
        inputs,targets=[x for x in batch]
        targets=torch.tensor(targets)
        tokenized_text=tokenizer(inputs,padding=True,return_tensors='pt',truncation=True,max_length=128)
        outputs=model(**tokenized_text.to(device))
        log_probs=linear(outputs.last_hidden_state[:,0,:])
        loss=criterion(log_probs,targets.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        loss_cur.append(loss.item())
    print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
    print(f"Loss:{total_loss:.2f}")
    scheduler.step()
plt.plot(loss_cur)
print(loss_cur)
plt.show()
#测试过程
model.eval()
acc=0
for batch in tqdm(batch_data(test_data,1)):
    inputs,targets=[x for x in batch]
    with torch.no_grad():
        targets=torch.tensor(targets)
        tokenized_text=tokenizer(inputs,padding=True,return_tensors='pt',truncation=True,max_length=200)
        outputs=model(**tokenized_text.to(device))
        output=linear(outputs.last_hidden_state[:,0,:])
        acc+=(output.argmax(dim=1)==targets.to(device)).sum().item()
print(f"Acc:{acc/len(batch_data(test_data,1)):.2f}")
