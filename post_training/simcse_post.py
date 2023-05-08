import argparse
import datetime
import os
from pickle import NONE
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model.bert_classify import *

from dataloader.dataset import *
from preprocess.read_data import *
from utils.common import *
from utils.metric import *
from torchsummary import summary
from criterions.cl_loss import loss_structrue
from criterions.cl_loss import RkdDistance
os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

## basic configuration
parser.add_argument('--save_model',action='store_true', default=True)
parser.add_argument('--save_model_dir', default='./save_model/')
parser.add_argument('--train_path', default='./data/train.csv')
parser.add_argument('--test_path', default='./data/test.csv')
parser.add_argument('--noise_ratio',type=float,default=0.0)
parser.add_argument('--noise_type',type=str,default="sym")
parser.add_argument('--fix_data',type=str,default='1')
parser.add_argument('--show_bar',action='store_true', default=False)
parser.add_argument('--seed',type=int,default=128)

## args for train
parser.add_argument('--epoch',type=int,default=6)
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--sentence_len',type=int,default=256)
parser.add_argument('--num_class',type=int,default=10)
parser.add_argument('--learning_rate',type=float,default=1e-5)
parser.add_argument('--dropout_rate',type=float,default=0.1)
parser.add_argument('--log_path',type=str,default="./")
parser.add_argument('--data',type=str,default="")

## args for model
parser.add_argument('--train_aug', type=bool, default=False,
    help='whether to use augement data')
parser.add_argument('--bert_type',type=str,default='bert-base-uncased')
parser.add_argument("--mix_option",type=int,default=0,
    help='mix option for bert , 0: base bert model from huggingface; 1: mix bert')


# args for contrasive 
parser.add_argument('--cl',type=str,default="simcse")
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--mlm_w', type=float, default=1)
args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print_args(args)
setup_seed(args.seed)

EPOCH = args.epoch
BATCH_SIZE = args.batch_size

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"use gpu device =",os.environ["CUDA_VISIBLE_DEVICES"])

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"load data from",args.train_path)

train_data, valid_data  = process_csv(args, args.train_path)
test_data = process_test_csv(args, args.test_path)

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"train data %d , valid data %d , test data %d " \
    %(len(train_data),len(valid_data),len(test_data)))

train_loader = MyDataloader(args,train_data).run("all")
#valid_loader = MyDataloader(args,valid_data).run("all")
test_loader = MyDataloader(args,test_data).run("all")



def train(args, mymodel, optimizer, dataset, valid_data=None, test_data=None):
    
    test_last_mlm = []
    test_last_cl = []
    
    test_best = 0.0

    # evaluate(valid_loader, 0, mymodel, 'valid before train')

    for epoch in range(1, EPOCH + 1):
        train_loss = 0.0
        train_acc = 0.0
        train_recall = 0.0
        ce_train = 0.0
        cl_train = 0.0
        st_train = 0.0
        mlm_train = 0.0
        mymodel.train()
        bar = None

        if args.show_bar:
            bar = get_progressbar(epoch, EPOCH, len(train_loader), 'train')

        for i, data in enumerate(train_loader):

            input_ids, attention_mask, labels, _ = [Variable(elem.cuda()) for elem in data]
            
            mlm_input1, mlm_labels1 = train_loader.dataset.mask_tokens(input_ids)
            mlm_input2, mlm_labels2 = train_loader.dataset.mask_tokens(input_ids)
            optimizer.zero_grad()

            _, feature1, mlm_loss1 = mymodel(mlm_input1, attention_mask,mlm_labels1)
            _, feature2, mlm_loss2 = mymodel(mlm_input2, attention_mask,mlm_labels2)

            loss_mlm = (mlm_loss1 + mlm_loss2) / 2

            feature1 = feature1.unsqueeze(dim=1)
            feature2 = feature2.unsqueeze(dim=1)

            features = torch.cat([feature1, feature2],dim=1) # -> shape [batch_size,2,sentence_len,hidden_size]
            
            loss_cl = args.alpha * CL(features)

            loss = loss_mlm + loss_cl

            mlm_train += loss_mlm.item()
            cl_train += loss_cl.item()
            loss.backward()
            optimizer.step()

        
            if args.show_bar:
                bar.dynamic_messages.mlm = mlm_train / (i+1)
                bar.dynamic_messages.cl = cl_train / (i+1)
                bar.update(i + 1)
            torch.save(mymodel.module.state_dict(),"ckpt/"+args.data+str(epoch))
        test_last_mlm.append(mlm_train / len(train_data))
        test_last_cl.append(cl_train / len(train_data))
        if bar:
            bar.finish()
        
    return test_last_mlm, test_last_cl


mymodel=PostBert(args)

# for name,params in mymodel.named_parameters():
#     print(name,params.shape)

log_file = open(args.log_path, 'w')

mymodel=torch.nn.DataParallel(mymodel).cuda()
optimizer=optim.Adam(mymodel.parameters(),lr=args.learning_rate)
CL = get_cl_criterion(args)

test_last_mlm, test_last_cl = train(args, mymodel, optimizer, train_data, test_data=test_loader)



print('mlm_loss',test_last_mlm,file=log_file)
print('cl_loss', test_last_cl,file=log_file)
log_file.close()