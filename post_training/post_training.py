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
    
    test_best_l = []
    test_last_l = []
    
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
            
            mlm_input, mlm_labels = train_loader.dataset.mask_tokens(input_ids)
          
            optimizer.zero_grad()

            logits_ori, feature_ori, mlm_loss = mymodel(mlm_input, attention_mask,mlm_labels)

            loss = mlm_loss


            if mlm_train != None:
                mlm_train += loss.item()

            loss.backward()
            optimizer.step()

        
            if args.show_bar:
                bar.dynamic_messages.mlm = mlm_train / (i+1)
                bar.update(i + 1)

        test_best_l.append(mlm_train / len(train_data))

        if bar:
            bar.finish()
        
        test_best_l.append
        
    return test_best_l


mymodel=PostBert(args)

# for name,params in mymodel.named_parameters():
#     print(name,params.shape)

log_file = open(args.log_path, 'w')

mymodel=torch.nn.DataParallel(mymodel).cuda()
optimizer=optim.Adam(mymodel.parameters(),lr=args.learning_rate)
CL = get_cl_criterion(args)

test_best_l = train(args, mymodel, optimizer, train_data, test_data=test_loader)

torch.save(mymodel.module.state_dict(),"ckpt/"+args.data)


print('mlm_loss',test_best_l,file=log_file)
log_file.close()