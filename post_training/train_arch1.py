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
parser.add_argument('--pre_trained_data',type=str,default="sst2")
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
        train_data = MyDataloader(args,dataset).run("all")
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
            bar = get_progressbar(epoch, EPOCH, len(train_data), 'train')

        for i, data in enumerate(train_data):

            input_ids, attention_mask, labels, _ = [Variable(elem.cuda()) for elem in data]

          
            optimizer.zero_grad()

            logits_ori, feature_ori, _ = mymodel(input_ids, attention_mask)
            ce_loss = F.cross_entropy(logits_ori, labels)
            st_loss = None
            cl_loss = None
            mlm_loss = None
            # import pdb
            # pdb.set_trace()
            if args.mlm_w >= 0.1: 
                
                mlm_input1, mlm_labels1 = train_data.dataset.mask_tokens(input_ids)
                mlm_input2, mlm_labels2 = train_data.dataset.mask_tokens(input_ids)

                logits_aug1, feature_aug1, mlm_loss1 = mymodel(mlm_input1, attention_mask, mlm_labels1)
                logits_aug2, feature_aug2, mlm_loss2 = mymodel(mlm_input2, attention_mask, mlm_labels2)
            
                feature_aug1 = feature_aug1.unsqueeze(dim=1)
                feature_aug2 = feature_aug2.unsqueeze(dim=1)

                features = torch.cat([feature_aug1, feature_aug2],dim=1) # -> shape [batch_size,2,sentence_len,hidden_size]

                ST = RkdDistance()

                st_loss = args.beta * ST(logits_ori, feature_ori) 
                mlm_loss = args.mlm_w * (mlm_loss1 + mlm_loss2) / 2

                if args.cl == "supcon":
                    cl_loss = args.alpha * CL(features,labels)
                else:
                    cl_loss = args.alpha * CL(features)

                loss = ce_loss + cl_loss + st_loss + mlm_loss
            else:
                loss = ce_loss


            out = F.log_softmax(logits_ori)

            # import pdb
            # pdb.set_trace()

            ce_train += ce_loss.item()
            if cl_loss != None:
                cl_train += cl_loss.item()
            if st_loss != None:
                st_train += st_loss.item()
            if mlm_loss != None:
                mlm_train += mlm_loss.item()

            loss.backward()
            optimizer.step()

            out = out.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            train_metric = metric(out,labels)
            train_acc += train_metric[0] * labels.size
            train_recall += train_metric[2] * labels.size

            if args.show_bar:
                bar.dynamic_messages.ce = ce_train / (i+1)
                bar.dynamic_messages.cl = cl_train / (i+1)
                bar.dynamic_messages.st = st_train / (i+1)
                bar.dynamic_messages.mlm = mlm_train / (i+1)
                bar.dynamic_messages.acc = train_acc / (i*args.batch_size + labels.size)
                bar.dynamic_messages.recall = train_recall / (i*args.batch_size + labels.size)
                bar.update(i + 1)
        
        if bar:
            bar.finish()

        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"train %d/%d epochs Loss:%f, Acc:%f, Recall:%f" \
        %(epoch, EPOCH, train_loss / (i + 1), train_acc / len(train_data.dataset), train_recall / len(train_data.dataset)))
        
        if valid_data:
            _, val_acc = evaluate(valid_data, epoch, mymodel, 'valid')

        if test_data:
            _, test_last = evaluate(test_data, epoch, mymodel, 'test')
            test_best = max(test_best,test_last)

            test_best_l.append(test_best)
            test_last_l.append(test_last)
        
    return test_best_l, test_last_l



def evaluate(data, epoch, mymodel, mode):
    loss = 0.0
    acc = 0.0
    recall = 0.0
    mymodel.eval()

    for j, batch in enumerate(data):   
        input_ids, attention_mask, labels, _ = [Variable(elem.cuda()) for elem in batch]
     
        with torch.no_grad():
            logits, tmp1,tmp2 = mymodel(input_ids, attention_mask)
            loss += F.cross_entropy(logits, labels).mean()
            pred = F.log_softmax(logits)
            pred = pred.cpu().detach().cpu().numpy()
            labels = labels.cpu().detach().cpu().numpy()

            metric_ = metric(pred, labels)
            acc += metric_[0] * labels.size
            recall += metric_[2] * labels.size
  
    loss /= len(data)
    acc /= len(data.dataset)
    recall /= len(data.dataset)

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), mode, " %d/%d epochs Loss:%f, Acc:%f, Recall:%f" \
    %(epoch, EPOCH, loss , acc , recall))

    return loss, acc

mymodel=PostBert(args)
params = torch.load('ckpt/'+args.pre_trained_data)
params = {k:params[k] for k in params if "classifier" not in k and "projector" not in k}
mymodel.load_state_dict(params,strict=False)
# for name,params in mymodel.named_parameters():
#     print(name,params.shape)

log_file = open(args.log_path, 'w')

mymodel=torch.nn.DataParallel(mymodel).cuda()

optimizer=optim.Adam(mymodel.parameters(),lr=args.learning_rate)
CL = get_cl_criterion(args)

test_best_l, test_last_l = train(args, mymodel, optimizer, train_data, test_data=test_loader)

print('test_best',test_best_l)
print('test_last',test_last_l)
print("Test best %f , last %f"%(test_best_l[-1], test_last_l[-1]))

print('test_best',test_best_l,file=log_file)
print('test_last',test_last_l,file=log_file)
print("Test best %f , last %f"%(test_best_l[-1], test_last_l[-1]),file=log_file)
log_file.close()