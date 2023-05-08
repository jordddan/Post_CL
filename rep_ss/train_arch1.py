import argparse
import datetime
import os
from pickle import NONE
import warnings
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model.bert_classify import *

from dataloader.dataset import *
from dataloader.read_data import *
from utils.common import *
from utils.metric import *
from torchsummary import summary
from criterions.cl_loss import loss_structrue
from criterions.cl_loss import RkdDistance
from utils.graph_utils import knn_search,check_select
os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

## basic configuration
parser.add_argument('--save_model',action='store_true', default=True)
parser.add_argument('--save_model_dir', default='./save_model/')
parser.add_argument('--train_path', default='./data/train.csv')
parser.add_argument('--train_aug1', type=str,default="")
parser.add_argument('--train_aug2', type=str,default="")
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
parser.add_argument('--pre_trained_data',type=str,default="none")

## args for model
parser.add_argument('--bert_type',type=str,default='bert-base-uncased')
parser.add_argument("--mix_option",type=int,default=0,
    help='mix option for bert , 0: base bert model from huggingface; 1: mix bert')


# args for contrasive and graph search
parser.add_argument('--cl',type=str,default="simcse")
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--mlm_w', type=float, default=1)
parser.add_argument('--num_k',type=int,default=500)
parser.add_argument('--thre',type=int,default=4)
parser.add_argument('--warmup_epoch',type=int,default=10)
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

train_data = get_train_dataset(args)
train_loader = MyDataloader(args,train_data).run()
test_data = get_test_dataset(args)
test_loader = MyDataloader(args,test_data).run()
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"train data %d , test data %d " \
    %(len(train_data),len(test_data)))

def eval_train(args,mymodel,train_loader):
    '''
        update_sample feature
    '''
    mymodel.eval()
    with torch.no_grad():
        dataset = train_loader.dataset
        n = dataset.n
        num_class = dataset.num_class
        class_cnt = dataset.class_cnt
        class_emb = torch.zeros(dataset.class_emb.shape).cuda()
        emb = torch.zeros(dataset.emb.shape).cuda()
        logits = torch.zeros(dataset.logits.shape).cuda()
        for i, data in enumerate(tqdm(train_loader)):
            input_ids = Variable(data["input_ids"].cuda())
            att_mask = Variable(data["att_mask"].cuda())
            labels = Variable(data['labels'].cuda())

            indexes = data["indexes"]

            out, feature, _, cls_emb = mymodel(input_ids,att_mask)

            for j in range(len(indexes)): 
                index = indexes[j]
                emb[index] = feature[j]
                class_emb[labels[j]] += feature[j]
                logits[index] = out[j]

        for i in range(num_class):
            class_emb[i] /= class_cnt[i]



        # sim_matrix = torch.zeros((n,n)).cuda()
        divi = torch.norm(emb,dim=1)
        # div = torch.matmul(divi.unsqueeze(1),divi.unsqueeze(0))
        for i in range(emb.shape[0]):
            emb[i] /= divi[i].item()
        dataset.logits = logits.cpu()
        # import pdb
        # pdb.set_trace()
        knn_logits = knn_search(args, emb, dataset)
        pred_labels = torch.argmax(knn_logits,dim=-1).cpu()
        # sim_matrix = torch.matmul(emb,emb.transpose(1,0))
        # sim_matrix = sim_matrix / div
        labels = torch.tensor(dataset.labels,dtype=int)
        clean_labels = torch.tensor(dataset.clean_labels,dtype=int)
        is_clean = labels == clean_labels        

        pred_clean = pred_labels == labels
        

        check_select(is_clean,pred_clean)

        # import pdb
        # pdb.set_trace()
        
        dataset.class_emb = class_emb.to("cpu")
        dataset.emb = emb.to("cpu")
    
        knn_logits = torch.softmax(knn_logits, dim=-1)
    
    return pred_clean, knn_logits

def train(args, mymodel, optimizer, train_loader, valid_data=None, test_loader=None):
    
    test_best_l = []
    test_last_l = []
    
    test_best = 0.0

    # evaluate(valid_loader, 0, mymodel, 'valid before train')

    for epoch in range(1, EPOCH + 1):
        # train_data = MyDataloader(args,dataset)
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
        if epoch <= args.warmup_epoch:
            for i, data in enumerate(train_loader):

                '''dict keys : [input_ids,att_mask,aug1_ids,aug1_mask,aug2_ids,aug2_mask,labels,clean_labels,indexes]'''
                st_loss = None
                cl_loss = None
                mlm_loss = None
                input_ids = Variable(data["input_ids"].cuda())
                att_mask = Variable(data["att_mask"].cuda())
                labels = Variable(data["labels"].cuda())

                logits, feature_ori, _, cls_emb = mymodel(input_ids,att_mask)

                ce_loss = F.cross_entropy(logits, labels)

                ST = RkdDistance()
                st_loss = args.beta * ST(logits, feature_ori) 

                if epoch >= 0 and args.alpha > 0:
                    aug1_ids = Variable(data["aug1_ids"].cuda())
                    aug1_mask = Variable(data["aug1_mask"].cuda())
                    aug2_ids = Variable(data["aug2_ids"].cuda())
                    aug2_mask = Variable(data["aug2_mask"].cuda())
                    logits_aug1, feat_aug1, _, cls_aug1 = mymodel(aug1_ids,aug1_mask)
                    logits_aug2, feat_aug2, _, cls_aug2 = mymodel(aug2_ids,aug2_mask)
                    feat_aug1 = feat_aug1.unsqueeze(dim=1)
                    feat_aug2 = feat_aug2.unsqueeze(dim=1)
                    features = torch.cat([feat_aug1, feat_aug2],dim=1) # -> shape [batch_size,2,sentence_len,hidden_size]
                    if args.cl == "supcon":
                        cl_loss = args.alpha * CL(features,labels)
                    else:
                        cl_loss = args.alpha * CL(features)
                    loss = ce_loss + cl_loss + st_loss
                else:
                    loss = ce_loss
                optimizer.zero_grad()
            
                out = F.log_softmax(logits)

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
        else:
            pred_clean, knn_logits = eval_train(args,mymodel,train_loader)
            mymodel.train()
            pred_clean = pred_clean.cpu()
            knn_logits = knn_logits.cpu()
            for i, data in enumerate(train_loader):

                '''dict keys : [input_ids,att_mask,aug1_ids,aug1_mask,aug2_ids,aug2_mask,labels,clean_labels,indexes]'''
                st_loss = None
                cl_loss = None
                mlm_loss = None
                input_ids = Variable(data["input_ids"].cuda())
                att_mask = Variable(data["att_mask"].cuda())
                labels = Variable(data["labels"].cuda())
                indexes = data["indexes"]

                mask_clean = pred_clean[indexes].cuda()             
                logits, feature_ori, _, cls_emb = mymodel(input_ids,att_mask)

                labels_onehot_batch = torch.tensor(np.eye(args.num_class)[labels.cpu().tolist()],dtype=float).cuda()
                knn_logits_batch = knn_logits[indexes].cuda()   
                mask_clean = mask_clean.unsqueeze(dim=-1).expand(logits.shape)

                soft_labels = torch.where(mask_clean,labels_onehot_batch,knn_logits_batch)

                ce_loss = F.cross_entropy(logits, soft_labels)  

                ST = RkdDistance()
                st_loss = args.beta * ST(logits, feature_ori) 

                aug1_ids = Variable(data["aug1_ids"].cuda())
                aug1_mask = Variable(data["aug1_mask"].cuda())
                aug2_ids = Variable(data["aug2_ids"].cuda())
                aug2_mask = Variable(data["aug2_mask"].cuda())
                logits_aug1, feat_aug1, _, cls_aug1 = mymodel(aug1_ids,aug1_mask)
                logits_aug2, feat_aug2, _, cls_aug2 = mymodel(aug2_ids,aug2_mask)
                feat_aug1 = feat_aug1.unsqueeze(dim=1)
                feat_aug2 = feat_aug2.unsqueeze(dim=1)
                features = torch.cat([feat_aug1, feat_aug2],dim=1) # -> shape [batch_size,2,sentence_len,hidden_size]
                if args.cl == "supcon":
                    cl_loss = args.alpha * CL(features,labels)
                else:
                    cl_loss = args.alpha * CL(features)
                loss = ce_loss + cl_loss + st_loss

                optimizer.zero_grad()
            
                out = F.log_softmax(logits)

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
        

        # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"train %d/%d epochs Loss:%f, Acc:%f, Recall:%f" \
        # %(epoch, EPOCH, train_loss / (i + 1), train_acc / len(train_data.dataset), train_recall / len(train_data.dataset)))

        if test_data:
            _, test_last = evaluate(test_loader, epoch, mymodel, 'test')
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
        input_ids, attention_mask, labels = [Variable(elem.cuda()) for elem in batch]
     
        with torch.no_grad():
            logits, tmp1, tmp2, cls_emb = mymodel(input_ids, attention_mask)
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
if args.pre_trained_data.strip() != "none":
    params = torch.load('ckpt/'+args.pre_trained_data)
    params = {k:params[k] for k in params if "classifier" not in k and "projector" not in k}
    mymodel.load_state_dict(params,strict=False)
# for name,params in mymodel.named_parameters():
#     print(name,params.shape)

log_file = open(args.log_path, 'w')

mymodel=torch.nn.DataParallel(mymodel).cuda()

optimizer=optim.Adam(mymodel.parameters(),lr=args.learning_rate)
CL = get_cl_criterion(args)

# eval_train(args,mymodel,train_loader)

test_best_l, test_last_l = train(args, mymodel, optimizer, train_loader, test_loader=test_loader)

print('test_best',test_best_l)
print('test_last',test_last_l)
print("Test best %f , last %f"%(test_best_l[-1], test_last_l[-1]))

print('test_best',test_best_l,file=log_file)
print('test_last',test_last_l,file=log_file)
print("Test best %f , last %f"%(test_best_l[-1], test_last_l[-1]),file=log_file)
log_file.close()