import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
src_root_path = '/home/taskarithmetic/'
checkpoint_path = '/home/taskarithmetic/checkpoints/'
dataset_path = '/home/taskarithmetic/data/'
sys.path.append(src_root_path)

import time
import tqdm
import torch
from task_vectors import TaskVector
from args import parse_arguments
from utils import create_log_dir
import pickle

from eval import eval_single_dataset_preprocess_mapping_head
from datasets.registry import get_dataset
from datasets.common import maybe_dictionarize, get_dataloader_shuffle
from merging_model import ModelWrapper, AlphaWrapper_Surgery, make_functional

# Config
exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD'] # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD
learn_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']

method_name = 'task_arithmetic' # choose: weight_averaging | task_arithmetic | ties_merging | tw_adamerging | lw_adamerging | tw_adamergingpp | lw_adamergingpp
model_name = 'ViT-B-32'  #  choose: ViT-B-32 | ViT-B-16 | ViT-L-14

is_visualization = True
iterations = 500
eval_iterations = 100

args = parse_arguments()
args.rank = 304
args.method_name = method_name
args.model_name = model_name
args.data_location = dataset_path
args.save = checkpoint_path + model_name

if is_visualization:
    args.logs_path = src_root_path + 'src/logs/Surgery/visualization/' + model_name
else:
    args.logs_path = src_root_path + 'src/logs/Surgery/run/' + model_name

pretrained_checkpoint = checkpoint_path + model_name+'/zeroshot.pt'

log_name = 'log_{}_{}_{}_rank_{}_{}'.format(str(__file__.split("/")[-1].split(".")[0]), method_name, model_name, str(args.rank), time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())))
log = create_log_dir(args.logs_path, log_name+'.txt')
log.info(str(args))

# Create the task vectors
if method_name in ['ties_merging', 'tw_adamergingpp', 'lw_adamergingpp']:
    # TIES Merging
    from ties_merging_utils import *

    ft_checks = [torch.load(checkpoint_path + model_name + '/' + dataset_name + '/finetuned.pt').state_dict() for dataset_name in exam_datasets]
    ptm_check = torch.load(pretrained_checkpoint).state_dict()
    check_parameterNamesMatch(ft_checks + [ptm_check])
    remove_keys = []

    flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks])
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)

    tv_flat_checks = flat_ft - flat_ptm
    assert check_state_dicts_equal(vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check)
    assert all([check_state_dicts_equal(vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i]) for i in range(len(ft_checks))])
    selected_entries, merged_tv = ties_merging_split(tv_flat_checks, reset_thresh=20, merge_func="dis-sum", )

    ties_task_vectors = []
    for vector_ in selected_entries:
        t_state_dict = vector_to_state_dict(vector_, ptm_check, remove_keys=remove_keys)
        ref_model = torch.load(pretrained_checkpoint)
        ref_model.load_state_dict(t_state_dict, strict=False)
        ties_task_vectors.append(ref_model.state_dict())

elif method_name in ['weight_averaging', 'task_arithmetic', 'tw_adamerging', 'lw_adamerging']:
    # Task Vector
    task_vectors = [TaskVector(pretrained_checkpoint, checkpoint_path + model_name + '/' + dataset_name + '/finetuned.pt') for dataset_name in exam_datasets]

else:
    print('method name error!')
    exit(-1)

pretrained_model = torch.load(pretrained_checkpoint)
pretrained_model_dic = pretrained_model.state_dict()

model = ModelWrapper(pretrained_model)
model = model.to(args.device)
_, names = make_functional(model)

paramslist = []
paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in pretrained_model_dic.items())] # pretrain
if method_name in ['ties_merging', 'tw_adamergingpp', 'lw_adamergingpp']:
    paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in sd.items())  for i, sd in enumerate(ties_task_vectors)] # task vectors
elif method_name in ['weight_averaging', 'task_arithmetic', 'tw_adamerging', 'lw_adamerging']:
    paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in sd.vector.items()) for i, sd in enumerate(task_vectors)]  # task vectors

torch.cuda.empty_cache()
alpha_model = AlphaWrapper_Surgery(paramslist, model, names, exam_datasets, args)

# Init Evaluate
Total_ACC = 0.
for dataset_name in learn_datasets:
    image_encoder = alpha_model.get_image_encoder()
    classification_head = alpha_model.get_classification_head(dataset_name)
    down_proj, up_proj = alpha_model.get_feature_mapping_to_head(dataset_name)
    metrics = eval_single_dataset_preprocess_mapping_head(image_encoder, classification_head, dataset_name, args, down_proj, up_proj)
    Total_ACC += metrics['top1']
    log.info('Ref: step 0: ' + ' dataset: ' + str(dataset_name) + ' ACC: ' + str(metrics['top1']))

if is_visualization:
    pass