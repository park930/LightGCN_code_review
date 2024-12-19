'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

# 다중 프로세싱 라이브러리 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 명령줄 argument파싱해서 args에 저장
args = parse_args()

# 폴더 구조 정의
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')

# sources폴더를 모듈 경로에 추가
import sys
sys.path.append(join(CODE_PATH, 'sources'))

# FILE_PATH 디렉토리 없으면 생성
if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

# 설정 값 저장 위해
config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book']
all_models  = ['mf', 'lgn']
# config['batch_size'] = 4096
# BPR Loss 계산 위한 배치 크기
config['bpr_batch_size'] = args.bpr_batch
# 잠재 벡터 차원
config['latent_dim_rec'] = args.recdim
# 계층 수
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
# 노드가 유지될 확률
config['keep_prob']  = args.keepprob
# 인접 행렬 분할 수
config['A_n_fold'] = args.a_fold
# 테스트 시, 배치 크기
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
# 학습률
config['lr'] = args.lr
# 가중치 감쇠 비율
config['decay'] = args.decay
# 사전학습 여부
config['pretrain'] = args.pretrain
# 인접 행렬 분할 여부
config['A_split'] = False
config['bigdata'] = False

# CUDA사용 가능 여부 확인
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")

# CPU코어 절반을 CORES에 할당
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

# 데이터셋 이름, 모델 이름 가져옴
dataset = args.dataset
model_name = args.model
# 지원하지 않는거면 에러
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")



# 학습 epoch 수
TRAIN_epochs = args.epochs
# 체크포인트 로드 여부
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# print(logo)
