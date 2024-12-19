'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    # 명령행의 argument 처리 위한 객체 생성
    parser = argparse.ArgumentParser(description="Go lightGCN")
    
    # BPR 손실 학습의 배치 크기 설정
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    # 임베딩 벡터 크기 설정
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    # LightGCN Layer 개수 설정
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    # 학습률 설정
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    # Weight Decay 설정 (L2 정규화 - 과적합 방지)
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    # 드롭 아웃 설정
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    # 드롭 아웃에서 뉴런 유지할 확률 설정 (1.0에 가까울수록 효과 줄어듦)
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    # large 인접행렬을 나눌 분할 수 설정
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    # 테스트할 때, 병렬 처리 위해 사용자 배치 크기 설정
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    # 사용할 dataset 이름
    parser.add_argument('--dataset', type=str,default='gowalla',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    # 모델 가중치 저장할 경로
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    # 상위 몇 개를 평가할 지 
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    # tensorboard 사용 여부
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    # tensorborad 디렉토리 이름에 추가할 comment
    parser.add_argument('--comment', type=str,default="lgn")
    # 저장된 모델 가중치 로드 여부 (1=사용)
    parser.add_argument('--load', type=int,default=0)
    # 학습의 epoch 수
    parser.add_argument('--epochs', type=int,default=1000)
    # 테스트에서 멀티코어 사용할 지
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    # 사전 학습된 가중치 사용할 지
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    # 랜덤 시드
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    # 사용할 모델
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    # argument들 파싱해서 반환
    return parser.parse_args()
