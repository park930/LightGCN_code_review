'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset
from time import time
from model import LightGCN
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import random
import os

# C++파일을 불러오기 위해
try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    
    # 랜덤 샘플링 시, 재현성을 위해 시드 설정
    sampling.seed(world.seed)
    # 로드 성공 시, true
    sample_ext = True
except:
    world.cprint("Cpp extension not loaded")
    sample_ext = False

# BPR 손실 구현한 클래스
class BPRLoss:
    # Pairwise방식의 모델
    def __init__(self,
                recmodel : PairWiseModel,
                config : dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        # Adam 옵티마이저 생성하여 모델 파라미터 업데이트
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    # BPR 손실 계산 -> 모델 업데이트
    def stageOne(self, users, pos, neg):
        # 사용자-아이템의 BPR 손실
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        # 정규화 손실 계산
        reg_loss = reg_loss*self.weight_decay
        # 최종 손실 계산
        loss = loss + reg_loss

        # 옵티마이저 초기화, 역전파, 가중치 갱신 수행
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # CPU에서 item 호출
        return loss.cpu().item()

# BPR 학습 위해, 사용자-아이템 샘플링
def UniformSample_original(dataset, neg_ratio = 1):
    dataset : BasicDataset
    # 각 사용자의 긍정 아이템
    allPos = dataset.allPos
    start = time()
    # C++ 확장이 Load 시, sample_negative 호출
    if sample_ext:
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                    dataset.trainDataSize, allPos, neg_ratio)
    else:
        S = UniformSample_original_python(dataset)
    # 샘플링된 사용자-긍정-부정 아이템 배열
    return S

# 파이썬으로 BPR 샘플링
def UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    # 학습 data 크기
    user_num = dataset.trainDataSize
    # n_users만큼 무작위로 사용자 샘플링
    users = np.random.randint(0, dataset.n_users, user_num)
    # 모든 사용자-긍정 아이템 
    allPos = dataset.allPos
    
    # 샘플링 값 저장할 리스트
    S = []
    sample_time1 = 0.
    sample_time2 = 0.

    # 각 사용자에 대해
    for i, user in enumerate(users):
        start = time()
        # 각 사용자의 긍정 아이템 추출
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        
        # 무작위로 긍정, 부정 아이템 샘플링
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            # 부정 아이템은 상호작용 하지 않은 것으로
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    
    # 사용자-긍정-부정 아이템을 numpy배열로 반환 
    return np.array(S)

# ===================end samplers==========================
# =====================utils====================================

# 학습,샘플링 과정의 재현성을 위해 시드 설정
def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

# 모델 종류에 따라 파일 이름 생성 다르게
def getFileName():
    if world.model_name == 'mf':
        file = f"mf-{world.dataset}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == 'lgn':
        file = f"lgn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    return os.path.join(world.FILE_PATH,file)

# 데이터를 배치 크기로 나눔
# tensors = 나눌 대상 , kwargs = 배치 크기
def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    # 텐서가 1개
    if len(tensors) == 1:
        tensor = tensors[0]
        # 텐서를 배치 크기로 나누기 위해 
        for i in range(0, len(tensor), batch_size):
            # 배치 크기로 슬라이싱
            yield tensor[i:i + batch_size]
    
    # 텐서가 여러개
    else:
        # 첫번째 텐서의 길이를 기준으로 인덱스 설정
        for i in range(0, len(tensors[0]), batch_size):
            # 여러 텐서를 배치 크기 단위로 슬라이싱
            yield tuple(x[i:i + batch_size] for x in tensors)

# 여러 배열을 섞기 위해
def shuffle(*arrays, **kwargs):
    # indicies 값을 가져옴
    require_indices = kwargs.get('indices', False)

    # 입력 배열 arrays의 길이 확인
    # 모든 배열이 동일한 길이일때만 수행
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                        'the same length.')

    # 첫번째 배열의 길이를 인덱스 배열로
    shuffle_indices = np.arange(len(arrays[0]))
    # 인덱스 배열을 랜덤으로 섞음
    np.random.shuffle(shuffle_indices)

    # arrays가 1개일때,
    if len(arrays) == 1:
        # 이것만 섞음
        result = arrays[0][shuffle_indices]
    else:
        # 입력 배열 여러개면, shuffle_indices로 동일하게 섞음
        result = tuple(x[shuffle_indices] for x in arrays)

    # true이면, 섞은 인덱스도 같이 반환
    if require_indices:
        return result, shuffle_indices
    else:
        return result

# 코드 실행시간 측정 위해
class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    # 전역 시간 기록 위해
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    # 전역 시간 기록에서 마지막 값 가져옴
    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    # timer의 name_type값을 반환
    # key를 선택해서 반환
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    # name_type값을 0으로 설정
    # key를 선택해서 초기화 가능
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    # timer초기화
    def __init__(self, tape=None, **kwargs):
        # name키워드가 주어진 경우
        if kwargs.get('name'):
            # name_type의 name값을 name키워드 값으로 설정
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    # with블록 시간 시, 현재 시간 기록
    def __enter__(self):
        self.start = timer.time()
        return self

    # with 블록 나갈 때, 실행
    def __exit__(self, exc_type, exc_val, exc_tb):
        # name이 있으면
        if self.named:
            # name_type의 값으로 종료시점-시작시점을 기록
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            # 아니면, tape에 저장
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================

# test_data와 r(예측결과)를 상위 k 기준에서 Recall, Precision 계산
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    # 각 사용자에 대해 상위 K 범위의 아이템 개수
    # sum(1) = 행 단위로 합함
    right_pred = r[:, :k].sum(1)
    # Precision의 분모는 K
    precis_n = k
    # 각 사용자마다 정답 개수 저장
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    # 각 사용자의 재현률 계산 -> 합산
    recall = np.sum(right_pred/recall_n)
    # 맞춘 아이템의 합을 precis_n으로 나눔 = 평균 정밀도
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


# Mean Reciprocal Rank 계산 (r=예측 결과, k=상위 k의 범위)
def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    # 각 사용자의 상위 k에 대한 예측 결과
    pred_data = r[:, :k]
    # Reciprocal Rank 계산을 위한 가중치
    # 순위 낮을수록 기여도 낮아짐
    scores = np.log2(1./np.arange(1, k+1))
    # 각 사용자의 Reciprocal Rank 계산
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    # 전체 사용자의 평균 반환
    return np.sum(pred_data)

# Normalized Discounted Cumulative Gain 계산
def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    # r과 test_data 길이가 같은지
    assert len(r) == len(test_data)

    # 각 사용자의 상위 K의 예측 값
    pred_data = r[:, :k]

    # 테스트 데이터 정답을 이진 행렬로 변환
    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        # 각 사용자마다 상위 k에 해당하는 위치에 1 할당
        test_matrix[i, :length] = 1
    
    # 이상적인 DCG 계산
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    
    # 실제 DCG 계산
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    
    # idcg가 0일 때를 위해
    idcg[idcg == 0.] = 1.

    # NDCG계산
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    
    # 모든 사용자에 대해 평균 NDCG 계산
    return np.sum(ndcg)

# AUC 계산위해
# all_item_scores = 모든 아이템 예측 점수
def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    # 모든 아이템 개수 만큼 0으로 초기화한 배열 생성
    r_all = np.zeros((dataset.m_items, ))
    # 테스트 데이터 위치에 1로
    r_all[test_data] = 1
    # 예측 점수가 0 이상인 것만
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    # ROC AUC 점수 계산, 반환
    return roc_auc_score(r, test_item_scores)

# 예측 데이터에서 테스트 데이터의 정답 여부
def getLabel(test_data, pred_data):
    r = []
    # 각 사용자별 테스트 데이터,예측 데이터 가져옴
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        
        # 예측 값이 groundTrue(실제값)에 포함되어있는지 
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        # true/false를 float로 변환
        pred = np.array(pred).astype("float")
        # 예측 결과를 리스트에 추가
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================
