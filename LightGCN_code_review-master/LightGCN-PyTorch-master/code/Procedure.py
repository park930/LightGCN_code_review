'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score

# CPU 코어 수의 절반을 계산 (병럴 처리를 위해)
CORES = multiprocessing.cpu_count() // 2

# BPR기반 학습 수행
# recommend_model = 추천 모델 / loss_class = 손실 계산 객체 / neg_k = 부정 샘플 개수
def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    # Recmodel에 추천 모델 할당
    Recmodel = recommend_model
    Recmodel.train()
    # utils의 BPRLoss 클래스 객체로 bpr할당
    bpr: utils.BPRLoss = loss_class
    
    # 
    with timer(name="Sample"):
        # 데이터를 균일 샘플링 -> 사용자-긍정-부정 아이템 모음 생성
        S = utils.UniformSample_original(dataset)
    # S로부터 users, posItems, negItems를 추출
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    # 데이터를 device에 이동
    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    # 데이터 섞음 = 학습 순서를 무작위로
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    # 배치의 총 개수 계산 = 전체 User 수 / 배치 사이즈
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    # 평균 손실 초기화
    aver_loss = 0.
    
    # 배치 크기로 데이터 나눠서 반복문
    # 반복문에서 유저,아이템의 배치를 추출함
    for (batch_i,
        (batch_users,
        batch_pos,
        batch_neg)) in enumerate(utils.minibatch(users,
                                                posItems,
                                                negItems,
                                                batch_size=world.config['bpr_batch_size'])):
        # 배치 데이터로 손실 계산
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        # 손실 누적
        aver_loss += cri
        
        # 텐서보드 사용 O = 배치의 손실을 로깅
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    
    # 평균 손실 계산 
    aver_loss = aver_loss / total_batch
    # 수행 시간 기록
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
# 하나의 배치에 대해 테스트 수행 
def test_one_batch(X):
    # X[0] = 예측값
    sorted_items = X[0].numpy()
    # X[1] = 실제값
    groundTrue = X[1]
    
    # 예측값(추천된 아이템)의 정답 여부를 binary label로 
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []

    # world.topks = 평가할 상위 k개 리스트
    for k in world.topks:
        # Recall, Precision값의 dict 생성
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        
        # Precision, Recall, NDCG 값 계산
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    
    # recall, pre, ndcg를 numpy 배열로 반환
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
# 테스트 데이터셋으로 모델 성능 평가
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN

    # 모델 평가모드로 전환, 드롭아웃 X
    Recmodel = Recmodel.eval()
    # 평가에 사용할 상위 K값
    max_K = max(world.topks)

    # multicore 1개면
    if multicore == 1:
        # CPU 코어수로 멀티프로세싱 풀 만듦
        pool = multiprocessing.Pool(CORES)
    
    # Precision, recall, ndcg값 저장할 dict 생성
    results = {'precision': np.zeros(len(world.topks)),
            'recall': np.zeros(len(world.topks)),
            'ndcg': np.zeros(len(world.topks))}
    
    # 테스트 데이터 준비 과정
    # 그래디언트 사용 X
    with torch.no_grad():
        # 테스트 데이터에서 users 추출
        users = list(testDict.keys())
        
        # 배치 크기 너무 큰지 확인 
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        
        # 사용자 배치 저장할 리스트
        users_list = []
        # 추천 점수 저장할 리스트
        rating_list = []
        # 실제 정답 저장할 리스트
        groundTrue_list = []
        # auc_record = []
        # ratings = []

        # 총 배치 수 계수 
        total_batch = len(users) // u_batch_size + 1

        # 사용자별 추천 결과 생성
        # 사용자 데이터를 배치 크기로 나눔
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            # 배치 사용자에 대한 긍정 아이템 추출
            allPos = dataset.getUserPosItems(batch_users)
            # 배치 사용자에 대한 실제 정답 리스트 생성
            groundTrue = [testDict[u] for u in batch_users]
            
            # 배치 사용자를 텐서로 변환 -> device로 이동
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            # 배치 사용자에 대한 전체 아이템의 추천 점수 계산
            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()

            # 이미 학습에 쓴거 제외하기 위해
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            
            # 이미 사용한 긍정 아이템의 점수를 매우 낮게 설정
            rating[exclude_index, exclude_items] = -(1<<10)
            # 상위 max_k개의 아이템 추출
            _, rating_K = torch.topk(rating, k=max_K)
            
            # 추천 점수 CPU로 이동 -> rating_list, groundTrue_list에 저장
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)

        # 배치 결과가 전체 배치 수와 같은지 확인 
        assert total_batch == len(users_list)

        # 예측값, 정답을 하나로 묶음
        X = zip(rating_list, groundTrue_list)
        
        # 멀티코어 O = 병렬로 배치 처리
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        # 멀티코어 X = test_one_batch 함수 순차적으로 호출
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        
        scale = float(u_batch_size/len(users))
        # 각 배치의 Recall, Precision, NDCG값을 dict에 누적
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        
        # 누적값을 사용자 수로 나눠서 평균화
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)

        # 텐서보드 O = 각 K값에 대한 성능 지표 값 로깅
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                        {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                        {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                        {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        
        # 멀티프로세싱 풀 닫음 , 최종 값 출력
        if multicore == 1:
            pool.close()
        print(results)
        return results
