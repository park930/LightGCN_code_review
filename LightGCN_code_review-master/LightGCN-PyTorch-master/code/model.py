"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np

# nn.Module을 상속받는 모델 기본 클래스
class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
# BasicModel 상속 받는 클래스
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    
    # BPR 손실 함수 구현위한 추상 메서드
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError

# 순수 행렬 분해(Matrix Factorization) 모델 구현
# config=모델의 하이퍼파라미터 , dataset=학습 데이터셋
class PureMF(BasicModel):
    def __init__(self, 
                config:dict, 
                dataset:BasicDataset):
        super(PureMF, self).__init__()
        
        # 사용자 수, 아이템 수 설정
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        # 잠재 변수의 차원 나타냄 (config에서 가져옴)
        self.latent_dim = config['latent_dim_rec']
        # f=시그모이드 함수 (활성화함수)
        self.f = nn.Sigmoid()
        # 모델의 임베딩 레이어 초기화
        self.__init_weight()
        
    # 사용자 임베딩, 아이템 임베딩의 임베딩 레이어 초기화
    def __init_weight(self):
        # nn.Embedding = 고정된 벡터를 학습하는 Layer
        # 사용자, 아이템에 대한 잠재 벡터 생성
        # 정규 분포 이용해서 초기화함
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    # 특정 사용자들에 대한 예측 평점 계산
    def getUsersRating(self, users):
        # users를 정수형 텐서로
        users = users.long()
        # 사용자 임베딩 얻음 (사용자 잠재 벡터)
        # 특정 사용자 인덱스에 해당하는 임베딩 벡터 가져옴
        users_emb = self.embedding_user(users)
        # item.weight를 통해 모든 아이템에 대한 잠재 벡터를 얻기 위해
        # embedding_item이 학습하는 아이템 들의 임베딩 가중치 행렬에 접근
        items_emb = self.embedding_item.weight
        # 사용자, 아이템 벡터의 내적으로 사용자-아이템의 예측 평점 구함
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    # BPR 손실함수 구현
    def bpr_loss(self, users, pos, neg):
        # 사용자 임베딩
        users_emb = self.embedding_user(users.long())
        # 긍정 아이템 임베딩
        pos_emb   = self.embedding_item(pos.long())
        # 부정 아이템 임베딩
        neg_emb   = self.embedding_item(neg.long())
        # 사용자-긍정 아이템의 점수
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        # 사용자-부정 아이템의 점수
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        
        # BPR손실 계산
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        # L2 정규화
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                        pos_emb.norm(2).pow(2) + 
                        neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    # 모델의 순전파
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        # 유저, 아이템의 임베딩 얻음
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        # 이를 통해 사용자-아이템 예측 점수 계산
        scores = torch.sum(users_emb*items_emb, dim=1)
        # 예측 점수에 시그모이드 적용
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                config:dict, 
                dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        # 임베딩 레이어,기타 초기화 작업
        self.__init_weight()

    def __init_weight(self):  
        # 데이터셋에서 사용자, 아이템 가져옴 
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        
        # 잠재 벡터 차원 수 가져옴
        self.latent_dim = self.config['latent_dim_rec']
        # conv 레이어 수
        self.n_layers = self.config['lightGCN_n_layers']
        # 드롭 아웃 확률
        self.keep_prob = self.config['keep_prob']
        # 인접 행렬 분할 여부
        self.A_split = self.config['A_split']
        
        # 사용자를 위한 임베딩 레이어 (각 사용자는 latent_dim 차원의 벡터로 임베딩)
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        
        # 아이템을 위한 임베딩 레이어 (각 아이템은 latent_dim 차원의 벡터로 임베딩)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        
        # pretrain이 0이면, 정규분포로 사용자,아이템 임베딩을 초기화
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            # 표준 편차 0.1로 설정해서 초기값 설정
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        
        # 사전 학습된 임베딩 써서 사용자,아이템 임베딩 초기화 
        else:
            # user_emb, item_emb는 사전 학습된 임베딩 벡터가 저장된 배열
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        
        # 활성화 함수 = 시그모이드
        self.f = nn.Sigmoid()
        # 데이터셋에서 희소 그래프 가져옴 (사용자-아이템 관계)
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
        
    
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        # 희소 행렬의 인덱스
        index = x.indices().t()
        # 희소 행렬의 값
        values = x.values()
        
        # 드롭 아웃에서 값 랜덤하게 선택하기 위해
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        # 랜덤 값을 keep_prob로 나눠서 보정
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    # 전체 그래프에 드롭아웃 적용
    def __dropout(self, keep_prob):
        # 분할 O = 각 분할된 그래프에 대해서 드롭아웃 적용
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        # 분할 X = 하나의 전체 그래프에 드롭아웃 적용
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    # 그래프 전파
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        
        # 사용자, 아이템 임베딩을 한번에 결합한 텐서
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            # 학습 중일 때, 드롭아웃 적용
            if self.training:
                print("droping")
                # 학습 중인 그래프에 드롭 아웃 적용하기 위해 __dropout호출
                g_droped = self.__dropout(self.keep_prob)
            else:
                # 원본 그래프에 드롭아웃 적용
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        # 그래프 전파
        for layer in range(self.n_layers):
            # 분할된 경우
            if self.A_split:
                temp_emb = []
                # 분할된 그래프 각각에 대해서
                for f in range(len(g_droped)):
                    # 각 레이어에서 희소 행렬 곱을 사용해서 사용자-아이템 임베딩 갱신
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                # 마찬가지로 희소행렬 곱으로 사용자-아이템 임베딩 갱신
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        
        # 각 레이어의 임베딩 쌓아놓은거
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        # 각 사용자-아이템의 최종 임베딩 구함
        light_out = torch.mean(embs, dim=1)
        # 사용자와 아이템 임베딩을 분리
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        # 사용자, 아이템의 최종 임베딩 계산 -> all_users, all_items 얻음
        all_users, all_items = self.computer()
        # 특정 users에 대한 임베딩을 추출함
        users_emb = all_users[users.long()]
        items_emb = all_items
        # 사용자, 아이템 임베딩의 내적을 계산 -> 사용자-아이템 간의 예측 평점 행렬
        # users_emb = [유저 수, latent_dim] , items_emb.t = [latent_dim, 아이템 수] ===> 결과 [유저 수, 아이템 수]
        # 시그모이드 적용
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        # 사용자, 아이템의 최종 임베딩 계산 -> all_users, all_items 얻음
        all_users, all_items = self.computer()
        # 특정 users의 임베딩 얻음
        users_emb = all_users[users]
        # pos_items 인덱스에 해당하는 긍정 아이템 임베딩
        pos_emb = all_items[pos_items]
        # neg_items 인덱스에 해당하는 부정 아이템 임베딩
        neg_emb = all_items[neg_items]
        # 사용자, 아이템의 초기 임베딩을 추출 (그래프 전파 전의 값)
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        # 위의 getEmbedding으로 users, pos, neg에 해당하는 임베딩들 얻음
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        # L2 정규화를 초기 임베딩에 적용
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                        posEmb0.norm(2).pow(2)  +
                        negEmb0.norm(2).pow(2))/float(len(users))
        # 사용자, 긍정 아이템 임베딩의 각 요소별 곱셈 수행
        pos_scores = torch.mul(users_emb, pos_emb)
        # 요소별 곱의 합을 구함 ==> 사용자-긍정아이템 점수 구함
        pos_scores = torch.sum(pos_scores, dim=1)
        
        # 사용자, 부정 아이템 임베딩의 각 요소별 곱셈 수행
        neg_scores = torch.mul(users_emb, neg_emb)
        # 요소별 곱의 합을 구함 ==> 사용자-부정아이템 점수 구함
        neg_scores = torch.sum(neg_scores, dim=1)
        
        # softplus = log(1 + exp(x)) ==> ReLu함수
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        # loss = BPR손실 , reg_loss = 정규화 손실
        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        # 사용자, 아이템의 최종 임베딩 계산 -> all_users, all_items 얻음
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        # 특정 users에 대한 임베딩
        users_emb = all_users[users]
        # 특정 items에 대한 임베딩
        items_emb = all_items[items]
        # 사용자, 아이템 임베딩의 각 요소별 곱을 구함
        inner_pro = torch.mul(users_emb, items_emb)
        # 요소별 곱의 합을 계산 = 사용자-아이템 간의 예측 점수 구함
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
