"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time

# 추상 클래스
# 파이토치의 DataSet을 상속 받아서 데이터셋의 기본 구조 정의
class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    # 데이터셋의 사용자 수를 반환
    @property
    def n_users(self):
        raise NotImplementedError
    
    # 데이터셋의 아이템 수를 반환
    @property
    def m_items(self):
        raise NotImplementedError
    
    # 학습 데이터셋의 크기 반환
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    # 테스트 데이터셋의 사용자-아이템 dict를 반환
    @property
    def testDict(self):
        raise NotImplementedError
    
    # 모든 사용자에 대한 긍정 상호작용 반환
    @property
    def allPos(self):
        raise NotImplementedError
    
    # 사용자-아이템의 피드백 반환
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    # 특정 유저의 긍정 아이템 목록 반환
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    # 특정 유저의 부정 아이템 목록 반환
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    # 희소 그래프 생성, 반환
    # 파이토치의 희소텐서 형식으로 반환
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

# BasicDataSet 상속 받아서 구현
class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """
    # lastFM 데이터셋 초기화
    def __init__(self, path="../data/lastfm"):
        # train or test
        cprint("loading [last fm]")
        
        # 각 학습/테스트 모드 관리
        # 학습/테스트에 따라 다른 데이터를 사용하기 위해
        self.mode_dict = {'train':0, "test":1}
        # 초기 = train
        self.mode    = self.mode_dict['train']
        
        # self.n_users = 1892
        # self.m_items = 4489
        # train은 data1.txt에서 데이터 로드
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        # print(trainData.head())

        # test는 test1.txt에서 데이터 로드
        testData  = pd.read_table(join(path, 'test1.txt'), header=None)
        # print(testData.head())
        
        # 사용자 간의 trustNet을 trustnetwork.txt에서 로드하고 numpy 배열로 변환
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        # print(trustNet[:5])
        
        # 인덱스 조정해서 0부터 시작하도록
        trustNet -= 1
        trainData-= 1
        testData -= 1
        
        # 클래스의 속성으로 저장
        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData
        
        # train데이터에서 user ID들 추출
        self.trainUser = np.array(trainData[:][0])
        # 고유 user ID들 추출
        self.trainUniqueUsers = np.unique(self.trainUser)
        # train데이터에서 item ID들 추출
        self.trainItem = np.array(trainData[:][1])
        
        # self.trainDataSize = len(self.trainUser)
        # train데이터와 마찬가지로 test 데이터도 user, item ID들 추출
        self.testUser  = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:][1])
        
        # 희소성 계산 + 출력
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")
        
        # (users,users)
        # trustNet으로 희소 행렬 생성 (socialNet)
        # 사용자 간의 trustNet을 그래프 형식으로
        self.socialNet    = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users))
        
        # (users,items), bipartite graph
        # train데이터의 사용자-아이템 상호작용으로 희소 행렬 생성
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items)) 
        
        # 전체 사용자의 긍정 아이템 목록 계산
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        
        # 전체 사용자의 부정 아이템 목록 계산
        self.allNeg = []
        # '전체-긍정=부정'을 이용해서
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        
        # build_test함수로 테스트 데이터를 dict형태로 저장
        self.__testDict = self.__build_test()

    # 이전의 BasicDataset클래스의 추상 메서드 구현
    @property
    def n_users(self):
        return 1892
    
    @property
    def m_items(self):
        return 4489
    
    @property
    def trainDataSize(self):
        return len(self.trainUser)
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos


    # 사용자-아이템 네트워크로 희소 그래프 생성
    def getSparseGraph(self):
        # 그래프 생성되지 않은 경우
        if self.Graph is None:
            # 훈련 데이터에서 사용자, 아이템의 id 나타내는 리스트 (정수형 텐서)
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)
            
            # 사용자-아이템 간의 연결 나타내는 인덱스
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            # 아이템-사용자 간의 연결 나타내는 인덱스
            second_sub = torch.stack([item_dim+self.n_users, user_dim])
            
            # 사용자-아이템, 아이템-사용자 인덱스를 열 방향으로 연결
            index = torch.cat([first_sub, second_sub], dim=1)
            # 각 연결의 값을 1로 설정
            data = torch.ones(index.size(-1)).int()
            # 위의 index, data로 희소 인접 행렬 생성
            self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            # 희소 행렬 -> 밀집 행렬로 변환 (모든 요소 저장되는 행렬)
            dense = self.Graph.to_dense()
            
            # 각 행의 합 계산 == 차수 구함 (행렬에서 해당 노드가 몇 번 나오는지) 
            D = torch.sum(dense, dim=1).float()
            # 분모가 0인 경우를 위해 차수가 0인 것을 1로
            D[D==0.] = 1.
            # D의 제곱근 계산해서 정규화에 사용
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            # dense(밀집행렬)를 D_sqrt로 나눠서 정규화함
            dense = dense/D_sqrt
            # 양방향 정규화 (열에 대해서도)
            dense = dense/D_sqrt.t()
            # 0이 아닌 값들의 index (희소 행렬로 변환할 때 사용)
            index = dense.nonzero()
            # 1e-9보다 작은 값은 무시함 (너무 작아서)
            data  = dense[dense >= 1e-9]
            # index와 data의 길이가 같은지
            assert len(index) == len(data)
            
            # index, data를 이용해서 희소 행렬 생성 (정규화된 희소 행렬)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            # coalesce = 중복된 인덱스 병합
            self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    # 테스트 데이터를 사용자-아이템 dict 형태로 변환
    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        
        # key를 user ID , value를 user와 상호작용한 item ID 리스트로 설정 
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    # 특정 사용자, 아이템의 상호작용 여부
    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))
    
    # 특정 사용자의 긍정 아이템 ID 리스트 반환
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    # 특정 사용자의 부정 아이템 ID 리스트 반환
    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems
            
    # 특정 index의 사용자 ID를 반환
    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user
    
    # 데이터셋 모드를 test로 변경
    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']
    
    # train 데이터에서 고유 사용자 수 반환
    def __len__(self):
        return len(self.trainUniqueUsers)

# BasicDataset을 상속 받으로 특정 DataSet을 다루기 위해 (gowalla)
class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self,config = world.config,path="../data/gowalla"):
        # train or test
        cprint(f'loading [{path}]')
        
        # 그래프 분할 방식
        self.split = config['A_split']
        # 분할 시, fold 수
        self.folds = config['A_n_fold']
        # 학습/테스트 모드 설정
        self.mode_dict = {'train': 0, "test": 1}
        # 초기 = train
        self.mode = self.mode_dict['train']
        # 사용자, 아이템 수 초기화
        self.n_user = 0
        self.m_item = 0
        # 학습/테스트 data파일 경로
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        # 학습 데이터 Load
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    # 각 줄의 1번째 = user id, 나머지 = items
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
                    
        # 리스트를 numpy 배열로 변환
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        # 테스트 데이터 Load
        # 위의 훈련 데이터 Load와 동일
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
                    
        # 사용자, 아이템의 ID는 0부터 시작이므로 +1
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        # 데이터셋 정보 출력
        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        # 사용자-아이템 관계를 희소행렬로 생성
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                    shape=(self.n_user, self.m_item))
        # 각 사용자들의 총 상호작용 수 (=차수)
        # 차수가 0이면 1로 (0으로 나누는 것 방지)
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        
        # 각 아이템들의 총 상호작용 수 (=차수)
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        
        # pre-calculate
        # 모든 사용자의 긍정 아이템 ID 리스트 계산
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        # 테스트 데이터를 사용자-아이템 dict로 변환
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    # BasicDataSet 클래스의 추상 메서드 구현
    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    # 희소행렬 A를 지정된 fold만큼 분할
    def _split_A_hat(self,A):
        A_fold = []
        # 각 fold의 크기
        fold_len = (self.n_users + self.m_items) // self.folds
        
        for i_fold in range(self.folds):
            # A 행렬을 fold 만큼 균등하게 나누기 위해 start, end 구함
            # 현재 fold의 start 설정
            start = i_fold*fold_len
            # A의 전체 크기가 fold수의 배수가 아닌 경우에 대한 if문
            if i_fold == self.folds - 1:
                # 마지막 fold가 남은 데이터 모두 가져가도록
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            
            # 희소 행렬을 희소 텐서로 변환
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        # 분할된 희소 텐서의 리스트 반환
        return A_fold

    # 희소 행렬 -> 희소 텐서 변환
    def _convert_sp_mat_to_sp_tensor(self, X):
        # COO 형식으로 변환
        coo = X.tocoo().astype(np.float32)
        # 행/열 인덱스를 텐서로 변환
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        # [row, col]로 구성된 index 텐서
        index = torch.stack([row, col])
        # 행렬 data 값
        data = torch.FloatTensor(coo.data)
        # 희소 텐서 생성
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    # 인접행렬 생성
    def getSparseGraph(self):
        print("loading adjacency matrix")
        # 인접행렬을 아직 생성하지 않은 경우
        if self.Graph is None:
            try:
                # 저장된 인접행렬 파일이 있으면 load
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                # UserItemNet을 통해 인접행렬 A 생성
                print("generating adjacency matrix")
                s = time()
                # dict 기반으로 희소 행렬을 저장 (크기 = 사용자 수 + 아이템 수)
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                # lil matrix 형식으로 (lil matrix = 행렬을 리스트 기반으로 저장)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()    # 사용자-아이템 상호작용에 대한 희소 행렬
                # 양방향 연결 갖는 그래프 생성 위해
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                # lil 형식을 다시 dict 기반으로 저장
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                # 정규화를 위해 
                # rowsum = 각 노드의 차수
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                # 무한 값을 0으로 처리
                d_inv[np.isinf(d_inv)] = 0.
                # 대각 행렬 생성
                d_mat = sp.diags(d_inv)
                
                # 인접행렬을 정규화 (각 노드의 차수로 scale조정)
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                # CSR 형식으로 변환 (메모리 효율성 좋음)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                # 정규화된 인접행렬 저장
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                # split이 true면, 인접행렬을 분할
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                # 아닐 경우, 전체 행렬을 희소 텐서로 변환
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                # coalesce는 중복된 인덱스 결합
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    # 테스트 데이터를 사용자-아이템 dict 형태로 변환
    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    # 특정 use-item에 대한 상호작용 여부
    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    # 각 사용자에 대한 긍정 아이템 ID 리스트 반환
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems
