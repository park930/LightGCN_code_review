import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
# 난수 시드 설정
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

# register파일의 MODELS dict에서 Key가 world.model_name인 모델을 가져와서 초기화
# 해당 모델은 world.config, dataset을 입력으로 받음
Recmodel = register.MODELS[world.model_name](world.config, dataset)
# 모델을 world.device로 이동시킴
Recmodel = Recmodel.to(world.device)
# utils.py 파일의 BPRLoss 클래스로 모델의 손실함수를 초기화함
bpr = utils.BPRLoss(Recmodel, world.config)

# 저장할 파일 이름 설정
weight_file = utils.getFileName()
print(f"load and save to {weight_file}")

# 모델의 가중치를 load할지
if world.LOAD:
    try:
        # 저장된 가중치를 불러옴 -> CPU에서 load하도록
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        # 파일 없으면 학습 처음부터 하도록
        print(f"{weight_file} not exists, start from beginning")

# 부정 샘플링 개수
Neg_k = 1

# init tensorboard
if world.tensorboard:
    # tensorboard 저장할 디렉토리 생성
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    # tensorboard 비활성화 = w를 none으로
    w = None
    world.cprint("not enable tensorflowboard")

try:
    # 전체 train_Epoch 수만큼 반복
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        # 10 Epoch마다 테스트 실행
        if epoch %10 == 0:
            cprint("[TEST]")
            # Procedure의 Test함수로 모델 테스트 수행
            # world.config['multicore'] = 멀티코어 사용할지 여부
            Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
        # BPR손실로 모델 학습하는 과정 (neg_k = 부정샘플링 개수)
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        # 현재 Epoch의 모델 가중치를 weight_file에 저장
        # 학습 중단되어도 나중에 다시 재개할 수 있도록
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    # 훈련 종료 시, tensorboard를 닫음
    if world.tensorboard:
        w.close()