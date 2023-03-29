import time
from collections import deque
import random

import numpy as np
import torch

from net2 import *
from game import Game
from data_loader import DataLoader


class Train(object):
    def __init__(self, use_cuda=USECUDA, lr=LR):

        if use_cuda:
            torch.cuda.manual_seed(1234)
        else:
            torch.manual_seed(1234)

        self.kl_targ = 0.02
        self.lr_multiplier = 1.
        self.use_cuda = use_cuda

        self.net = Net()
        self.eval_net = Net()
        if use_cuda:
            self.net = self.net.cuda()
            self.eval_net = self.eval_net.cuda()

        self.dl = DataLoader(use_cuda, MINIBATCH)
        # 队列
        self.sample_data = deque(maxlen=TRAINLEN)
        self.gen_optim(lr)
        self.entropy = AlphaEntropy()
        self.writer = SummaryWriter('./log')

    def sample(self, datas):
        for state, pi, reward in datas:
            c_sta
            te = state.copy()
            c_pi = pi.copy()
            # 五子棋是对称游戏
            for i in range(4):
                # 旋转
                c_state = np.array([np.rot90(s, i) for s in c_state])
                c_pi = np.rot90(c_pi.reshape(SIZE, SIZE), i)
                self.sample_data.append([c_state, c_pi.flatten(), reward])
                # 左右反转
                c_state = np.array([np.fliplr(s) for s in c_state])
                c_pi = np.fliplr(c_pi)
                self.sample_data.append([c_state, c_pi.flatten(), reward])

        return len(datas)

    def gen_optim(self, lr):
        optim = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=L2)
        self.optim = ScheduledOptim(optim, lr)

    def run(self):
        offset = 0
        model_path = f"modelNet2.pt"
        # model_path = f"model_{time.strftime('%Y%m%d%H%M', time.localtime())}.pt"
        self.net.save_model(path=model_path)
        self.eval_net.load_model(path=model_path, cuda=self.use_cuda)

        for step in range(1 + offset, 1 + GAMETIMES):
            game = Game(self.net, self.eval_net)
            print(f"Game - {step} | data length - {self.sample(game.play())}")
            if len(self.sample_data) < MINIBATCH:
                continue

            states, pi, rewards = self.dl(self.sample_data)
            _, old_props = self.net(states)

            for _ in range(EPOCHS):
                self.optim.zero_grad()

                # Cross-entropy
                v, props = self.net(states)
                loss = self.entropy(props, v, pi, rewards)
                loss.backward()

                self.optim.step()

                _, new_props = self.net(states)
                kl = torch.mean(torch.sum(
                    torch.exp(old_props) * (old_props - new_props), 1)).item()
                if kl > self.kl_targ * 4:
                    break
            # KL散度，这是一个用来衡量两个概率分布的相似性的一个度量指标。
            # 近似估计的概率分布和数据整体真实的概率分布的相似度，或者说差异程度，可以用 KL 散度来表示。
            if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
            elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5

            self.optim.update_learning_rate(self.lr_multiplier)

            print(
                f"kl - {kl} | lr_multiplier - {self.lr_multiplier} | loss - {loss}")
            print("-" * 100 + "\r\n")

            self.writer.add_scalar('loss', loss, step)
            self.writer.add_scalar('KL', kl, step)

            if step % CHECKOUT == 0:
                result = [0, 0, 0]  # draw win loss
                for _ in range(EVALNUMS):
                    print(f"evaluate - {_}")
                    game.reset()
                    game.evaluate(result)

                if result[1] + result[2] == 0:
                    rate = 0
                else:
                    rate = result[1] / (result[1] + result[2])
                self.writer.add_scalar('rate', rate, step // CHECKOUT)

                print(f"step - {step} evaluation")
                print(
                    f"win - {result[1]} | loss - {result[2]} | draw - {result[0]}")

                # save or reload model
                if rate >= WINRATE:
                    print(f"new best model. rate - {rate}")
                    self.net.save_model(path=model_path)
                    self.eval_net.load_model(
                        path=model_path, cuda=self.use_cuda)
                else:
                    print(f"load last model. rate - {rate}")
                    self.net.load_model(path=model_path, cuda=self.use_cuda)

                print("-" * 100 + "\r\n")


if __name__ == "__main__":
    t = Train()
    t.run()
