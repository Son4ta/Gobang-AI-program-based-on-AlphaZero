import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from const import *


class PolicyValue(nn.Module):
    """policy-value network module"""

    def __init__(self, board_width=SIZE, board_height=SIZE):
        super(PolicyValue, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height,
                                 board_width * board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act))
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


# 上述三个网络
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.feat = PolicyValue()

    def forward(self, x):
        props, winners = self.feat.forward(x)
        return winners, props

    def save_model(self, path="model.pt"):
        torch.save(self.state_dict(), path)

    def load_model(self, path="model.pt", cuda=True):
        if cuda:
            self.load_state_dict(torch.load(path))
            self.cuda()
        else:
            self.load_state_dict(torch.load(
                path, map_location=lambda storage, loc: storage))
            self.cpu()


# 损失计算
class AlphaEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.v_loss = nn.MSELoss()

    def forward(self, props, v, pi, reward):
        v_loss = self.v_loss(v, reward)
        p_loss = -torch.mean(torch.sum(props * pi, 1))

        return p_loss + v_loss


# 优化器
class ScheduledOptim(object):
    def __init__(self, optimizer, lr):
        self.lr = lr
        self.optimizer = optimizer

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_learning_rate(self, lr_multiplier):
        new_lr = self.lr * lr_multiplier
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


if __name__ == "__main__":
    net = Net()
    # dummy_input = torch.rand(8, 1, 8, 8)  # 假设输入8张1*8*8的图片
    writer = SummaryWriter("./alphaZeroNet2")
    writer.add_graph(net, torch.rand(1, 8, 8, 8))
    writer.close()
    # with SummaryWriter(comment='LeNet') as w:
    #     w.add_graph(net, (dummy_input,))
