from copy import deepcopy

import numpy as np

from const import *
from data_loader import to_tensor, to_numpy


class TreeNode(object):
    def __init__(self,
                 action=None,
                 props=None,
                 parent=None):

        self.parent = parent
        self.action = action
        self.children = []

        self.N = 0  # visit count
        self.Q = .0  # mean action value
        self.W = .0  # total action value
        self.P = props  # prior probability

    def is_leaf(self):
        return len(self.children) == 0

    # 选扩展结点
    def select_child(self):
        index = np.argmax(np.asarray([c.uct() for c in self.children]))
        return self.children[index]

    # MCTS算法的一个非常流行的变体是树的上置信限(Upper Confidence Bounds for Trees，简称为UCT )
    def uct(self):
        return self.Q + self.P * CPUCT * (np.sqrt(self.parent.N) / (1 + self.N))

    # 扩展结点
    def expand_node(self, props):
        self.children = [TreeNode(action=action, props=p, parent=self)
                         for action, p in enumerate(props) if p > 0.]

    def backup(self, v):
        self.N += 1
        self.W += v
        self.Q = self.W / self.N


class MonteCarloTreeSearch(object):
    def __init__(self, net,
                 ms_num=MCTSSIMNUM):

        self.net = net
        # ms_num决定层数
        self.ms_num = ms_num

    def search(self, borad, node, temperature=.001):
        self.borad = borad
        self.root = node

        for _ in range(self.ms_num):
            node = self.root
            borad = self.borad.clone()

            # 有孩子 选择孩子结点
            while not node.is_leaf():
                node = node.select_child()
                borad.move(node.action)
                borad.trigger()

            # be carefull - opponent state
            value, props = self.net(
                to_tensor(borad.gen_state(), unsqueeze=True))
            value = to_numpy(value, USECUDA)[0]
            props = np.exp(to_numpy(props, USECUDA))

            # add dirichlet noise for root node
            if node.parent is None:
                props = self.dirichlet_noise(props)

            # normalize正规化
            # 让不能下的地方概率为0，再规范化概率
            props[borad.invalid_moves] = 0.
            total_p = np.sum(props)
            if total_p > 0:
                props /= total_p

            # winner, draw or continue
            if borad.is_draw():
                # 是否下满了
                value = 0.
            else:
                done = borad.is_game_over(player=borad.last_player)
                # 游戏结束？
                if done:
                    value = -1.
                else:
                    # 扩展结点 生成孩子结点
                    node.expand_node(props)

            # 反向传播更新N W Q
            while node is not None:
                value = -value
                node.backup(value)
                node = node.parent

        action_times = np.zeros(borad.size**2)
        for child in self.root.children:
            action_times[child.action] = child.N

        action, pi = self.decision(action_times, temperature)
        for child in self.root.children:
            if child.action == action:
                return pi, child

    # 通过将Dirichlet噪声添加到根节点的先验概率来实现额外的探索
    # 这种噪音确保可以尝试所有动作，但是搜索可能仍会否决不良动作。
    @staticmethod
    def dirichlet_noise(props, eps=DLEPS, alpha=DLALPHA):
        return (1 - eps) * props + eps * np.random.dirichlet(np.full(len(props), alpha))

    # 决策函数
    @staticmethod
    def decision(pi, temperature):
        # 论文中Play阶段公式实现
        pi = (1.0 / temperature) * np.log(pi + 1e-10)
        pi = np.exp(pi - np.max(pi))
        pi /= np.sum(pi)
        # 按概率随机选一个 轮盘赌我测
        action = np.random.choice(len(pi), p=pi)
        return action, pi
