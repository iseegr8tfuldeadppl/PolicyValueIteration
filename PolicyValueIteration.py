import gym
import time
import numpy
from Resources import *


def main():
    gamma = 1.0
    game = 'FrozenLake-v0'
    improvementIterations = 10000
    evaluationIterations = 100
    eps = 1e-20
    
    #For Ubunto game = 'Deterministic-4x4FrozenLake-v0'
    env = gym.make(game).env

    PolicyIterator(env=env, gamma=gamma, evaluationIterations=evaluationIterations, improvementIterations=improvementIterations, eps=eps)
    ValueIterator(env=env, gamma=gamma, evaluationIterations=evaluationIterations, improvementIterations=improvementIterations, eps=eps)

if __name__ == '__main__':
    main()
