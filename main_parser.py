import numpy as np
import prepare_data as pd
import random
import argparse


# 得到psfm概率矩阵
def psfm(mat_frag):
    t, L = mat_frag.shape[0], mat_frag.shape[1]
    mat_p = np.array([[0.0] * L] * 4)
    for i in range(L):
        for j in range(t):
            char = mat_frag[j][i]
            if char == 'A':
                mat_p[0][i] += 1
            elif char == 'C':
                mat_p[1][i] += 1
            elif char == 'G':
                mat_p[2][i] += 1
            elif char == 'T':
                mat_p[3][i] += 1
            else:
                raise Exception('无效字符')
    mat_p /= t
    return mat_p


# 得到背景概率
def p_base(data_1d):
    baseP = [0.0] * 4
    for char in data_1d:
        if char == 'A':
            baseP[0] += 1
        elif char == 'C':
            baseP[1] += 1
        elif char == 'G':
            baseP[2] += 1
        elif char == 'T':
            baseP[3] += 1
        else:
             raise Exception('无效字符')
    baseP /= len
    return baseP


# 随机顺序集
class randomSeq():
    # 初始化 得到随机顺序集
    def __init__(self, t):
        self.set = [x for x in range(t)]
        random.shuffle(self.set)

    # 弹出序号
    def pop(self):
        return self.set.pop()

# 随机起点序列 randomStartSeq
class randomSS():
    # 初始化 得到随机起点集
    def __init__(self, s, L):
        self.set = [x for x in range(s - L + 1)]
        random.shuffle(self.set)

    # 弹出起点
    def pop(self):
        return self.set.pop()

# 得到片段矩阵
def get_frag_mat(mat, list, L):
    t = mat.shape[0]
    mat_frag = np.array([[None] * L] * t)
    for (index0, index1) in enumerate(list):
        for (index2, index3) in enumerate(range(index1, index1 + L)):
            mat_frag[index0][index2] = mat[index0][index3]
    return mat_frag


# 得到scoreE
def scoreE(frag, mat_psfm):
    score = 1.0
    for index, char in enumerate(frag):
        if char == 'A':
            score *= mat_psfm[0][index]
        elif char == 'C':
            score *= mat_psfm[1][index]
        elif char == 'G':
            score *= mat_psfm[2][index]
        elif char == 'T':
            score *= mat_psfm[3][index]
        else:
            raise Exception('无效字符')
    return score


# 得到scoreB
def scoreB(frag, baseP):
    score = 1.0
    for char in frag:
        # tmp.append(char)
        if char == 'A':
            score *= baseP[0]
        elif char == 'C':
            score *= baseP[1]
        elif char == 'G':
            score *= baseP[2]
        elif char == 'T':
            score *= baseP[3]
        else:
            raise Exception('无效字符')
        # print(tmp)
    return score

def train(args):
    path = args.path
    L = args.fraglenth
    data_1d, data_2d = pd.get_data(path)
    t, s = data_2d.shape
    baseP = p_base(data_1d)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest = 'mode')
    parser_train =sub_parsers.add_parser('train')
    parser_train.add_argument('--path', type=str, required=True)
    parser_train.add_argument('--fraglength',type=int, required=True)

    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    else: raise Exception('Error')
