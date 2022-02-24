import numpy as np
import prepare_data as pd
import random
import time


# 矩阵抽掉一行
def mat_extract_row(mat, row):
    return np.vstack((mat[:row, :], mat[row + 1:, :]))


# 矩阵抽掉一列
def mat_extract_col(mat, col):
    return np.hstack((mat[:, :col], mat[:, col + 1:]))


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


def frag_psfm(mat_psfm, L):
    list = []
    dict = ['A', 'C', 'G', 'T']
    mat = mat_psfm.transpose()
    for line in mat:
        line = [tup for tup in enumerate(line.tolist())]
        line.sort(key=lambda tup: tup[1], reverse=True)
        index = line[0][0]
        list.append(dict[index])
    return list


# 得到背景概率
def p_base(data_1d):
    l = len(data_1d)
    baseP = np.array([0.0] * 4)
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
    baseP /= l
    return baseP


# 随机顺序集
class randomSeq():
    # 初始化 得到随机顺序集
    def __init__(self, t):
        self.set = [x for x in range(t)]
        random.shuffle(self.set)
        self.set = list(zip(list(range(t - 1, -1, -1)), self.set))

    # 弹出序号
    def generate(self):
        while (self.set):
            yield self.set.pop()


# 随机起点序列 randomStartSeq
class randomSS():
    # 初始化 得到随机起点集
    def __init__(self, s, L):
        self.set = [x for x in range(s - L + 1)]
        random.shuffle(self.set)

    # 弹出起点
    def generate(self):
        while (self.set):
            yield self.set.pop()


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


# 得到scoreF
def scoreF(baseP, mat_psfm):
    score = 0
    for (index, line) in enumerate(mat_psfm):
        basep = baseP[index]
        for p in line:
            score += p * np.log(p / basep + 1e-5)
    return score


# 移除偏置
def remove_bias(mat_psfm):
    func1 = lambda x: (1 if x >= 0.5 else 0)
    func2 = lambda x: sum([func1(i) for i in x])
    mat_psfm = mat_psfm.transpose()
    bias = 0
    if func2(mat_psfm[0]) == 0:
        for row in mat_psfm:
            if func2(row) == 0:
                bias += 1
            else:
                break
        return bias
    else:
        bias = 0
        for row in mat_psfm:
            if func2(row) >= 1:
                bias += 1
    return -(mat_psfm.shape[0] - bias)


if __name__ == '__main__':
    path = 'C:\\Users\\DELL\\Desktop\\DIR\\随机数据生成器0.3\\data.txt'
    L = 15
    data_1d, data_2d = pd.get_data(path)
    t, s = data_2d.shape
    baseP = p_base(data_1d)
    iteration_in = 10
    iteration_out = 5
    print('iteration:', iteration_out)

    # 初始起点集
    list_randomss = np.array([random.randint(0, s - L) for j in range(t)])
    mat_psfm = np.array
    print('the first start set:\t', list(list_randomss))
    # 迭代
    time_start = time.time()
    for i in range(iteration_out):
        randomseq = randomSeq(t)
        time_iter = time.time()
        time_seq = time.time()
        for seq in randomseq.generate():
            mat_frag = get_frag_mat(data_2d, list_randomss, L)

            # 随机选一个序列 计算剩余片段序列的psfm和scoreF
            num_out = seq[1]
            mat_frag = mat_extract_row(mat_frag, num_out)
            mat_psfm = psfm(mat_frag)
            score_f_1 = scoreF(baseP, mat_psfm)

            # 计算scoreE和scoreB 得到list_score_e和list_score_b
            list_score_e, list_score_b = [], []
            randomss = randomSS(s, L)
            for index in randomss.generate():
                list_score_e.append(scoreE(data_2d[num_out][index: index + L - 1], mat_psfm))
                list_score_b.append(scoreB(data_2d[num_out][index: index + L - 1], baseP))

            # 计算rate = scoreE/socreB 得到list_rate 并对list_rate进行排序
            list_rate = [x / y for x, y in zip(list_score_e, list_score_b)]
            list_rate = [tup for tup in enumerate(list_rate)]
            list_rate.sort(key=lambda tup: tup[1], reverse=True)

            # 遍历rate表
            mat_frag_temp = mat_frag
            for tup in list_rate:
                num_in = tup[0]
                mat_row = np.array([[char for char in data_2d[num_out][num_in:num_in + L]]])
                mat_frag_temp = np.vstack((mat_frag, mat_row))
                mat_psfm_temp = psfm(mat_frag_temp)
                score_f_2 = scoreF(baseP, mat_psfm_temp)
                if score_f_2 > score_f_1:
                    list_randomss[num_out] = num_in
                    mat_frag = mat_frag_temp
                    mat_psfm = mat_psfm_temp
                    break

            # 50个seq输出一次结果
            if (seq[0] + 1) % 50 == 0:
                print('iter%d' % i, '\t', '%-4s\t%-4s' % seq, '\t', frag_psfm(mat_psfm, L), end='\t')
                print('50 seqs take time: %fs' % (time.time() - time_seq))
                time_seq = time.time()
        print('the iter takes time: %fs' % (time.time() - time_iter))

    # 消除偏置影响
    bias = remove_bias(mat_psfm)
    print('bias:\t', bias)
    list_randomss += bias
    mat_frag = get_frag_mat(data_2d, list_randomss, L)
    mat_psfm = psfm(mat_frag)
    frag = frag_psfm(mat_psfm, L)
    print('the final start set:\t', [tup for tup in enumerate(list(list_randomss))])
    print('the final fragment:\t', frag)
    print('Running time:\t%fs' % (time.time() - time_start))
