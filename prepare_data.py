import numpy as np


def get_data(path):
    file = open(path)
    data_str= file.read()
    data_split = data_str.split('\n')
    data = [None] * (len(data_split) // 2)
    index = 0
    for x in enumerate(data_split):
        if x[0] % 2 == 1:
            data[index] = x[1]
            index += 1
    data_char_1d = np.array([x for y in data for x in y])
    data_char_2d = np.array([[x for x in y] for y in data])
    return  data_char_1d, data_char_2d

if __name__ == '__main__':
    path = './随机数据生成器0.3/data.txt'
    data_1d, data_2d = get_data(path)
    print (data_1d)
    print (data_2d)