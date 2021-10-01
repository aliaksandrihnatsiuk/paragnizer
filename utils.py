import numpy as np

def simple_split(img, interval, coordinate):

        img_slice = img[interval[0]: interval[-1]]
        hist_new = img_slice.sum(axis=0)
        # Список с координатами ненулевых элементов

        no_zero_coord = np.where(hist_new != 0)[0]
        print('no_zero_coord', no_zero_coord)
        # Новые Интревалы по direction

        new_interval = np.split(no_zero_coord, np.where(np.diff(no_zero_coord) != 1)[0] + 1)
        len_interval = len(new_interval)
        # Добавление координат в лист
        coordinate.append([no_zero_coord[0], interval[0], no_zero_coord[-1], interval[-1]])

        return coordinate


def find_interval(img):
    # Находим гистограммы - проеции матрицы img на каждую ось
    h_hist = img.sum(axis=1)  # projection on Y
    v_hist = img.sum(axis=0)  # projection on Х

    # Находим индексы ненулевых элементов в гистограммах
    x = np.where(h_hist != 0)[0]  # индексы ненулевый элементов по Y
    y = np.where(v_hist != 0)[0]  # индексы ненулевый элементов по Х

    # Записываем ненулевые интервалы через np.split
    interval_X = np.split(y, np.where(np.diff(y) != 1)[0] + 1)  # Интервалы по X
    interval_Y = np.split(x, np.where(np.diff(x) != 1)[0] + 1)  # Интервалы по Y

    # Находим длины интервалов
    interval_num_Y = len(interval_Y)
    interval_num_X = len(interval_X)

    return interval_X, interval_Y
