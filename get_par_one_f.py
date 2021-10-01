import pytesseract
from pytesseract import Output
import cv2
from pdf2image import convert_from_path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tesserocr import PyTessBaseAPI, RIL, PSM, iterate_level
import pandas as pd
import ctypes
import matplotlib
import matplotlib.pyplot as plt


from get_par import convert_from_path, get_par_from_bbox
H=2896
W=2048

def get_par(pdf_path, page_n, h=H, w=W, ):
    pil_imgs = convert_from_path(pdf_path, size=(w, h), first_page=page_n, last_page=page_n, grayscale=True)
    imgs = [np.array(i) for i in pil_imgs]

    rgb_im_pill = convert_from_path(pdf_path, size=(w, h), first_page=page_n, last_page=page_n, grayscale=True)
    rgb_im = [np.array(i) for i in rgb_im_pill][0]

    res = []
    words = []
    with PyTessBaseAPI(path='/home/user/PycharmProjects/tessdata', psm=PSM.AUTO) as api:
        api.SetImageBytes(
            imagedata=imgs[0].tobytes(),
            width=W,
            height=H,
            bytes_per_pixel=1,
            bytes_per_line=W)
        boxes = api.GetComponentImages(RIL.WORD, True)
        api.Recognize()
        level = getattr(RIL, 'WORD')
        iter = api.GetIterator()
        for r in iterate_level(iter, level):
            element = r.GetUTF8Text(level)
            word_attributes = {}
            if element and not element.isspace():
                res.append(r.BoundingBox(level))
                words.append(element)

    bboxes = np.array(res, dtype='int32')

    def draw_bbox(bbox, img):
        img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255

    def calc_mean_len(gaps, mat, result):
        w_hist = mat[gaps[0]:gaps[1], :].sum(axis=0)

        w_bin = (w_hist != 0) * 1

        w_bin_dif = -np.diff(w_bin)

        minus = (w_bin_dif < 0).nonzero()[0]
        ones = (w_bin_dif > 0).nonzero()[0]
        words_sizes = minus[1:] - ones[:-1]

        result.append(words_sizes)

    img = np.full((H, W), 0, dtype='uint8')

    np.apply_along_axis(draw_bbox, 1, bboxes, img)

    h_hist = img.sum(axis=1)
    h_bin = h_hist == 0
    h_bin = h_bin * 1
    h_bin_dif = np.diff(h_bin * 1)

    minus = (h_bin_dif < 0).nonzero()[0]
    ones = (h_bin_dif > 0).nonzero()[0]

    df = pd.DataFrame(data={'word': words,
                            'x1': bboxes[:, 0],
                            'y1': bboxes[:, 1],
                            'x2': bboxes[:, 2],
                            'y2': bboxes[:, 3]})

    return bboxes, imgs[0], df, rgb_im


if __name__ == '__main__':
    pdf_path = '/pdf/XTX Execution Services_Terms of Business_September 2020.pdf'

    bboxes, img_text, df_s, rgb_im = get_par(pdf_path, 1)

    def (bboxes):
        list_red_pixels = []
        for i in range(len(bboxes)):
            occurrences = np.count_nonzero(img_text[bboxes[i][1]:bboxes[i][3], bboxes[i][0]:bboxes[i][2]] == 83)
            list_red_pixels.append(occurrences)
        return list_red_pixels


    # Add column Square and count it
    df_s.insert(5, 'square', 0)
    df = df_s.assign(square = (df_s.x2 - df_s.x1) * (df_s.y2 - df_s.y1))
    # Add column Red_pixel and count their numbers
    list_red_pixels = number_red_pixels(bboxes)
    df.insert(6, 'red_pixels', list_red_pixels)
    # Calculate ratio Red_pixels to All pixels
    df = df_s.assign(ratio = df.red_pixels / df.square)
    # Show table Red_word (True or False)
    df['red_word'] = df['ratio'].apply(lambda x: 'False' if x <= 0.15 else 'True')
    print(df.head(10))


    #ax[0].imshow(img_text, 'gray')
    #cv2.imshow('rgb_im', rgb_im)
    #plt.show()
    #cv2.waitKey(0)