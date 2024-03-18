from xml.dom import minidom
from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
from configparser import ConfigParser
import ast

cfg = ConfigParser()
cfg.read("config.ini",encoding="utf-8")

txt_no_cat = cfg.get("TEXT", "txt_no_cat")
txt_1_cat = cfg.get("TEXT", "txt_1_cat")
txt_more_than_one = cfg.get("TEXT", "txt_more_than_one")
labels_requires = ast.literal_eval(cfg.get("DATASET", "labels_necessary"))


def printText(bg, txt, color=(0,255,0,0), size=0.7, pos=(0,0), type="Chinese"):
    (b,g,r,a) = color

    if(type=="English"):
        cv2.putText(bg,  txt, pos, cv2.FONT_HERSHEY_SIMPLEX, size,  (b,g,r), 2, cv2.LINE_AA)

    else:
        ## Use simsum.ttf to write Chinese.
        fontpath = "fonts/wt009.ttf"
        font = ImageFont.truetype(fontpath, int(size*10*2))
        img_pil = Image.fromarray(bg)
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos,  txt, font = font, fill = (b, g, r, a))
        bg = np.array(img_pil)
    return bg
    
def bb_intersection_over_union(A, B):
    boxA = [A[0], A[1], A[0]+A[2], A[1]+A[3]]
    boxB = [B[0], B[1], B[0]+B[2], B[1]+B[3]]
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def output_txt(labels, bboxes):
    cat_dic = {}
    body_txt = ""
    total = 0
    for i, l in enumerate(labels):
        if l in labels_requires:
            if l not in cat_dic:
                count = 0
            else:
                count = cat_dic[l]                

            cat_dic.update( {l:count+1} )
            total += 1

    if len(cat_dic) == 0:
        body_txt = txt_no_cat

    else:
        if len(cat_dic) == 1:
            if list(cat_dic.keys())[0]==1:
                body_txt = txt_1_cat.format(list(cat_dic.keys())[0])
            else:
                body_txt = txt_more_than_one.format(list(cat_dic.keys())[0])
        else:
            body_txt = txt_more_than_one.format(total)
            for cat in cat_dic:
                if cat_dic[cat] > 1:
                    body_txt += ", {} are {}".format(cat_dic[cat],cat)
                else:    
                    body_txt += ", {} is {}".format(cat_dic[cat],cat)

    return body_txt

def getLabels(imgFile, xmlFile):
    labelXML = minidom.parse(xmlFile)
    labelName = []
    labelXmin = []
    labelYmin = []
    labelXmax = []
    labelYmax = []
    totalW = 0
    totalH = 0
    countLabels = 0

    tmpArrays = labelXML.getElementsByTagName("name")
    for elem in tmpArrays:
        labelName.append(str(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("xmin")
    for elem in tmpArrays:
        labelXmin.append(int(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("ymin")
    for elem in tmpArrays:
        labelYmin.append(int(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("xmax")
    for elem in tmpArrays:
        labelXmax.append(int(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("ymax")
    for elem in tmpArrays:
        labelYmax.append(int(elem.firstChild.data))

    return labelName, labelXmin, labelYmin, labelXmax, labelYmax

def write_lale_images(label, img, saveto, filename):
    writePath = os.path.join(output_db_path,"images")
    if not os.path.exists(writePath):
        os.makedirs(writePath)

    cv2.imwrite(os.path.join(writePath, filename), img)