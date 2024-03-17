#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import imutils
import os, time
import os.path
import numpy as np
import pandas as pd
from libFuncs import *
from tqdm import tqdm
from configparser import ConfigParser
import ast

#-------------------------------------------
cfg = ConfigParser()
cfg.read("config.ini",encoding="utf-8")

output_db_path = cfg.get("OUTPUT", "output_db_path")
dataset_images = cfg.get("DATASET", "dataset_images")
dataset_labels = cfg.get("DATASET", "dataset_labels")
dataset_negatives = cfg.get("DATASET", "dataset_negatives")

label_img_maps = cfg.get("OUTPUT", "label_img_maps")
#-------------------------------------------
output_db_path = output_db_path.replace('\\', '/')
dataset_images = dataset_images.replace('\\', '/')
dataset_labels = dataset_labels.replace('\\', '/')
dataset_negatives = dataset_negatives.replace('\\', '/')

def chkEnv():
    if not os.path.exists( os.path.join(output_db_path, "images")):
        os.makedirs(os.path.join(output_db_path, "images"))
        print("no {} folder, created.".format(os.path.join(output_db_path, "images")))

    if(not os.path.exists(dataset_images)):
        print("There is no such folder {}".format(dataset_images))
        quit()

    if(not os.path.exists(dataset_labels)):
        print("There is no such folder {}".format(dataset_labels))
        quit()

#--------------------------------------------

chkEnv()

print("I) positive images")
i = 0
f = open(os.path.join(output_db_path, label_img_maps), "w", encoding='utf-8')
for file in tqdm(os.listdir(dataset_images)):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):

        if not os.path.exists(os.path.join(dataset_labels, filename+".xml")):
            print("Cannot find the file {} for the image.".format(os.path.join(dataset_labels, filename+".xml")))

        else:
            image_path = os.path.join(dataset_images, file)
            xml_path = os.path.join(dataset_labels, filename+".xml")
            labelName, labelXmin, labelYmin, labelXmax, labelYmax = getLabels(image_path, xml_path)

            image = cv2.imread(image_path)
            try:
                test = image.shape

            except:
                continue
                
            bboxes = []
            for id, label in enumerate(labelName):
                x1, x2, y1, y2 = labelXmin[id], labelXmax[id], labelYmin[id], labelYmax[id]
                bboxes.append( [x1, y1, x2-x1, y2-y1] )
                
            lbl_txt = output_txt(labelName, bboxes)

            img_output_path = os.path.join(output_db_path, "images", filename+".jpg")
            cv2.imwrite(img_output_path, image)
            txt_label = "{}||{}\n".format(filename+".jpg", lbl_txt)
            f.write(txt_label)

print("II) negative images")
txt_no_cat = cfg.get("TEXT", "txt_no_cat")
for file in tqdm(os.listdir(dataset_negatives)):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
        image_path = os.path.join(dataset_negatives, file)
        image = cv2.imread(image_path)
        try:
            test = image.shape
        except:
            continue
            
        img_output_path = os.path.join(output_db_path, "images", filename+".jpg")
        cv2.imwrite(img_output_path, image)
        lbl_txt = txt_no_cat
        txt_label = "{}||{}\n".format(filename+".jpg", lbl_txt)
        f.write(txt_label)

f.close()

print("III) make parquet and pickle from")
df = pd.DataFrame()
path_images = os.path.join(output_db_path, "images")
with open(os.path.join(output_db_path, label_img_maps), 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        line = line.strip()
        if len(line) == 0:
            continue
        line = line.split('||')
        img_txt = line[1]
        img_name = line[0]
        img_path = os.path.join(path_images, img_name)

        new_row = pd.DataFrame({
                'image': [img_path],
                'prompt': [img_txt]
            })

        if len(df) > 0:
            df = pd.concat([new_row,df.loc[:]]).reset_index(drop=True)
        else:
            df = pd.DataFrame({
                'image': [img_path],
                'prompt': [img_txt]
            })

parquet_file = cfg.get("OUTPUT", "parquet_file")
pickle_file = cfg.get("OUTPUT", "pickle_file")
df.to_parquet(os.path.join(output_db_path, parquet_file).replace('\\', '/'))
df.to_pickle(os.path.join(output_db_path, pickle_file).replace('\\', '/'))
