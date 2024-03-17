import torch
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2Tokenizer
import os
from torch.optim import AdamW
from PIL import Image
import pandas as pd
import numpy as np
from configparser import ConfigParser
import ast

#-------------------------------------------
cfg = ConfigParser()
cfg.read("config.ini",encoding="utf-8")

sourceModel = cfg.get("MODEL", "HuggingFaceModel")
#"Ayansk11/Image_Caption_using_ViT_GPT2"
#"ydshieh/vit-gpt2-coco-en"
#"nlpconnect/vit-gpt2-image-captioning"
finetune_ds = cfg.get("OUTPUT", "pickle_file")
output_db_path = cfg.get("OUTPUT", "output_db_path").replace('\\', '/')
output_model = cfg.get("OUTPUT", "model_finetuned")
keep_training_from_output = cfg.getboolean("TRAIN", "continue_train_from_last")
epochs = cfg.getint("TRAIN", "epochs")
batch_size = cfg.getint("TRAIN", "batch_size")
lr = float(cfg.get("TRAIN", "learning_rate"))

# 定義自己的Dataset
class ImageTextDataset(Dataset):
    def __init__(self, dataframe, feature_extractor, tokenizer):
        self.dataframe = dataframe
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        image = Image.open(item['image']).convert('RGB')
        text = item['prompt']
        
        # 使用feature_extractor處理圖像
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        
        
        # 使用tokenizer處理文本
        targets = self.tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors="pt")

        return {
            'pixel_values': inputs['pixel_values'].squeeze(), 
            'input_ids': targets['input_ids'].squeeze(),
            'attention_mask': targets['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

# 加載預訓練模型和分詞器
if keep_training_from_output is True:
    model = VisionEncoderDecoderModel.from_pretrained(output_model)
else:
    model = VisionEncoderDecoderModel.from_pretrained(sourceModel)
#feature_extractor = ViTFeatureExtractor.from_pretrained("ydshieh/vit-gpt2-coco-en")
feature_extractor = ViTImageProcessor.from_pretrained(sourceModel)

tokenizer = GPT2Tokenizer.from_pretrained(sourceModel)

# 創建數據集實例
df = pd.read_pickle(os.path.join(output_db_path,finetune_ds))
dataset = ImageTextDataset(df, feature_extractor, tokenizer)

# 創建數據加載器
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定義優化器
optimizer = AdamW(model.parameters(), lr=lr)


# 將模型移動到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

tokenizer = GPT2Tokenizer.from_pretrained(sourceModel)
tokenizer.save_pretrained(os.path.join(output_db_path, output_model))

# 微調模型
print("Start training... total {} image files.".format(len(df)))
print('-----------------------------------------------------')
model.train()
min_loss = 9999.9
for epoch in range(epochs):  
    total_count = len(dataloader)
    for ii, batch in enumerate(dataloader):
        # 將數據移動到GPU
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 歸零梯度
        optimizer.zero_grad()

        # 前向傳播
        outputs = model(pixel_values=pixel_values, labels=labels, decoder_attention_mask=attention_mask)

        # 計算損失
        loss = outputs.loss

        # 反向傳播
        loss.backward()

        # 更新參數
        optimizer.step()
        if epoch==total_count-2:
            endtxt = ''
        else:
            endtxt = '\r'

        np_loss = loss.detach().cpu().numpy()
        print("[{}/{}] {}/{}.................... loss:{}".format(epoch+1, epochs, ii+1, \
                                                total_count, np.round(np_loss,6)), end=endtxt)

    
    if np_loss < min_loss:
        print("[{}/{}] saved.".format(epoch+1, epochs))
        # 保存微調後的模型
        model.save_pretrained(os.path.join(output_db_path,output_model))
    else:
        print(" ")

    if min_loss > np_loss: min_loss = np_loss


