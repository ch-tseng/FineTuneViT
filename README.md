# FineTuneViT
Finetune your ViT model

**steps:**
1. pip install requirements.txt
2. modify config.ini
3. python make_db.py
4. python finetune.py

**Parameter description of config.ini**
[MODEL]
HuggingFaceModel = {ViT model name}

[DATASET]
dataset_images = {the cat dataset path of images}
dataset_labels = {the cat dataset path of labels}
dataset_negatives = (path of negative images )
labels_necessary = ["abyssinian","bengal","birman","bombay","british_shorthair","egyptian_mau","maine_coon","persian","ragdoll","russian_blue","siamese","sphynx"]

[OUTPUT]
model_finetuned = {Finetuned model name}
output_db_path = {output base-path for finetuned model}
label_img_maps = output_labels.txt
pickle_file = ds.pickle
parquet_file = ds.parquet

[TRAIN]
continue_train_from_last = False
epochs = 20
batch_size = 36
learning_rate = 5e-5

[TEXT]
txt_no_cat = There is no cat in the picture
txt_1_cat = There is a cat in the picture, its breed is {}.
txt_more_than_one = There are {} cats in the picture, 
