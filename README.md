# FineTuneViT
Finetune your ViT model, so it can look at pictures and speak, and tell the breed of cat in the picture.

The dataset used in the Github rep. comes from the "Cat Breeds and Body Markings" tagged by the Eden Foundation. There are a total of 871 images marked with 16 categories, 12 of which are cat breed categories and 4 are body parts.
- Body parts: eye, head, body, tail
- Cat breeds: abyssinian, bengal, birman, bombay, british_shorthair, egyptian_mau, maine_coon, persian, ragdoll, russian_blue, siamese, sphynx

you can purchase the "Cat Breeds and Body Markings" here: https://yolo.dog/product/model-cat-body-parts
and select the Dataset in model_types.

## How to use FineTuneViT
**Step 1: Install python packages**
pip install opencv-python
pip install imutils
pip install pandas
pip install tqdm
pip install pyarrow fastparquet
pip install torch torchvision torchaudio
pip install transformers

**Step 2: Edit config.ini**
1. HuggingFaceModel: The default is ydshieh/vit-gpt2-coco-en. You can also find other ViT models in huggingface.
2. dataset_images: (required) images path of the "Cat Breeds and Body Marks" dataset
3. dataset_labels: (required) labels path of the "Cat Breeds and Body Marks" dataset
4. dataset_negatives: (required) negatives path of the "Cat Breeds and Body Marks" dataset
5. labels_necessary: ​​Define the label name to be used. Here we only use the classes of 12 cat breeds.
6. model_finetuned: (required) The directory name of the Finetune model after training is completed (if there is no such directory, it will be created automatically)
7. output_db_path: (required) Directory path of the converted files and model output during training (if this directory does not exist, it will be automatically created)
8. label_img_maps: file name corresponding to the description text of the image
9. pickle_file: The name of the pickle file corresponding to the description text of the image
10. parquet_file: parquet DB name corresponding to the description text of the picture
11. continue_train_from_last: Whether to continue training from the directory where the last training was stored
12. epochs: how many epochs to train
13. batch_size: training batch_size
14. learning_rate: starting learning_rate of training
15. txt_no_cat: When there is no cat in the picture, the description text of the picture
16. txt_1_cat: If there is only one cat in the picture, the description text of the picture (note that {} will be replaced with the cat’s breed name)
17. txt_more_than_one: If there is more than one cat in the picture, the description text of the picture (note that {} will be replaced with the number of cats, and the program will automatically add the cat breed name and number at the end of the text)

**Step 3: Convert VOC dataset to image/text format**
If you already have a ready-made image/text dataset that can be used in finetine, you can omit this step.

1. pip install requirements.txt
2. modify config.ini
3. python make_db.py
4. python finetune.py

**Step 4: Start to finetune**
python finetune.py
