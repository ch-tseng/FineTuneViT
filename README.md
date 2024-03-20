# How to finetune a ViT (Vision Transformer) model

Train a ViT(Vision Transformer) that talks by looking at pictures, so that it can tell you the type of cat in the picture.

If you need "cat breed and body markings" dataset which marked by Eden Digital Shelter Factory, please contact them: https://www.facebook.com/eden0112/


Steps:

## A) Environment setting installation (it is recommended to use Virtualenv)
- pip install opencv-python
- pip install imutils
- pip install pandas
- pip install tqdm
- pip install pyarrow fastparquet
- pip install torch torchvision torchaudio
- pip install transformers

## B) Prepare the dataset for finetune
You could find some pictures of cats and tag each picture with a description, but this would take a lot of time and effort. Therefore, we directly used the "Cat Breed and Body Marking" provided by Eden Digital Shelter Factory. There are a total of 871 pictures marked with 16 categories, of which 12 are cat breed categories and 4 are body parts.
   - Body parts: eye, head, body, tail
   - Varieties: abyssinian, bengal, birman, bombay, british_shorthair, egyptian_mau, maine_coon, persian, ragdoll, russian_blue, siamese, sphynx
  
## C) Start finetune
   1. git clone https://github.com/ch-tseng/FineTuneViT.git
   2. cd FineTuneViT
   3. Just modify the following parameters
      dataset_images: The images path of the dataset "Cat Breeds and Body Markings"
      dataset_labels: The labels path of the "Cat Breeds and Body Marks" dataset
      dataset_negatives: negatives path of the "Cat Breeds and Body Marks" dataset
   4. Execute python make_db.py: The VOC Dataset will be converted into the corresponding format of Image/Text and stored in pickle format.
   5. Execute python finetune.py: start finetune
  
   The model after training will be placed in the path defined by this parameter model_finetuned in config.ini.
  
## D) Test ViT model
Prepare a few pictures and execute: python reference.py.
![Demo1](https://github.com/ch-tseng/FineTuneViT/blob/main/demos/predict1.jpg)

If an "OSError occurs during inference… does not appear to have a file named preprocessor_config.json…"
error, please manually copy the preprocessor_config.json under the original ViT model (for example, ydshieh/vit-gpt2-coco-en) to the finetuned model directory.
