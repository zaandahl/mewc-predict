import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import string
import pandas as pd
import tensorflow as tf
import warnings
from datetime import datetime
from pathlib import Path
from tqdm import tqdm 
from tensorflow.keras.models import load_model
from lib_common import read_yaml
import tensorflow_addons as tfa

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

config = read_yaml("config.yaml")
for conf_key in config.keys():
    if conf_key in os.environ:
        config[conf_key] = os.environ[conf_key]

try:
    class_map = read_yaml("class_list.yaml")
except Exception as e:
    print(e)
    exit("ERROR: you must bind mount your class_list.yaml to /code/class_list.yaml")

inv_class = {v: k for k, v in class_map.items()}

try:
    gpus = tf.config.list_logical_devices('GPU')
    print(gpus)
    #strategy = tf.distribute.MirroredStrategy(devices=gpus, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    #with strategy.scope():
    en_model = load_model("/code/model.h5",
        custom_objects={'loss': tfa.losses.SigmoidFocalCrossEntropy()})
    en_model.summary()
except Exception as e:
    print(e) 
    exit("ERROR: you must bind mount your EfficientNet model to /code/model.h5")

img_generator = tf.keras.preprocessing.image_dataset_from_directory(
    config["INPUT_DIR"] + '/' + config["SNIP_DIR"], 
    labels=None,
    label_mode=None,
    batch_size=int(config["BATCH_SIZE"]), 
    image_size=(int(config["TARGET_SIZE"]), int(config["TARGET_SIZE"])),
    shuffle=False
)

try:
    en_path = Path(config['INPUT_DIR'],config['EN_FILE'])
    en_out = pd.read_pickle(en_path)
    timestamp = '{:.%Y%m%d-%H%M%S}'.format(datetime.now())
    en_out.to_pickle(Path(config['INPUT_DIR'],config['EN_FILE'] + timestamp))
    en_out.to_csv(Path(config['INPUT_DIR'],config['EN_CSV'] + timestamp))
except Exception as e:
    print(e)
    print("No existing en_out file found. Creating new one...")
    en_out = pd.DataFrame()

file_paths = img_generator.file_paths
filenames = list(map(lambda x : Path(x).name, file_paths)) # Path to extract just the filename

try:
    filename_map = dict(zip(en_out['filename'], en_out['rand_name']))
except:
    filename_map = None

labels = list(map(lambda x : Path(x).parent.name, file_paths)) # Get the parent directory as the label
preds = en_model.predict(img_generator) # get proba predictions

class_ids = sorted(inv_class.values())
class_names = [class_map.get(i,i)  for i in class_ids]
pred_df = pd.DataFrame(preds, columns=class_ids)

file_series = pd.Series(filenames)
label_series = pd.Series(labels)
pred_df.insert(0, "filename", file_series, True)
pred_df.insert(1, "label", label_series, True)
pred_df = pd.melt(pred_df, id_vars=['filename', 'label'], value_vars=class_ids, var_name="class_id", value_name="prob")
# Add the class_name back 
pred_df["class_name"] = pred_df["class_id"].replace(class_map)
# Rank the class probabilities grouped by filename
pred_df["class_rank"] = pred_df.groupby("filename")["prob"].rank("average", ascending=False)

if filename_map is not None:
    inv_filename = {v: k for k, v in filename_map.items()}
    pred_df["rand_name"] = None
    pred_df = pred_df.replace({"filename": inv_filename})
    # fill out rand_name by mapping from filename
    pred_df["rand_name"] = pred_df["filename"].replace(filename_map)

if(config["RENAME_SNIPS"] == "True"):
    print("Renaming snip files using " + str(config["SNIP_CHARS"]) + " alphanumeric characters...")
    pred_df["rand_name"] = ''
    for path in tqdm(Path(config["INPUT_DIR"],config["SNIP_DIR"]).iterdir()):
        if path.is_file():
            file_ext = path.suffix
            directory = path.parent
            new_name = ''.join(random.choices(string.ascii_letters + string.digits, k=int(config["SNIP_CHARS"]))) + file_ext
            pred_df.loc[pred_df['filename'] == path.name, 'rand_name'] = new_name
            path.rename(Path(directory,new_name))

if(config["TOP_CLASSES"] == "True"):
    pred_df = pred_df[pred_df["class_rank"] == 1.0]

pred_df.to_pickle(Path(config["INPUT_DIR"],config["EN_FILE"]))
pred_df.to_csv(Path(config["INPUT_DIR"],config["EN_CSV"]))
