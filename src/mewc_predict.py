import os, random, string
import absl.logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.set_verbosity(absl.logging.ERROR)
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
os.environ["KERAS_BACKEND"] = "jax"
import tensorflow as tf

from datetime import datetime
from keras import saving
from lib_common import read_yaml, model_img_size_mapping, update_config_from_env, setup_strategy
from pathlib import Path
from tqdm import tqdm 

config = update_config_from_env(read_yaml("config.yaml"))

try:
    class_map = read_yaml('class_map.yaml')
except Exception as e:
    print(e)
    exit("ERROR: you must bind mount your class-map file to /code/class_map.yaml")

inv_class = {v: k for k, v in class_map.items()}
img_size = model_img_size_mapping(config['MODEL']) # Get the image size for the model

try:
    strategy = setup_strategy() # Set up the strategy for distributed inference
    with strategy.scope():
        print("Loading model...")
        #model = saving.load_model("case_study_ENS_best.keras") # for local use
        model = saving.load_model("/code/model.keras") # use this version for Docker
    model.summary()
except Exception as e:
    print(e) 
    exit("ERROR: you must bind mount your EfficientNet model to /code/model.h5")

img_generator = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(config["INPUT_DIR"], config["SNIP_DIR"]), 
    labels=None,
    label_mode=None,
    batch_size=int(config["BATCH_SIZE"]), 
    image_size=(img_size, img_size),
    shuffle=False
)

try:
    path = Path(config['INPUT_DIR'],config['PRED_FILE'])
    model_out = pd.read_pickle(path)
    timestamp = '{:.%Y%m%d-%H%M%S}'.format(datetime.now())
    model_out.to_pickle(Path(config['INPUT_DIR'],config['PRED_FILE'] + timestamp))
    model_out.to_csv(Path(config['INPUT_DIR'],config['PRED_CSV'] + timestamp))
except Exception as e:
    print(e)
    print("No existing model-prediction file found. Creating new one...")
    model_out = pd.DataFrame()

file_paths = img_generator.file_paths
filenames = list(map(lambda x : Path(x).name, file_paths))

try:
    filename_map = dict(zip(model_out['filename'], model_out['rand_name']))
except:
    filename_map = None

labels = list(map(lambda x : Path(x).parent.name, file_paths))
preds = model.predict(img_generator)

class_ids = sorted(inv_class.values())
class_names = [class_map.get(i,i)  for i in class_ids]
pred_df = pd.DataFrame(preds, columns=class_ids)

file_series = pd.Series(filenames)
label_series = pd.Series(labels)
pred_df.insert(0, "filename", file_series, True)
pred_df.insert(1, "label", label_series, True)
pred_df = pd.melt(pred_df, id_vars=['filename', 'label'], value_vars=class_ids, var_name="class_id", value_name="prob")
pred_df["class_name"] = pred_df["class_id"].replace(class_map)
pred_df["class_rank"] = pred_df.groupby("filename")["prob"].rank("average", ascending=False)

if filename_map is not None:
    inv_filename = {v: k for k, v in filename_map.items()}
    pred_df["rand_name"] = None
    pred_df = pred_df.replace({"filename": inv_filename})
    pred_df["rand_name"] = pred_df["filename"].replace(filename_map)

if(config["RENAME_SNIPS"] == True):
    print("Renaming snip files using " + str(config["SNIP_CHARS"]) + " alphanumeric characters...")
    pred_df["rand_name"] = ''
    for path in tqdm(Path(config["INPUT_DIR"],config["SNIP_DIR"]).iterdir()):
        if path.is_file():
            file_ext = path.suffix
            directory = path.parent
            new_name = ''.join(random.choices(string.ascii_letters + string.digits, k=int(config["SNIP_CHARS"]))) + file_ext
            pred_df.loc[pred_df['filename'] == path.name, 'rand_name'] = new_name
            path.rename(Path(directory,new_name))

if(config["TOP_CLASSES"] == True):
    pred_df = pred_df[pred_df["class_rank"] == 1.0]

pred_df.to_pickle(Path(config["INPUT_DIR"],config["PRED_FILE"]))
pred_df.to_csv(Path(config["INPUT_DIR"],config["PRED_CSV"]))
