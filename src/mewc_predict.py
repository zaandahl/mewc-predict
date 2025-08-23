import os, random, string, shutil, time
import numpy as np
import absl.logging
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
absl.logging.set_verbosity(absl.logging.ERROR)
# Choose backend at process start; default to TF for inference
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import tensorflow as tf

from datetime import datetime
from keras import saving
from lib_common import read_yaml, model_img_size_mapping, update_config_from_env
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

def _log(msg, t0=[time.perf_counter()]):  # tiny timer helper
    print(f"[{time.perf_counter()-t0[0]:6.2f}s] {msg}")

# Optional control over XLA JIT compile (can add startup cost on small jobs)
try:
    xla_flag = str(config.get("XLA_JIT", "auto")).strip().lower()
    if xla_flag in ("true", "on", "1", "yes"):
        tf.config.optimizer.set_jit(True)
    elif xla_flag in ("false", "off", "0", "no"):
        tf.config.optimizer.set_jit(False)
    # else: "auto" => leave TF defaults
except Exception:
    pass

# Make GPU initialization cheaper by avoiding full upfront allocation
try:
    for _gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass

try:
    # Optional SavedModel support (preferred if present)
    use_savedmodel = bool(str(config.get("USE_SAVEDMODEL", "True")).lower() == "true")
    export_dir = config.get("MODEL_EXPORT_DIR", "/code/model_export")

    model_is_savedmodel = False

    if use_savedmodel and os.path.isdir(export_dir):
        staged_export = "/tmp/model_export"
        _log("Staging SavedModel to local container FS...")
        if os.path.exists(staged_export):
            shutil.rmtree(staged_export)
        shutil.copytree(export_dir, staged_export, dirs_exist_ok=True)
        _log("Loading SavedModel...")
        model = tf.saved_model.load(staged_export)
        model_is_savedmodel = True
    else:
        MODEL_PATH = str(config.get("MODEL_PATH", "/code/model.keras"))
        # Stage the model onto fast local storage inside the container to avoid slow bind-mount I/O
        staged_path = "/tmp/model.keras"
        _log("Staging Keras model to local container FS...")
        # copyfile avoids extra metadata work on Windows bind mounts
        shutil.copyfile(MODEL_PATH, staged_path)
        MODEL_PATH = staged_path
        _log("Loading Keras model...")
        safe_mode = bool(str(config.get("SAFE_MODE", "True")).lower() == "true")
        model = saving.load_model(MODEL_PATH, compile=False, safe_mode=safe_mode)
        if bool(str(config.get("PRINT_SUMMARY", "False")).lower() == "true"):
            model.summary()
except Exception as e:
    print(e)
    exit("ERROR: could not load a model. Either mount a SavedModel directory to 'MODEL_EXPORT_DIR' (default /code/model_export) or mount a Keras .keras file to 'MODEL_PATH' (default /code/model.keras).")

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(config["INPUT_DIR"], config["SNIP_DIR"]), 
    labels=None,
    label_mode=None,
    batch_size=int(config["BATCH_SIZE"]), 
    image_size=(img_size, img_size),
    shuffle=False
)

# Preserve file paths before adding prefetch; prefetch returns a new Dataset
file_paths = dataset.file_paths
img_generator = dataset.prefetch(tf.data.AUTOTUNE)

try:
    path = Path(config['INPUT_DIR'],config['PRED_FILE'])
    model_out = pd.read_pickle(path)
    # Correct timestamp formatting for backup filenames
    timestamp = '{:%Y%m%d-%H%M%S}'.format(datetime.now())
    model_out.to_pickle(Path(config['INPUT_DIR'],config['PRED_FILE'] + timestamp))
    model_out.to_csv(Path(config['INPUT_DIR'],config['PRED_CSV'] + timestamp))
except Exception as e:
    print(e)
    print("No existing model-prediction file found. Creating new one...")
    model_out = pd.DataFrame()

filenames = list(map(lambda x : Path(x).name, file_paths))

try:
    filename_map = dict(zip(model_out['filename'], model_out['rand_name']))
except:
    filename_map = None

labels = list(map(lambda x : Path(x).parent.name, file_paths))
def _predict_with_savedmodel(sm, ds):
    # Use serving_default if present, else first signature
    infer = sm.signatures.get("serving_default")
    if infer is None:
        infer = next(iter(sm.signatures.values()))
    preds = []
    # Try to determine if signature expects named or positional input
    try:
        args, kwargs = infer.structured_input_signature
        for batch in ds:
            if kwargs:
                key = next(iter(kwargs.keys()))
                out = infer(**{key: batch})
            else:
                out = infer(batch)
            # Unwrap dict outputs
            if isinstance(out, dict):
                out = next(iter(out.values()))
            preds.append(out.numpy())
    except Exception:
        # Fallback: attempt simple positional call
        for batch in ds:
            out = infer(batch)
            if isinstance(out, dict):
                out = next(iter(out.values()))
            preds.append(out.numpy())
    return np.concatenate(preds, axis=0)

preds = _predict_with_savedmodel(model, img_generator) if 'model_is_savedmodel' in locals() and model_is_savedmodel else model.predict(img_generator)

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
