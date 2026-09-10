"""Classify an explicitly accounted crop inventory without changing its identity."""
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
from lib_common import read_yaml, model_img_size_mapping, update_config_from_env
from prediction_contract import (
    PREPROCESSING, atomic_json, class_names_in_order, crop_inventory,
    predict_batches, prediction_table, safe_relative, savedmodel_dispatch,
    sha256_path, validate_bundle, validate_shapes, write_predictions, write_prediction_scores,
)


def load_config(path='config.yaml'):
    config = read_yaml(path)
    types = {'MODEL': str, 'INPUT_DIR': str, 'PRED_FILE': str, 'PRED_CSV': str,
             'RENAME_SNIPS': bool, 'SNIP_DIR': str, 'BATCH_SIZE': int,
             'TOP_CLASSES': bool, 'KERAS_BACKEND': str, 'PRINT_SUMMARY': bool,
             'MODEL_PATH': str, 'USE_SAVEDMODEL': bool, 'MODEL_EXPORT_DIR': str,
             'SAFE_MODE': bool, 'XLA_JIT': str, 'CLASS_MAP_PATH': str,
             'MODEL_MANIFEST_PATH': str}
    if not isinstance(config, dict) or set(config) != set(types):
        raise ValueError('Configuration must contain exactly the documented options')
    for key, kind in types.items():
        if type(config[key]) is not kind:
            raise ValueError(f'{key} requires {kind.__name__}')
    config = update_config_from_env(config)
    if config['RENAME_SNIPS']:
        raise ValueError('RENAME_SNIPS=True is no longer supported: crop identities are immutable')
    if 'SNIP_CHARS' in os.environ:
        raise ValueError('SNIP_CHARS is obsolete: crop identities are immutable')
    if config['BATCH_SIZE'] <= 0:
        raise ValueError('BATCH_SIZE must be positive')
    if config['XLA_JIT'].lower() not in ('auto', 'on', 'off'):
        raise ValueError('XLA_JIT must be auto, on, or off')
    if config['KERAS_BACKEND'] != 'tensorflow':
        raise ValueError('Inference requires KERAS_BACKEND=tensorflow')
    for key in ('SNIP_DIR', 'PRED_FILE', 'PRED_CSV'):
        safe_relative(config[key])
    if len({config['PRED_FILE'], config['PRED_CSV'], 'prediction_manifest.json', 'prediction_scores.npz', '.mewc-predict.lock'}) != 5:
        raise ValueError('Output paths must be distinct')
    model_img_size_mapping(config['MODEL'])
    return config


def run(config):
    root = Path(config['INPUT_DIR']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if 'SNIP_DIR' in config:
        snips = (root / config['SNIP_DIR']).resolve()
        if not snips.is_relative_to(root) or snips == root:
            raise ValueError('SNIP_DIR must stay within the input directory')
        for name in ('PRED_FILE', 'PRED_CSV'):
            destination = (root / config[name]).resolve()
            if not destination.is_relative_to(root) or destination.is_relative_to(snips):
                raise ValueError('Prediction outputs must stay within the input root and outside the crop directory')
    # Prevent simultaneous predictions from racing to replace one output pair.
    with (root / '.mewc-predict.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {'schema_version': 1, 'complete': False, 'effective_config': config,
                  'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
        marker = root / 'prediction_manifest.json'
        atomic_json(marker, status)
        try:
            result = _run(config, root)
            status.update(result, complete=True)
            atomic_json(marker, status)
        except Exception as error:
            status['error'] = str(error)
            atomic_json(marker, status)
            raise


def _run(config, root):
    size = model_img_size_mapping(config['MODEL'])
    class_path = Path(config['CLASS_MAP_PATH'])
    class_hash = sha256_path(class_path)
    names = class_names_in_order(read_yaml(class_path))
    if class_hash != sha256_path(class_path):
        raise ValueError('Class map changed during preflight')
    inventory, identity_mode = crop_inventory(root / config['SNIP_DIR'], root / config['PRED_CSV'])
    saved = config['USE_SAVEDMODEL'] and Path(config['MODEL_EXPORT_DIR']).is_dir()
    source = Path(config['MODEL_EXPORT_DIR'] if saved else config['MODEL_PATH'])
    model_hash = sha256_path(source)
    manifest_path = config['MODEL_MANIFEST_PATH']
    bundle = read_yaml(manifest_path) if manifest_path else None
    model_contract, limitations = validate_bundle(bundle, config['MODEL'], names, model_hash, class_hash, size)
    if identity_mode != 'crop-manifest-v1':
        limitations.append('Historical random-name recovery has no verified source/detection association; regenerate crops for the current pipeline.')
    if inventory:
        predictions = _infer(config, root, inventory, names, size, source, saved, model_hash)
    else:
        # Hashes and the declared bundle contract were checked above. With no
        # eligible crops there is no reason to import or deserialize the model.
        predictions = predict_batches(None, [], 0, len(names))
        limitations.append('Model loading and runtime input/output shape validation skipped: no eligible crops.')
    table = prediction_table(predictions, inventory, names, config['TOP_CLASSES'])
    pickle_path, csv_path = root / config['PRED_FILE'], root / config['PRED_CSV']
    scores_path = root / 'prediction_scores.npz'
    for path in (pickle_path, csv_path, scores_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            backup = path.with_name(path.name + '.previous.' + sha256_path(path))
            if not backup.exists():
                shutil.copyfile(path, backup)
    write_prediction_scores(scores_path, predictions, inventory, names)
    write_predictions(table, pickle_path, csv_path)
    return {'identity_mode': identity_mode, 'crop_count': len(inventory),
            'crop_manifest_sha256': sha256_path(root / config['SNIP_DIR'] / 'crop_manifest.json') if identity_mode == 'crop-manifest-v1' else None,
            'model_manifest_sha256': sha256_path(manifest_path) if manifest_path else None,
            'prediction_count': len(predictions), 'output_row_count': len(table),
            'model_runtime_validated': bool(inventory),
            'skipped_reason': None if inventory else 'no-eligible-crops',
            'class_count': len(names), 'top_classes': config['TOP_CLASSES'],
            'tie_policy': 'retain-all-exact-maxima', 'model_contract': model_contract,
            'limitations': limitations,
            'outputs': {config['PRED_FILE']: sha256_path(pickle_path), config['PRED_CSV']: sha256_path(csv_path),
                        'prediction_scores.npz': sha256_path(scores_path)}}


def _infer(config, root, inventory, names, size, source, saved, model_hash):
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    import tensorflow as tf
    from keras import saving
    xla = config['XLA_JIT'].lower()
    if xla != 'auto':
        tf.config.optimizer.set_jit(xla == 'on')
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    with tempfile.TemporaryDirectory(prefix='mewc-model-') as staging:
        staged = Path(staging) / ('model_export' if saved else 'model.keras')
        if saved:
            shutil.copytree(source, staged)
        else:
            shutil.copyfile(source, staged)
        if sha256_path(staged) != model_hash or sha256_path(source) != model_hash:
            raise ValueError('Model changed during staging')
        if saved:
            model = tf.saved_model.load(str(staged))
            dispatch = savedmodel_dispatch(model, size, len(names))
        else:
            model = saving.load_model(str(staged), compile=False, safe_mode=config['SAFE_MODE'])
            if len(model.inputs) != 1 or len(model.outputs) != 1:
                raise ValueError('Keras model must have one image input and one classification output')
            validate_shapes(model.input_shape, model.output_shape, size, len(names))
            if str(model.inputs[0].dtype) != 'float32':
                raise ValueError('Keras image input must be float32')
            if config['PRINT_SUMMARY']:
                model.summary()
            dispatch = lambda batch: model(batch, training=False)
        snip_root = (root / config['SNIP_DIR']).resolve()
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            str(snip_root), labels=None, label_mode=None, color_mode='rgb',
            batch_size=config['BATCH_SIZE'], image_size=(size, size),
            interpolation='bilinear', crop_to_aspect_ratio=False, shuffle=False)
        file_order = [Path(path).resolve().relative_to(snip_root).as_posix() for path in dataset.file_paths]
        if file_order != [row['rand_name'] for row in inventory]:
            raise ValueError('TensorFlow image inventory/order differs from the validated crop inventory')
        batches = dataset.prefetch(tf.data.AUTOTUNE)
        predictions = predict_batches(dispatch, batches, len(inventory), len(names))
    return predictions


if __name__ == '__main__':
    run(load_config())
