"""Prediction integrity checks, independent of TensorFlow for fixture testing."""
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
import numpy as np
import pandas as pd
from lib_common import canonical_model_name

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
PREPROCESSING = {'color_mode': 'rgb', 'interpolation': 'bilinear',
                 'crop_to_aspect_ratio': False, 'value_range': [0, 255],
                 'external_normalization': 'none', 'dtype': 'float32'}


def sha256_path(path):
    """Files: standard SHA256; directories: sorted path NUL content-digest lines."""
    path = Path(path)
    if path.is_file():
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()
    if not path.is_dir():
        raise ValueError(f'Model path missing: {path}')
    digest = hashlib.sha256()
    for entry in sorted(path.rglob('*')):
        if entry.is_symlink():
            raise ValueError(f'Symlink in model bundle: {entry}')
        if entry.is_file():
            digest.update(entry.relative_to(path).as_posix().encode() + b'\0')
            digest.update(sha256_path(entry).encode() + b'\n')
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name + '.')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def safe_relative(value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError(f'Invalid relative path: {value!r}')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or value != path.as_posix():
        raise ValueError(f'Invalid relative path: {value!r}')
    return value


def _validate_class_codes(codes):
    if not codes:
        raise ValueError('Class codes must be nonempty')
    kinds = {type(code) for code in codes}
    if kinds == {str}:
        # Canonical decimal strings prevent aliases such as "01" and "1".
        if any(not code.isascii() or not code.isdecimal() or str(int(code)) != code for code in codes):
            raise ValueError('String class codes must be canonical nonnegative decimal codes without aliases')
    elif kinds == {int}:
        if any(code < 0 or code > np.iinfo(np.int64).max for code in codes):
            raise ValueError('Integer class codes must be nonnegative int64 values')
    else:
        raise ValueError('Class codes must be uniformly strings or integers; mixed types and booleans are invalid')
    if len(set(codes)) != len(codes):
        raise ValueError('Class codes must be unique')
    return codes


def class_ids_in_order(class_map, class_ids=None):
    """Resolve model axes to original class codes without relabelling them."""
    if not isinstance(class_map, dict) or not class_map:
        raise ValueError('Class map must be a nonempty mapping of class codes to names')
    codes = _validate_class_codes(list(class_map))
    if class_ids is None:
        if type(codes[0]) is not int or sorted(codes) != list(range(len(codes))):
            raise ValueError('String or noncontiguous class codes require explicit model manifest class_ids')
        return list(range(len(codes)))
    if not isinstance(class_ids, list):
        raise ValueError('Model manifest class_ids must be an ordered array')
    _validate_class_codes(class_ids)
    if type(class_ids[0]) is not type(codes[0]) or set(class_ids) != set(codes):
        raise ValueError('Model manifest class_ids must exactly cover the original class-map codes without coercion')
    return list(class_ids)


def class_names_in_order(class_map, class_ids=None):
    codes = class_ids_in_order(class_map, class_ids)
    names = [class_map[code] for code in codes]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError('Class names must be nonempty strings')
    if len(set(names)) != len(names):
        raise ValueError('Class names must be unique')
    return names


def _axis_class_ids(names, class_ids):
    codes = list(range(len(names))) if class_ids is None else class_ids
    if not isinstance(codes, list) or len(codes) != len(names):
        raise ValueError('Class ID count must match output-axis names')
    return _validate_class_codes(codes)


def crop_inventory(snip_root, prior_csv=None):
    """Authoritative new manifests; read-only recovery for historical random names."""
    root = Path(snip_root).resolve()
    disk = {}
    for path in root.rglob('*'):
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError(f'Unsafe crop path: {path}')
            disk[path.relative_to(root).as_posix()] = path
    manifest_path = root / 'crop_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('schema_version') != 1 or manifest.get('complete') is not True:
            raise ValueError('Crop manifest is incomplete or unsupported')
        records = manifest.get('crops')
        if not isinstance(records, list):
            raise ValueError('Crop manifest crops must be a list')
        rows = []
        for record in records:
            crop_file = safe_relative(record['crop_file'])
            crop_id = safe_relative(record['crop_id'])
            if crop_id != crop_file:
                raise ValueError('Manifest crop_id must equal immutable crop_file')
            source = safe_relative(record['source_file'])
            index = record['detection_index']
            if type(index) is not int or index < 0:
                raise ValueError('Detection index must be a nonnegative integer')
            rows.append(dict(crop_id=crop_id, filename=crop_id, rand_name=crop_file,
                             source_file=source, detection_index=index))
        if len({r['crop_id'] for r in rows}) != len(rows) or len({r['rand_name'] for r in rows}) != len(rows):
            raise ValueError('Duplicate crop identity or crop file')
        if len({(r['source_file'], r['detection_index']) for r in rows}) != len(rows):
            raise ValueError('Duplicate source/detection identity')
        expected = {r['rand_name'] for r in rows}
        if set(disk) != expected:
            raise ValueError(f'Crop inventory mismatch: missing={sorted(expected-set(disk))}, extra={sorted(set(disk)-expected)}')
        return sorted(rows, key=lambda r: r['rand_name']), 'crop-manifest-v1'
    # Historical recovery is explicitly mapping-based. Guessing a random name's
    # source after an old interrupted rename would manufacture an identity.
    if prior_csv is None or not Path(prior_csv).is_file():
        raise ValueError('Missing crop_manifest.json; regenerate snips or supply the existing prediction CSV for legacy recovery')
    previous = pd.read_csv(prior_csv, keep_default_na=False)
    if not {'filename', 'rand_name'}.issubset(previous.columns):
        raise ValueError('Legacy recovery needs filename and rand_name columns')
    rows, seen = [], set()
    for original, actual in previous[['filename', 'rand_name']].drop_duplicates().itertuples(index=False, name=None):
        original, actual = safe_relative(original), safe_relative(actual or original)
        if original in seen:
            raise ValueError('Legacy mapping contains conflicting crop identities')
        seen.add(original)
        rows.append(dict(crop_id=original, filename=original, rand_name=actual,
                         source_file='', detection_index=None))
    if len({r['rand_name'] for r in rows}) != len(rows) or set(disk) != {r['rand_name'] for r in rows}:
        raise ValueError('Legacy mapping does not uniquely account for every crop; restore the original mapping or regenerate snips')
    return sorted(rows, key=lambda r: r['rand_name']), 'legacy-csv-recovery'


def validate_shapes(input_shape, output_shape, size, count):
    input_shape, output_shape = tuple(input_shape), tuple(output_shape)
    if len(input_shape) != 4 or input_shape[0] not in (None, -1) or input_shape[1:] != (size, size, 3):
        raise ValueError(f'Model input shape {input_shape} does not match dynamic-batch RGB {size}x{size}')
    if len(output_shape) != 2 or output_shape[0] not in (None, -1) or output_shape[1] != count:
        raise ValueError(f'Model output shape {output_shape} does not match {count} class indices')


def savedmodel_dispatch(model, size, count):
    signatures = model.signatures
    if 'serving_default' in signatures:
        infer = signatures['serving_default']
    elif len(signatures) == 1:
        infer = next(iter(signatures.values()))
    else:
        raise ValueError('SavedModel requires serving_default or exactly one signature')
    args, kwargs = infer.structured_input_signature
    if not args and len(kwargs) == 1:
        key, spec = next(iter(kwargs.items()))
        call = lambda batch: infer(**{key: batch})
    elif len(args) == 1 and not kwargs:
        spec = args[0]
        call = lambda batch: infer(batch)
    else:
        raise ValueError('SavedModel signature must accept exactly one image tensor')
    outputs = infer.structured_outputs
    if isinstance(outputs, dict):
        if len(outputs) != 1:
            raise ValueError('SavedModel must expose exactly one classification output')
        output_key, output_spec = next(iter(outputs.items()))
    else:
        output_key, output_spec = None, outputs
    validate_shapes(spec.shape, output_spec.shape, size, count)
    if getattr(spec.dtype, 'name', str(spec.dtype)) != 'float32':
        raise ValueError('SavedModel image input must be float32')
    def dispatch(batch):
        result = call(batch)  # Never retry a failed/partially accumulated run.
        return result[output_key] if output_key is not None else result
    return dispatch


def validate_predictions(predictions, rows, classes):
    values = np.asarray(predictions)
    if values.shape != (rows, classes):
        raise ValueError(f'Expected prediction shape {(rows, classes)}, received {values.shape}')
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite prediction scores')
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError('Prediction probabilities must be within [0, 1]')
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError('Prediction probabilities must sum to one per crop')
    return values


def predict_batches(dispatch, batches, expected_rows, classes):
    chunks = []
    for batch in batches:
        result = dispatch(batch)
        if hasattr(result, 'numpy'):
            result = result.numpy()
        chunks.append(validate_predictions(result, len(batch), classes))
    output = np.concatenate(chunks, axis=0) if chunks else np.empty((0, classes))
    return validate_predictions(output, expected_rows, classes)


def prediction_table(predictions, inventory, names, top_only, class_ids=None):
    codes = _axis_class_ids(names, class_ids)
    values = validate_predictions(predictions, len(inventory), len(names))
    records = []
    for crop, scores in zip(inventory, values):
        # Competition rank preserves all exactly tied maxima without inventing
        # a scientific tie-breaking policy. Class index controls stable ordering.
        for index, score in enumerate(scores):
            rank = 1 + int(np.count_nonzero(scores > score))
            if not top_only or rank == 1:
                records.append({**crop, 'label': str(PurePosixPath(crop['rand_name']).parent),
                                'class_id': codes[index], 'class_index': index, 'prob': float(score),
                                'class_name': names[index], 'class_rank': rank})
    columns = ['crop_id', 'filename', 'rand_name', 'source_file', 'detection_index',
               'label', 'class_id', 'class_index', 'prob', 'class_name', 'class_rank']
    table = pd.DataFrame(records, columns=columns)
    if len(inventory) and table['crop_id'].nunique() != len(inventory):
        raise ValueError('Output failed to account for every input crop')
    return table


def validate_bundle(manifest, architecture, names, model_hash, class_hash, size, class_ids=None):
    codes = _axis_class_ids(names, class_ids)
    expected = {'architecture': canonical_model_name(architecture), 'class_order': names, 'class_ids': codes,
                'model_sha256': model_hash, 'class_map_sha256': class_hash,
                'input_shape': [None, size, size, 3], 'preprocessing': PREPROCESSING}
    if manifest is None:
        return expected, ['Training/export class order and architecture provenance unverified: no model bundle manifest supplied.']
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 1:
        raise ValueError('Unsupported model bundle manifest')
    _axis_class_ids(names, manifest.get('class_ids'))
    for key, value in expected.items():
        actual = manifest.get(key)
        if key == 'architecture' and actual is not None:
            actual = canonical_model_name(actual)
        if actual != value:
            raise ValueError(f'Model bundle {key} mismatch: expected {value!r}, received {actual!r}')
    provenance = manifest.get('class_order_provenance', 'declared-unverified')
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError('class_order_provenance must be a nonempty string')
    expected['class_order_provenance'] = provenance
    limitations = [] if provenance == 'training-export-verified' else [
        'Class order follows the supplied axis/code declaration; training/export order provenance is unverified.']
    return expected, limitations


def write_predictions(table, pickle_path, csv_path):
    """Stage both formats before replacement; run completion is a separate marker."""
    temporary = []
    try:
        for destination, writer in ((Path(pickle_path), table.to_pickle), (Path(csv_path), table.to_csv)):
            fd, name = tempfile.mkstemp(dir=destination.parent, prefix='.' + destination.name + '.')
            os.close(fd)
            temporary.append((name, destination))
            if destination == Path(csv_path):
                writer(name, index=False)
            else:
                writer(name)
            with open(name, 'rb') as stream:
                os.fsync(stream.fileno())
        for name, destination in temporary:
            os.replace(name, destination)
    finally:
        for name, _ in temporary:
            if os.path.exists(name):
                os.unlink(name)


def write_prediction_scores(path, predictions, inventory, names, class_ids=None):
    """Persist all unmodified scores and their exact row/column identities."""
    codes = _axis_class_ids(names, class_ids)
    values = validate_predictions(predictions, len(inventory), len(names))
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name + '.')
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, probabilities=values,
                                crop_ids=np.asarray([row['crop_id'] for row in inventory], dtype=str),
                                class_order=np.asarray(names, dtype=str),
                                class_ids=np.asarray(codes, dtype=str if type(codes[0]) is str else np.int64))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
