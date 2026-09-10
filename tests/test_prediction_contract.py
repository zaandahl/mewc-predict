import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from prediction_contract import (
    PREPROCESSING, class_names_in_order, crop_inventory, predict_batches,
    prediction_table, savedmodel_dispatch, sha256_path, validate_bundle,
    validate_shapes, write_predictions,
)
from mewc_predict import load_config, run


def crops(tmp_path, records):
    for record in records:
        path = tmp_path / record['crop_file']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture image')
    (tmp_path / 'crop_manifest.json').write_text(json.dumps(
        {'schema_version': 1, 'complete': True, 'crops': records}))
    return crop_inventory(tmp_path)[0]


def record(parent='a', index=0):
    crop = f'{parent}/same-{index}.jpg'
    return dict(crop_id=crop, crop_file=crop, source_file=f'{parent}/same.jpg', detection_index=index)


def test_nested_duplicate_basenames_and_rerun_keep_identity(tmp_path):
    records = [record('a'), record('b')]
    inventory = crops(tmp_path, records)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*.jpg')}
    values = [[.5, .5], [.1, .9]]
    first = prediction_table(values, inventory, ['quoll', 'devil'], True)
    write_predictions(first, tmp_path/'out.pkl', tmp_path/'out.csv')
    second = prediction_table(values, crop_inventory(tmp_path)[0], ['quoll', 'devil'], True)
    pd.testing.assert_frame_equal(first, second)
    assert first.groupby('crop_id').size().to_dict() == {'a/same-0.jpg': 2, 'b/same-0.jpg': 1}
    assert list(first.class_id) == [0, 1, 1]
    assert set(first.class_rank) == {1}
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('*.jpg')}


def test_legacy_random_names_recovered_without_renaming(tmp_path):
    (tmp_path/'random123.jpg').write_bytes(b'original crop bytes')
    csv = tmp_path/'previous.csv'
    pd.DataFrame([{'filename':'source-2.jpg','rand_name':'random123.jpg'}]).to_csv(csv,index=False)
    rows, mode = crop_inventory(tmp_path, csv)
    table = prediction_table([[.2,.8]], rows, ['a','b'], True)
    write_predictions(table,tmp_path/'new.pkl',csv)
    assert crop_inventory(tmp_path,csv)[0] == rows
    assert mode == 'legacy-csv-recovery'
    assert rows[0]['filename'] == 'source-2.jpg'
    assert rows[0]['rand_name'] == 'random123.jpg'
    assert (tmp_path/'random123.jpg').read_bytes() == b'original crop bytes'


@pytest.mark.parametrize('root_kind', ['missing', 'deleted', 'file'])
def test_legacy_empty_inventory_requires_an_existing_snip_directory(tmp_path, root_kind):
    snip_root = tmp_path / 'snips'
    if root_kind == 'deleted':
        snip_root.mkdir()
        snip_root.rmdir()
    if root_kind == 'file':
        snip_root.write_bytes(b'not a directory')
    prior_csv = tmp_path / 'previous.csv'
    pd.DataFrame(columns=['filename', 'rand_name']).to_csv(prior_csv, index=False)
    with pytest.raises(ValueError, match='existing directory'):
        crop_inventory(snip_root, prior_csv)


def test_empty_existing_legacy_inventory_remains_valid(tmp_path):
    prior_csv = tmp_path / 'previous.csv'
    pd.DataFrame(columns=['filename', 'rand_name']).to_csv(prior_csv, index=False)
    rows, mode = crop_inventory(tmp_path, prior_csv)
    assert rows == []
    assert mode == 'legacy-csv-recovery'


@pytest.mark.parametrize('damage',['missing','extra','duplicate','incomplete','traversal'])
def test_crop_manifest_rejects_unaccounted_or_unsafe_inventory(tmp_path,damage):
    crops(tmp_path,[record()])
    path = tmp_path/'crop_manifest.json'
    manifest=json.loads(path.read_text())
    if damage=='missing': (tmp_path/'a/same-0.jpg').unlink()
    if damage=='extra': (tmp_path/'extra.jpg').write_bytes(b'junk')
    if damage=='duplicate': manifest['crops'].append(record())
    if damage=='incomplete': manifest['complete']=False
    if damage=='traversal': manifest['crops'][0]['source_file']='../secret.jpg'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): crop_inventory(tmp_path)


@pytest.mark.parametrize('mapping',[{0:'a',1:'a'},{1:'a'},{0:'a',2:'b'},{'0':'a'},{False:'a'},{0:''},{}])
def test_class_map_rejects_collapsing_and_ambiguous_indices(mapping):
    with pytest.raises(ValueError): class_names_in_order(mapping)


def test_class_map_orders_explicit_integer_indices():
    assert class_names_in_order({1:'b',0:'a'}) == ['a','b']


def signature(call, named=True, outputs=1):
    spec=SimpleNamespace(shape=(None,384,384,3),dtype=SimpleNamespace(name='float32'))
    class Infer:
        structured_input_signature=((),{'images':spec}) if named else ((spec,),{})
        structured_outputs={str(i):SimpleNamespace(shape=(None,2)) for i in range(outputs)}
        def __call__(self,*args,**kwargs): return {'0':call(args[0] if args else kwargs['images'])}
    return SimpleNamespace(signatures={'serving_default':Infer()})


@pytest.mark.parametrize('named',[True,False])
def test_dispatch_chosen_before_iteration_no_retry_after_partial_failure(named):
    calls=[]
    def infer(batch):
        calls.append(len(batch))
        if len(calls)==2: raise RuntimeError('injected device failure')
        return np.full((len(batch),2),.5)
    dispatch=savedmodel_dispatch(signature(infer,named),384,2)
    with pytest.raises(RuntimeError,match='injected'):
        predict_batches(dispatch,[np.zeros((1,1)),np.zeros((1,1))],2,2)
    assert calls == [1,1]


def test_multi_output_dispatch_fails_before_inference():
    calls=[]
    with pytest.raises(ValueError,match='exactly one classification'):
        savedmodel_dispatch(signature(lambda b:calls.append(b),outputs=2),384,2)
    assert calls == []


@pytest.mark.parametrize('shape',[(1,3),(0,2),(2,2)])
def test_wrong_batch_count_or_class_dimension_fails(shape):
    with pytest.raises(ValueError,match='prediction shape'):
        predict_batches(lambda _:np.zeros(shape),[np.zeros((1,1))],1,2)


def test_total_output_count_and_nonfinite_scores_fail():
    with pytest.raises(ValueError): predict_batches(lambda _:np.full((1,2),.5),[np.zeros((1,1))],2,2)
    with pytest.raises(ValueError,match='Nonfinite'): prediction_table([[np.nan,.2]],[{}],['a','b'],True)


def test_empty_complete_inventory_emits_header_only(tmp_path):
    rows=crops(tmp_path,[])
    table=prediction_table(np.empty((0,2)),rows,['a','b'],True)
    write_predictions(table,tmp_path/'out.pkl',tmp_path/'out.csv')
    assert pd.read_csv(tmp_path/'out.csv').empty


def test_model_bundle_binds_shapes_class_order_preprocessing_and_hashes(tmp_path):
    path=tmp_path/'model.keras'; path.write_bytes(b'frozen model')
    checksum=sha256_path(path)
    expected,limitations=validate_bundle(None,'VTL',['a','b'],checksum,'classhash',384)
    assert limitations
    bundle={'schema_version':1,**expected}
    assert 'unverified' in validate_bundle(bundle,'ViTL',['a','b'],checksum,'classhash',384)[1][0]
    for key,value in [('architecture','ENS'),('class_order',['b','a']),('model_sha256','bad'),
                      ('class_map_sha256','bad'),('input_shape',[None,224,224,3]),('preprocessing',{})]:
        with pytest.raises(ValueError,match=key):
            validate_bundle({**bundle,key:value},'VTL',['a','b'],checksum,'classhash',384)
    validate_shapes((None,384,384,3),(None,2),384,2)
    with pytest.raises(ValueError): validate_shapes((None,224,224,3),(None,2),384,2)


def test_failed_run_invalidates_prior_completion_without_touching_crops(tmp_path,monkeypatch):
    (tmp_path/'prediction_manifest.json').write_text('{"complete": true}')
    (tmp_path/'crop.jpg').write_bytes(b'keep')
    def fail(*args): raise RuntimeError('failure after first output replacement')
    monkeypatch.setattr('mewc_predict._run',fail)
    with pytest.raises(RuntimeError): run({'INPUT_DIR':str(tmp_path)})
    assert json.loads((tmp_path/'prediction_manifest.json').read_text())['complete'] is False
    assert (tmp_path/'crop.jpg').read_bytes() == b'keep'


def test_config_rejects_ineffective_or_unknown_architecture_options(monkeypatch):
    path=Path(__file__).parents[1]/'src/config.yaml'
    monkeypatch.setenv('RENAME_SNIPS','True')
    with pytest.raises(ValueError,match='immutable'): load_config(path)
    monkeypatch.setenv('RENAME_SNIPS','False')
    monkeypatch.setenv('MODEL','VTLTYPO')
    with pytest.raises(ValueError,match='Unknown model'): load_config(path)


def test_runtime_preflight_inference_and_completion_with_mocked_keras(tmp_path,monkeypatch):
    import sys
    from mewc_predict import run, load_config
    snips=tmp_path/'snips'
    snips.mkdir()
    crops(snips,[record('a'),record('b')])
    (tmp_path/'classes.yaml').write_text('0: quoll\n1: devil\n')
    (tmp_path/'model.keras').write_bytes(b'fixture model artifact')
    calls=[]
    class Model:
        inputs=[SimpleNamespace(dtype='float32')]
        outputs=[object()]
        input_shape=(None,384,384,3)
        output_shape=(None,2)
        def __call__(self,*args,**kwargs):
            raise AssertionError('Keras inference must preserve model.predict execution')
        def predict(self,batches,verbose):
            assert verbose == 0
            count=sum(len(batch) for batch in batches)
            calls.append(count)
            return np.array([[.5,.5],[.1,.9]])[:count]
    class Dataset:
        file_paths=[str(snips/'a/same-0.jpg'),str(snips/'b/same-0.jpg')]
        def prefetch(self,*args): return [np.zeros((2,1,1,3))]
    tf=SimpleNamespace(
        config=SimpleNamespace(list_physical_devices=lambda _:[]),
        data=SimpleNamespace(AUTOTUNE=1),
        keras=SimpleNamespace(preprocessing=SimpleNamespace(
            image_dataset_from_directory=lambda *args,**kwargs:Dataset())))
    monkeypatch.setitem(sys.modules,'tensorflow',tf)
    monkeypatch.setitem(sys.modules,'keras',SimpleNamespace(saving=SimpleNamespace(load_model=lambda *args,**kwargs:Model())))
    config=load_config(Path(__file__).parents[1]/'src/config.yaml')
    config.update(INPUT_DIR=str(tmp_path),MODEL='VTL',MODEL_PATH=str(tmp_path/'model.keras'),
                  CLASS_MAP_PATH=str(tmp_path/'classes.yaml'),USE_SAVEDMODEL=False)
    run(config)
    manifest=json.loads((tmp_path/'prediction_manifest.json').read_text())
    assert manifest['complete'] is True
    assert manifest['crop_count'] == manifest['prediction_count'] == 2
    assert manifest['output_row_count'] == 3
    assert manifest['model_runtime_validated'] is True
    assert manifest['skipped_reason'] is None
    assert manifest['outputs']['prediction_scores.npz']==sha256_path(tmp_path/'prediction_scores.npz')
    with np.load(tmp_path/'prediction_scores.npz',allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['probabilities'],[[.5,.5],[.1,.9]])
    assert manifest['outputs']['mewc_out.csv']==sha256_path(tmp_path/'mewc_out.csv')
    original=(tmp_path/'mewc_out.csv').read_bytes()
    run(config)
    assert original==(tmp_path/'mewc_out.csv').read_bytes()
    assert len(list(tmp_path.glob('mewc_out.csv.previous.*')))==1
    assert calls==[2,2]
    # A malformed class dimension must fail before any additional model call.
    (tmp_path/'classes.yaml').write_text('0: quoll\n1: devil\n2: possum\n')
    with pytest.raises(ValueError,match='output shape'): run(config)
    assert calls==[2,2]
    assert json.loads((tmp_path/'prediction_manifest.json').read_text())['complete'] is False
    assert original==(tmp_path/'mewc_out.csv').read_bytes()


@pytest.mark.parametrize('saved',[False,True])
def test_empty_runtime_validates_declaration_but_never_imports_or_loads_model(tmp_path,monkeypatch,saved):
    import builtins
    snips=tmp_path/'snips'; snips.mkdir()
    crops(snips,[])
    class_path=tmp_path/'classes.yaml'; class_path.write_text('0: quoll\n1: devil\n')
    model_path=tmp_path/'model.keras'; model_path.write_bytes(b'fixture frozen artifact')
    if saved:
        model_path=tmp_path/'model_export'; model_path.mkdir()
        (model_path/'saved_model.pb').write_bytes(b'fixture SavedModel')
    config=load_config(Path(__file__).parents[1]/'src/config.yaml')
    config.update(INPUT_DIR=str(tmp_path),MODEL='VTL',MODEL_PATH=str(model_path),
                  MODEL_EXPORT_DIR=str(model_path),CLASS_MAP_PATH=str(class_path),USE_SAVEDMODEL=saved)
    declaration,_=validate_bundle(None,'VTL',['quoll','devil'],sha256_path(model_path),sha256_path(class_path),384)
    bundle=tmp_path/'bundle.json'; bundle.write_text(json.dumps({'schema_version':1,**declaration}))
    config['MODEL_MANIFEST_PATH']=str(bundle)
    real_import=builtins.__import__
    def no_model_import(name,*args,**kwargs):
        if name.split('.')[0] in ('tensorflow','keras'):
            raise AssertionError('An empty run must not import the inference runtime')
        return real_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',no_model_import)
    def no_staging(*args,**kwargs): raise AssertionError('An empty run must not stage/load the model')
    monkeypatch.setattr('mewc_predict.tempfile.TemporaryDirectory',no_staging)
    monkeypatch.setattr('mewc_predict._infer',no_staging)
    run(config)
    manifest=json.loads((tmp_path/'prediction_manifest.json').read_text())
    assert manifest['complete'] is True
    assert manifest['model_runtime_validated'] is False
    assert manifest['skipped_reason']=='no-eligible-crops'
    assert manifest['crop_count']==manifest['prediction_count']==manifest['output_row_count']==0
    assert pd.read_csv(tmp_path/'mewc_out.csv').empty
    assert pd.read_pickle(tmp_path/'mewc_out.pkl').empty
    with np.load(tmp_path/'prediction_scores.npz',allow_pickle=False) as archive:
        assert archive['probabilities'].shape==(0,2)
        assert archive['crop_ids'].shape==(0,)
        assert list(archive['class_order'])==['quoll','devil']
    for filename,digest in manifest['outputs'].items():
        assert sha256_path(tmp_path/filename)==digest
    # Empty crops must not bypass model/class-map declaration validation.
    bundle.write_text(json.dumps({'schema_version':1,**declaration,'model_sha256':'wrong'}))
    with pytest.raises(ValueError,match='model_sha256'): run(config)
    assert json.loads((tmp_path/'prediction_manifest.json').read_text())['complete'] is False


@pytest.mark.parametrize('manifest_text', ['', 'null\n'])
def test_supplied_empty_or_null_model_manifest_fails_with_contiguous_classmap(tmp_path, manifest_text):
    snips = tmp_path / 'snips'
    snips.mkdir()
    (snips / 'crop_manifest.json').write_text(
        json.dumps({'schema_version': 1, 'complete': True, 'crops': []}))
    class_path = tmp_path / 'classes.yaml'
    class_path.write_text('0: quoll\n1: devil\n')
    model_path = tmp_path / 'model.keras'
    model_path.write_bytes(b'fixture frozen artifact')
    bundle = tmp_path / 'bundle.yaml'
    bundle.write_text(manifest_text)
    config = load_config(Path(__file__).parents[1] / 'src/config.yaml')
    config.update(INPUT_DIR=str(tmp_path), MODEL='VTL', MODEL_PATH=str(model_path),
                  CLASS_MAP_PATH=str(class_path), MODEL_MANIFEST_PATH=str(bundle),
                  USE_SAVEDMODEL=False)
    with pytest.raises(ValueError, match='explicit class_ids'):
        run(config)


def test_empty_model_manifest_path_preserves_no_manifest_mode(tmp_path, monkeypatch):
    snips = tmp_path / 'snips'
    snips.mkdir()
    (snips / 'crop_manifest.json').write_text(
        json.dumps({'schema_version': 1, 'complete': True, 'crops': []}))
    class_path = tmp_path / 'classes.yaml'
    class_path.write_text('0: quoll\n1: devil\n')
    model_path = tmp_path / 'model.keras'
    model_path.write_bytes(b'fixture frozen artifact')
    config = load_config(Path(__file__).parents[1] / 'src/config.yaml')
    config.update(INPUT_DIR=str(tmp_path), MODEL='VTL', MODEL_PATH=str(model_path),
                  CLASS_MAP_PATH=str(class_path), MODEL_MANIFEST_PATH='',
                  USE_SAVEDMODEL=False)
    monkeypatch.setattr('mewc_predict._infer',
                        lambda *args: pytest.fail('empty run must not load the model'))
    run(config)
    manifest = json.loads((tmp_path / 'prediction_manifest.json').read_text())
    assert manifest['complete'] is True
    assert manifest['crop_count'] == 0
    assert manifest['model_contract']['class_ids'] == [0, 1]


def test_score_archive_preserves_full_matrix_dtype_and_order(tmp_path):
    from prediction_contract import write_prediction_scores
    values=np.array([[.5,.5],[.125,.875]],dtype=np.float32)
    rows=[{'crop_id':'nested/a-0.jpg'},{'crop_id':'other/a-0.jpg'}]
    path=tmp_path/'scores.npz'
    write_prediction_scores(path,values,rows,['quoll','devil'])
    with np.load(path,allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['probabilities'],values)
        assert archive['probabilities'].dtype==values.dtype
        assert list(archive['crop_ids'])==[r['crop_id'] for r in rows]
        assert list(archive['class_order'])==['quoll','devil']
        assert all(archive[key].dtype.kind != 'O' for key in archive.files)
    before=path.read_bytes()
    with pytest.raises(ValueError,match='prediction shape'):
        write_prediction_scores(path,values[:1],rows,['quoll','devil'])
    assert path.read_bytes()==before


def test_failed_score_archive_write_keeps_prior_archive(tmp_path,monkeypatch):
    from prediction_contract import write_prediction_scores
    path=tmp_path/'scores.npz'; path.write_bytes(b'previous complete archive')
    def fail(stream,**kwargs):
        stream.write(b'partial archive')
        raise OSError('injected score write failure')
    monkeypatch.setattr('prediction_contract.np.savez_compressed',fail)
    with pytest.raises(OSError,match='injected'):
        write_prediction_scores(path,np.full((1,2),.5),[{'crop_id':'a.jpg'}],['a','b'])
    assert path.read_bytes()==b'previous complete archive'
    assert list(tmp_path.iterdir())==[path]


@pytest.mark.parametrize('probabilities',[[[-.1,1.1]], [[.8,.8]], [[0.,0.]], [[2.,-1.]]])
def test_logits_out_of_range_and_unnormalized_probabilities_are_rejected(probabilities):
    from prediction_contract import validate_predictions
    with pytest.raises(ValueError,match='Prediction probabilities'):
        validate_predictions(probabilities,1,2)


def test_probability_conservation_accepts_float32_roundoff_and_empty_rows():
    from prediction_contract import validate_predictions
    probabilities=np.array([[.33333334,.33333334,.33333334]],dtype=np.float32)
    assert validate_predictions(probabilities,1,3) is probabilities
    assert validate_predictions(np.empty((0,3)),0,3).shape==(0,3)



def test_explicit_lexical_class_codes_preserve_axis_mapping_ranks_and_archive(tmp_path):
    from prediction_contract import class_ids_in_order, write_prediction_scores
    mapping={'999':'bait','2':'quoll','0':'blank','10':'devil'}
    codes=['0','10','2','999']
    names=class_names_in_order(mapping,codes)
    assert names==['blank','devil','quoll','bait']
    assert class_ids_in_order(mapping,codes)==codes
    inventory=[{'crop_id':'camera/img-0.jpg','filename':'camera/img-0.jpg',
                'rand_name':'camera/img-0.jpg','source_file':'camera/img.jpg','detection_index':0}]
    probabilities=np.array([[.1,.6,.1,.2]],dtype=np.float32)
    table=prediction_table(probabilities,inventory,names,True,codes)
    assert table.class_id.tolist()==['10']
    assert table.class_index.tolist()==[1]
    assert table.class_name.tolist()==['devil']
    assert table.class_rank.tolist()==[1]
    path=tmp_path/'scores.npz'
    write_prediction_scores(path,probabilities,inventory,names,codes)
    with np.load(path,allow_pickle=False) as archive:
        assert archive['class_ids'].dtype.kind=='U'
        assert archive['class_ids'].tolist()==codes
        np.testing.assert_array_equal(archive['probabilities'],probabilities)
    tied=prediction_table([[.05,.45,.05,.45]],inventory,names,True,codes)
    assert tied.class_id.tolist()==['10','999']
    assert tied.class_index.tolist()==[1,3]
    assert tied.class_rank.tolist()==[1,1]
    write_predictions(table,tmp_path/'out.pkl',tmp_path/'out.csv')
    assert pd.read_csv(tmp_path/'out.csv',dtype={'class_id':str}).class_id.tolist()==['10']
    assert pd.read_pickle(tmp_path/'out.pkl').class_id.tolist()==['10']


@pytest.mark.parametrize('mapping,codes',[
    ({'0':'a','10':'b'},None),
    ({0:'a',10:'b'},None),
    ({'0':'a','10':'b'},['0','0']),
    ({'0':'a','10':'b'},['0','2']),
    ({'0':'a','10':'b'},[0,10]),
    ({0:'a',10:'b'},['0','10']),
    ({'0':'a',10:'b'},['0',10]),
    ({'01':'a','1':'b'},['01','1']),
    ({-1:'a',0:'b'},[-1,0]),
    ({0:'a',1:'b'},[False,True]),
    ({'0':'a','10':'a'},['0','10']),
])
def test_ambiguous_incomplete_or_coerced_class_code_orders_fail(mapping,codes):
    with pytest.raises(ValueError): class_names_in_order(mapping,codes)


def test_noncontiguous_integer_codes_and_explicit_reordered_axes(tmp_path):
    from prediction_contract import write_prediction_scores
    codes=[999,10,0]
    names=class_names_in_order({0:'blank',10:'devil',999:'bait'},codes)
    assert names==['bait','devil','blank']
    path=tmp_path/'scores.npz'
    write_prediction_scores(path,[[.2,.7,.1]],[{'crop_id':'a.jpg'}],names,codes)
    with np.load(path,allow_pickle=False) as archive:
        assert archive['class_ids'].dtype==np.dtype('int64')
        assert archive['class_ids'].tolist()==codes


def test_declared_codes_must_agree_with_names_and_preserve_provenance():
    codes=['0','10','2','999']
    names=['blank','devil','quoll','bait']
    expected,_=validate_bundle(None,'VTL',names,'modelhash','classhash',384,codes)
    bundle={'schema_version':1,**expected,'class_order_provenance':'historical-predictor-lexical-order-unverified'}
    contract,limitations=validate_bundle(bundle,'VTL',names,'modelhash','classhash',384,codes)
    assert contract['class_ids']==codes
    assert contract['class_order_provenance']==bundle['class_order_provenance']
    assert limitations and 'unverified' in limitations[0]
    with pytest.raises(ValueError,match='class_order'):
        validate_bundle({**bundle,'class_order':['blank','quoll','devil','bait']},'VTL',names,'modelhash','classhash',384,codes)
    with pytest.raises(ValueError,match='class_ids'):
        validate_bundle({**bundle,'class_ids':['0','2','10','999']},'VTL',names,'modelhash','classhash',384,codes)


def test_runtime_uses_declared_string_codes_without_relabelling(tmp_path,monkeypatch):
    snips=tmp_path/'snips'; snips.mkdir()
    crops(snips,[record()])
    class_path=tmp_path/'classes.yaml'
    class_path.write_text("'999': bait\n'2': quoll\n'0': blank\n'10': devil\n")
    model_path=tmp_path/'model.keras'; model_path.write_bytes(b'frozen fixture')
    codes=['0','10','2','999']; names=['blank','devil','quoll','bait']
    contract,_=validate_bundle(None,'VTL',names,sha256_path(model_path),sha256_path(class_path),384,codes)
    bundle=tmp_path/'bundle.json'
    bundle.write_text(json.dumps({'schema_version':1,**contract,'class_order_provenance':'historical-predictor-lexical-order-unverified'}))
    config=load_config(Path(__file__).parents[1]/'src/config.yaml')
    config.update(INPUT_DIR=str(tmp_path),MODEL='VTL',MODEL_PATH=str(model_path),
                  CLASS_MAP_PATH=str(class_path),MODEL_MANIFEST_PATH=str(bundle),USE_SAVEDMODEL=False)
    calls=[]
    def infer(config,root,inventory,actual_names,*args):
        calls.append(actual_names)
        return np.array([[.1,.6,.1,.2]],dtype=np.float32)
    monkeypatch.setattr('mewc_predict._infer',infer)
    run(config)
    manifest=json.loads((tmp_path/'prediction_manifest.json').read_text())
    assert calls==[names]
    assert manifest['complete'] is True
    assert manifest['model_contract']['class_ids']==codes
    assert any('unverified' in item for item in manifest['limitations'])
    table=pd.read_pickle(tmp_path/'mewc_out.pkl')
    assert table.class_id.tolist()==['10']
    assert table.class_index.tolist()==[1]
    with np.load(tmp_path/'prediction_scores.npz',allow_pickle=False) as archive:
        assert archive['class_ids'].tolist()==codes
    # Omitting the declaration cannot silently infer lexical or numeric order.
    config['MODEL_MANIFEST_PATH']=''
    with pytest.raises(ValueError,match='explicit model manifest class_ids'): run(config)
    assert calls==[names]



def test_keras_dispatch_calls_predict_once_preserving_scores_and_dataset():
    from prediction_contract import predict_keras
    batches=object()
    probabilities=np.array([[.125,.875],[.5,.5]],dtype=np.float32)
    calls=[]
    class Model:
        def __call__(self,*args,**kwargs):
            raise AssertionError('Direct eager execution changes the numerical path')
        def predict(self,dataset,verbose):
            calls.append((dataset,verbose))
            return probabilities
    result=predict_keras(Model(),batches,2,2)
    assert calls==[(batches,0)]
    assert result is probabilities
    np.testing.assert_array_equal(result,probabilities)


def test_keras_dispatch_failure_is_not_retried_or_replaced_by_eager_calls():
    from prediction_contract import predict_keras
    calls=[]
    class Model:
        def __call__(self,*args,**kwargs):
            raise AssertionError('No eager fallback is allowed')
        def predict(self,dataset,verbose):
            calls.append(dataset)
            raise RuntimeError('injected predict failure after internal partial work')
    dataset=object()
    with pytest.raises(RuntimeError,match='injected predict failure'):
        predict_keras(Model(),dataset,2,2)
    assert calls==[dataset]


@pytest.mark.parametrize('returned',[
    [[.2,.8],[.3,.7]],  # Too many rows.
    [[.2,.3,.5]],  # Incorrect output width.
    [[float('nan'),.8]],
    [[-.1,1.1]],
    [[.2,.2]],
])
def test_keras_dispatch_retains_shape_and_probability_validation(returned):
    from prediction_contract import predict_keras
    calls=[]
    class Model:
        def predict(self,dataset,verbose):
            calls.append(dataset)
            return np.asarray(returned)
    with pytest.raises(ValueError):
        predict_keras(Model(),'ordered dataset',1,2)
    assert calls==['ordered dataset']
