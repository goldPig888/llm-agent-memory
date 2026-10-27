import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image


def module():
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('run_native_pointing',root/'scripts/run_native_pointing.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def test_native_parser_matches_gist_keys_and_1000_center():
    m=module()
    box,point=m.parse_native_box("Answer {'x_min': 380, 'y_min': 290, 'x_max': 470, 'y_max': 410}")
    assert point==pytest.approx([.425,.35]) and box['x_min']==380
    for bad in ['{"x_min":-1,"y_min":0,"x_max":100,"y_max":100}',
                '{"x_min":400,"y_min":0,"x_max":100,"y_max":100}',
                '{"x_min":true,"y_min":0,"x_max":100,"y_max":100}',
                '{"x_min":0,"y_min":0,"x_max":1001,"y_max":100}',
                '{"x_min":__import__("os"),"y_min":0,"x_max":100,"y_max":100}']:
        with pytest.raises(ValueError):m.parse_native_box(bad)


def test_native_input_uses_raw_image_and_excludes_labels(tmp_path):
    m=module();Image.fromarray(np.zeros((10,10,3),np.uint8)).save(tmp_path/'raw.png')
    sample={'image':'raw.png','scoring_secret':'DO_NOT_SEND'}
    message=m.native_message(sample,tmp_path)
    assert 'DO_NOT_SEND' not in str(message)
    assert '1000' not in message['content'][0]['text']
    assert 'x_min' not in message['content'][0]['text']
    assert 'native object-detection format' in message['content'][0]['text']
