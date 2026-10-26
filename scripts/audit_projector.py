"""Report actual GGUF tensor types; a filename is not a precision guarantee."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('projector',type=Path);p.add_argument('--gguf-python',required=True,type=Path)
    p.add_argument('--require-unquantized',action='store_true');args=p.parse_args()
    sys.path.insert(0,str(args.gguf_python))
    from gguf import GGUFReader
    tensors=Counter(t.tensor_type.name for t in GGUFReader(str(args.projector)).tensors)
    quantized=any(name not in ('F16','F32','BF16','F64') for name in tensors)
    print(json.dumps(dict(file=args.projector.name,tensor_types=dict(tensors),quantized=quantized,
        note='Dequantizing an existing quantized projector does not restore lost precision.'),indent=2))
    if args.require_unquantized and quantized:raise SystemExit(1)


if __name__=='__main__':main()
