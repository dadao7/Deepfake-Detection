import torch, sys
sys.path.insert(0, r'C:\m2f2\M2F2_Det')
sys.path.insert(0, r'C:\m2f2\code')

ckpt = torch.load(r'C:\m2f2\weights\current_model_180.pth',
                  map_location='cpu', weights_only=False)

print('타입:', type(ckpt))
if isinstance(ckpt, dict):
    print('최상위 키:', list(ckpt.keys()))
    for k, v in ckpt.items():
        print(f'  {k}: {type(v).__name__}', end='')
        if hasattr(v, 'shape'):
            print(f' shape={v.shape}')
        elif isinstance(v, dict):
            print(f' (dict, keys={list(v.keys())[:5]})')
        else:
            print(f' = {v}')