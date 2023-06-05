

def load_x_from_safetensor(checkpoint, key):
    x_generator = {}
    for k,v in checkpoint.items():
        if key in k:
            x_generator[k.replace(key+'.', '')] = v
    return x_generator