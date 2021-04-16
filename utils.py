import yaml

def load_yaml(load_path):
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)

    return loaded


def mkdir():
    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)