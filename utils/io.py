import pickle, os
import yaml

def mkdir(dir):
    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)
    return dir

def save_log(output_dir, logs, name='log'):
    with open(output_dir+'{}.txt'.format(name), 'w') as f:
        for line in logs:
            f.write(line)
            f.write('\n')

def dump_to_file(filename, content):
    with open(filename, 'wb') as handle:
        pickle.dump(content, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    handle.close()

def load_from_file(file):
    with open(file, 'rb') as handle:
        content = pickle.load(handle)

    handle.close()
    return content

def save_config(output_dir, args, name='config'):
    with open(output_dir+'/{}.yaml'.format(name), "w", encoding = "utf-8") as yaml_file:
        dump = yaml.dump(args, default_flow_style = False, allow_unicode = True, encoding = None)
        yaml_file.write(dump)