import pickle, os
import yaml

def mkdir(dir):
    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)
    return dir

def save_log(output_dir, logs):
    with open(output_dir+'log.txt', 'w') as f:
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