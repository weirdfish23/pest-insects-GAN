import gdown
import zipfile
import os
import shutil

from utils.read import read_config

def download_data():
    config = read_config()

    dest_dir = config['base_data']['dest_dir']
    src_filename = config['base_data']['src_filename']

    if src_filename in os.listdir(dest_dir):
        os.remove(os.path.join(dest_dir, src_filename))

    gdown.download(config['base_data']['url'], os.path.join(dest_dir, src_filename, ), quiet=False)

    if 'base' in os.listdir(dest_dir):
        shutil.rmtree(os.path.join(dest_dir, 'base'))
    
    with zipfile.ZipFile(os.path.join(dest_dir, src_filename),"r") as zip_ref:
        zip_ref.extractall(os.path.join(dest_dir, 'base'))

    os.remove(os.path.join(dest_dir, src_filename))


if __name__ == '__main__':
    download_data()

