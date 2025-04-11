# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
from os.path import join

import subprocess
from urllib.request import Request, urlopen

__author__ = 'Fisher Yu'
__email__ = 'fy@cs.princeton.edu'
__license__ = 'MIT'


def list_categories():
    url = 'http://dl.yf.io/lsun/categories.txt'
    with urlopen(Request(url)) as response:
        return response.read().decode().strip().split('\n')


def download(out_dir, category, set_name):
    url = 'http://dl.yf.io/lsun/scenes/{category}_' \
          '{set_name}_lmdb.zip'.format(**locals())
    if set_name == 'test':
        out_name = 'test_lmdb.zip'
        url = 'http://dl.yf.io/lsun/scenes/{set_name}_lmdb.zip'
    else:
        out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
    out_path = join(out_dir, out_name)
    cmd = ['curl', '-C', '-', url, '-o', out_path]
    print('Downloading', category, set_name, 'set')
    subprocess.call(cmd)
    return out_path

def unzip(filename, extract_to):
    import zipfile

    if zipfile.is_zipfile(filename):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            
            try: 
                filename.unlink()
                print(f"Deleted: {filename}")
            except:
                print("File not found.")
    else:
        print("This is not a valid zip file.")
             
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir', default='')
    parser.add_argument('-c', '--category', default=None)
    parser.add_argument('-s', '--set', default="val")
    args = parser.parse_args()

    categories = list_categories()
    if args.category is None:
        print('Downloading', len(categories), 'categories')
        for category in categories:
            out_path = download(args.out_dir, category, args.set)
            unzip(out_path, join(args.out_dir, args.set))
            # download(args.out_dir, category, 'val')
            
        out_path = download(args.out_dir, '', 'test')
        unzip(out_path, join(args.out_dir, "test_data"))
     
    else:
        if args.category == 'test':
            out_path = download(args.out_dir, '', f"{args.set}")
            unzip(out_path, join(args.out_dir, f"{args.set}_data"))
            # download(args.out_dir, '', 'test')
        elif args.category not in categories:
            print('Error:', args.category, "doesn't exist in", 'LSUN release')
        else:
            out_path = download(args.out_dir, args.category, args.set)
            unzip(out_path, join(args.out_dir, f"{args.set}_data"))
            
            # download(args.out_dir, args.category, 'train')
            # download(args.out_dir, args.category, 'val')

if __name__ == '__main__':
    main()