import os
import hickle
import pickle
import logging
'''
logging.basicConfig(filename='logging/debug.log', format='%(asctime)s %()-4s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)
'''

def check_cache():
    try:
        os.stat('cache')
    except:
        os.mkdir('cache')

def save_hickle_file(filename, data):
    check_cache()
    filename = filename + '.hickle'

    with open(filename, 'w') as f:
        hickle.dump(data, f, mode='w')

def load_hickle_file(filename):
    check_cache()
    filename = filename + '.hickle'
    if os.path.isfile(filename):
        data = hickle.load(filename)
        return data

    logging.debug('Filename: <' + filename + '> does not exist.')
    return None
