import os
import hickle
import pickle
import logging
'''
logging.basicConfig(filename='logging/debug.log', format='%(asctime)s %()-4s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)
'''
def save_hickle_file(filename, data):
    filename = filename + '.hickle'

    with open(filename, 'w') as f:
        hickle.dump(data, f, mode='w')

def load_hickle_file(filename):
    filename = filename + '.hickle'
    if os.path.isfile(filename):
        data = hickle.load(filename)
        return data

    logging.debug('Filename: <' + filename + '> does not exist.')
    return None
