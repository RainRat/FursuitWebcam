import cv2
import csv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import numpy as np
import os
import argparse
from tqdm import tqdm
import logging

INPUT_SIZE = (224, 224)

def getAllFilesRecursive(root):
    return [os.path.join(dp, f) for dp, dn, filenames in os.walk(root) for f in filenames]

def preprocess_image(filename):
    frame = cv2.imread(filename)
    frame = cv2.resize(frame, INPUT_SIZE)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    test_image = np.expand_dims(image.img_to_array(frame), axis=0)
    mobilenet_test_image = preprocess_input(np.copy(test_image) * 255.0)
    return test_image, mobilenet_test_image


def process_image_and_write_results(writer, fn, model_custom_name, model_custom_count, custom_names, custom_counts):
    '''This sub will just refer to the model as "custom" so it can be repurposed.'''
    try:
        test_image, mobilenet_test_image = preprocess_image(fn)
        custom_name_result = model_custom_name.predict(mobilenet_test_image, batch_size=1)[0]
        maxidxname = np.argmax(custom_name_result)
        maxconname = custom_name_result[maxidxname]

        custom_count_result = model_custom_count.predict(test_image, batch_size=1)[0]
        maxidx = np.argmax(custom_count_result)
        maxcon = custom_count_result[maxidx]

        writer.writerow([fn, custom_counts[maxidx], maxcon, custom_names[maxidxname][1], maxconname])
    except Exception as e:
        logging.exception(f"Exception occurred while processing file {fn}")

list_classes = ["file", "numfursuit", "numfursuitconfidence", "fursuitname", "fursuitnameconfidence"]
fursuitcountnames=[0, 1, 2, 3]

parser = argparse.ArgumentParser()
parser.add_argument('targdir', help='Directory to search (input)')
parser.add_argument('targcsv', help='Output csv file (output)')
parser.add_argument('--log-level', help='Set the logging level (debug, info, warning, error)', default='info')
args = parser.parse_args()

# Set up logging based on the command-line argument
logging_level = getattr(logging, args.log_level.upper(), None)
if not isinstance(logging_level, int):
    raise ValueError('Invalid log level: %s' % args.log_level)
logging.basicConfig(filename='processing.log', level=logging_level, format='%(asctime)s:%(levelname)s:%(message)s')

if __name__ == "__main__":

    # Open the CSV file using the with statement
    with open(args.targcsv, 'w', newline='') as outputcsvfile:
        writer = csv.writer(outputcsvfile)
        writer.writerow(list_classes)

        # Load models outside of the loop
        model_fursuit_name = load_model('fursuitname.h5')
        model_fursuit_count = load_model('fursuitcount.h5')

        # Load fursuit names
        with open('fursuitlookup.csv', 'r') as lookupfile:
            csv_reader = csv.reader(lookupfile)
            fursuitnames = list(csv_reader)

        # Get all files
        fndict = getAllFilesRecursive(args.targdir)

        # Process each file and write results
        for fn in tqdm(fndict):
            logging.info(fn)
            process_image_and_write_results(writer, fn, model_fursuit_name, model_fursuit_count, fursuitnames, fursuitcountnames)
