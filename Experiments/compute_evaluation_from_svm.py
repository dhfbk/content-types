import sys
import os, os.path
import collections
from lxml import etree
import numpy
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



def get_data(testg, predict):

    gold_final = []
    predicted_final = []

    fileObject = open(testg)
    for line in fileObject:
        line_stripped = line.strip()
        line_splitted = line_stripped.split(" ")
        if len(line_splitted) > 1:
            gold_value = line_splitted[0]

            gold_final.append(gold_value)

    fileObject.close()


    fileObject = open(predict)
    for line in fileObject:
        line_stripped = line.strip()

        predicted_final.append(line_stripped)

    fileObject.close()

    print(classification_report(gold_final, predicted_final))



def main(argv=None):
    if argv is None:
        argv = sys.argv

    if len(argv) < 3:
        print('Usage: python compute_evaluation_svm.py test.scale test.predict')
    else:
        get_data(argv[1], argv[2])

if __name__ == '__main__':
    main()
