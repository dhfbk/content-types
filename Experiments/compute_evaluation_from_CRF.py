import sys
import os, os.path
import collections
from lxml import etree
import numpy
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



#print(classification_report(gold_emo_matrix_only, test_list_matrix_emo_class))


def get_data(testf):

    gold_final = []
    predicted_final = []

    gold_final_class = []
    predicted_final_class = []


    fileObject = open(testf)
    for line in fileObject:
        line_stripped = line.strip()
        line_splitted = line_stripped.split("\t")
        if len(line_splitted) > 1:
            gold_value = line_splitted[-2]
            predicted_value = line_splitted[-1]

            gold_final.append(line_splitted[0] + "\t" + gold_value)
            gold_final_class.append(gold_value)
            predicted_final.append(line_splitted[0] + "\t" + predicted_value)
            predicted_final_class.append(predicted_value)

    fileObject.close()


    print(classification_report(gold_final_class, predicted_final_class))


def main(argv=None):
    if argv is None:
        argv = sys.argv

    if len(argv) < 2:
        print('Usage: python compute_evaluation_CRF.py out_crf')
    else:
        get_data(argv[1])

if __name__ == '__main__':
    main()
