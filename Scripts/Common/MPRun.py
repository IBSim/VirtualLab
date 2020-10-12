import copy
import pickle

def main(Info, StudyDict, Function):
    OrigDict = copy.deepcopy(StudyDict)

    Function(Info, StudyDict)

    if not OrigDict == StudyDict:
        print('different')
        with open("{}/StudyDict.pickle".format(StudyDict["TMP_CALC_DIR"]),"wb") as handle:
        	pickle.dump(StudyDict, handle)
