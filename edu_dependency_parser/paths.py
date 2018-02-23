import os.path

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
RHETORICAL_INSTLALATION_PATH = os.path.join(ROOT_PATH, 'rhetorical-installation', 'gCRF_dist')

PARSED_TEXTS_PATH = os.path.join(ROOT_PATH, 'texts/parsed_texts/')
PENN2MALT_PATH = os.path.join(ROOT_PATH, 'tools/Penn2Malt/')
SVM_TOOLS = os.path.join(ROOT_PATH, 'tools/svm_tools/')
CRFSUITE_PATH = os.path.join(RHETORICAL_INSTLALATION_PATH, 'tools/crfsuite/')
MALLET_PATH = os.path.join(ROOT_PATH, 'tools/mallet-2.07/')
STANFORD_PARSER_PATH = os.path.join(ROOT_PATH, 'tools/stanford-parser-full-2014-01-04/')

MODEL_PATH = os.path.join(ROOT_PATH, 'model/')
TREE_BUILD_MODEL_PATH = os.path.join(MODEL_PATH, 'tree_build_set_CRF/')
SEGMENTER_MODEL_PATH = os.path.join(MODEL_PATH, 'seg_set_CRF/')
CHARNIAK_PARSER_MODEL_PATH = os.path.join(MODEL_PATH, 'WSJ/')
SBD_MODEL_PATH = os.path.join(MODEL_PATH, 'sbd_models/model_nb/')

tmp_folder = os.path.join(ROOT_PATH, 'tmp/')

save_folder = os.path.join(MODEL_PATH, 'serial_data/')

RST_DT_ROOT = os.path.join(ROOT_PATH, 'texts/RST_DT_fixed/')
DECOREF_PATH = os.path.join(ROOT_PATH, 'texts/dcoref/')

LOGS_PATH = os.path.join(ROOT_PATH, 'logs/')
