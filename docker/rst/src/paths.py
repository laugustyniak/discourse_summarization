from pathlib import Path

ROOT_PATH = Path(__file__).absolute().parent

PARSED_TEXTS_PATH = ROOT_PATH / 'texts' / 'parsed_texts'
STANFORD_PATH = ROOT_PATH / 'tools' / 'stanford_parser'
PENN2MALT_PATH = ROOT_PATH / 'tools' / 'Penn2Malt'
SVM_TOOLS = ROOT_PATH / 'tools' / 'svm_tools'
CRFSUITE_PATH = (ROOT_PATH / 'tools' / 'crfsuite').as_posix()
MALLET_PATH = ROOT_PATH / 'tools' / 'mallet-2.07'
STANFORD_CORENLP_PATH = ROOT_PATH / 'tools' / 'stanford-corenlp-full-2013-11-12'
SSPLITTER_PATH = (ROOT_PATH / 'tools' / 'CCGSsplitter').as_posix()
STANFORD_PARSER_PATH = ROOT_PATH / 'tools' / 'stanford-parser-full-2014-01-04'

MODEL_PATH = ROOT_PATH / 'model'
TREE_BUILD_MODEL_PATH = (MODEL_PATH / 'tree_build_set_CRF').as_posix()
SEGMENTER_MODEL_PATH = (MODEL_PATH / 'seg_set_CRF').as_posix()
CHARNIAK_PARSER_MODEL_PATH = MODEL_PATH / 'WSJ'
SBD_MODEL_PATH = MODEL_PATH / 'sbd_models' / 'model_nb'

tmp_folder = (ROOT_PATH / 'tmp').as_posix()

save_folder = (MODEL_PATH / 'serial_data').as_posix()

RST_DT_ROOT = ROOT_PATH / 'texts' / 'RST_DT_fixed'
DECOREF_PATH = ROOT_PATH / 'texts' / 'dcoref'

LOGS_PATH = ROOT_PATH / 'logs'
