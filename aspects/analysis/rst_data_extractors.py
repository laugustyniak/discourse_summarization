from more_itertools import flatten
from pathlib import Path
from tqdm import tqdm

from utilities.transformations import load_serialized


def get_extracted_documents(data_path_extracted_documents):
    extracted_documents = Path(data_path_extracted_documents).glob('*')
    return {
        int(extracted_document.name): load_serialized(extracted_document.as_posix())
        for extracted_document
        in tqdm(extracted_documents)
    }


def get_document_based_on_aspect(docs_info, extracted_documents_data, aspect):
    return {
        doc_id: extracted_documents_data[doc_id]
        for doc_id, doc_info
        in tqdm(docs_info.items())
        if aspect in flatten(doc_info['aspects'].values())
    }


def get_docs_info_with_at_least_n_aspects_accepted(docs_info, n):
    return {
        k: v
        for k, v
        in docs_info.items()
        if len(v['accepted_edus']) >= n
    }
