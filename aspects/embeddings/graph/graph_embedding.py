"""Handles base utilities for Node Embedding purposes."""

import pickle

import gensim
import numpy as np


class KeyedModel:
    """Provides utilities for embedding dictionary."""

    def __init__(self, size, node_emb_vectors=None, fill_unknown_nodes=True):
        """Inits KeyedModel.

        :param size: Embedding vectors size
        :type size: int
        :param node_emb_vectors: Dictionary containing embedding vectors
        :type node_emb_vectors: dict
        :param fill_unknown_nodes: Whether to fill unknown nodes
        :type fill_unknown_nodes: bool
        """
        self._emb_dim = size
        self._node_emb_vectors = node_emb_vectors or {}
        self.fill_unknown_nodes = fill_unknown_nodes

    def add_node(self, node, embedding_vector):
        """Adds node representation to the class.

        :param node: Input node
        :type node: str
        :param embedding_vector: Node vector
        :type embedding_vector: list
        """
        assert len(embedding_vector) == self._emb_dim
        self._node_emb_vectors[node] = (np.array(embedding_vector)
                                        .astype('float32'))

    def get_vector(self, node):
        """Returns vector representation for a given node.

        If the vector representation is not found,
        returns array of ones of embedding vector size.

        :param node: Input node
        :type node: str
        :return: Vector representation
        :rtype: np.ndarray
        :raise ValueError: Whether given node is not found in embedding matrix
        """
        if node in self._node_emb_vectors.keys():
            return self._node_emb_vectors[node]

        if self.fill_unknown_nodes:
            return np.ones(self._emb_dim)

        raise ValueError(f'Node {node} not found in embedding matrix!')

    def get_subset(self, nodes):
        """Returns subset of KeyedModel for a given nodes.

        :param nodes: List of nodes
        :rtype: list
        :return: Dictionary containing representation of a given subset
        :rtype: dict
        """
        subset = dict()
        for node in nodes:
            subset[node] = self.get_vector(node)
        return subset

    @classmethod
    def from_file(cls, filepath):
        """Load KeyedModel from given path.

        :param filepath: Path of the file
        :type: filepath: str
        :return: Loaded KeyedModel
        :rtype: KeyedModel
        """
        with open(filepath, 'rb') as f:
            emb_dict = pickle.load(f)

        return KeyedModel(
            node_emb_vectors=emb_dict.get('vectors'),
            size=emb_dict.get('size')
        )

    @classmethod
    def from_gensim_w2v_file(cls, filepath):
        """Load gensim word2vec file from given path.

        :param filepath: Path of the file
        :type: filepath: str
        :return: Loaded KeyedModel
        :rtype: KeyedModel
        """
        w2v_emb = gensim.models.KeyedVectors.load_word2vec_format(filepath)
        return cls.from_gensim_w2v_format(w2v_embedding=w2v_emb)

    @classmethod
    def from_gensim_w2v_format(cls, w2v_embedding, fill_unknown_nodes=True):
        """Convert Word2Vec format to our format."""
        node_emb_vecs = {}
        for idx, node in enumerate(w2v_embedding.index2word):
            node_emb_vecs[node] = w2v_embedding.vectors[idx]

        return KeyedModel(
            size=w2v_embedding.vector_size,
            node_emb_vectors=node_emb_vecs,
            fill_unknown_nodes=fill_unknown_nodes
        )

    def to_pickle(self, filepath):
        """Save current embedding to file.

        :param filepath: Path of output file
        :type filepath: str
        """
        with open(filepath, 'wb') as f:
            pickle.dump(
                obj={'vectors': self._node_emb_vectors,
                     'size': self._emb_dim},
                file=f
            )

    def to_dict(self):
        """Returns KeyedModel items.

        :return: Dict of representations
        :rtype: dict[str, np.array]
        """
        return self._node_emb_vectors

    def to_numpy(self, nodes, dtype=np.float32):
        """Converts KeyedModel to numpy array for given list of nodes.

        :param nodes: List of nodes
        :type nodes: list
        :param dtype: Datatype of ndarray
        :return: Numpy array of node vector representations
        :rtype: np.ndarray
        """
        arr = []
        for node in nodes:
            arr.append(self.get_vector(str(node)))
        return np.array(arr, dtype=dtype)

    def items(self):
        """Returns items of KeyedModel.

        :return Items of KeyedModel
        """
        return self._node_emb_vectors.items()

    @property
    def nodes(self):
        """Gets list of nodes.

        :return: List of nodes
        :rtype: list
        """
        return list(self._node_emb_vectors.keys())

    @property
    def shape(self):
        """Gets shape of KeyedModel in form (number_of_nodes, vector_size)."""
        return len(self.nodes), self._emb_dim

    @property
    def emb_dim(self):
        """Gets KeyedModel dimensions."""
        return self._emb_dim

    def __repr__(self):
        """Creates string summary of class object."""
        return 'Embedding(shape={})'.format(self.shape)


class KeyedListModel:
    """Wrapper for handling sequence of KeyedModels."""

    def __init__(self, km_seq):
        """Inits KeyedListModel class.

        :param km_seq: Input sequence of KeyedModels
        :type km_seq: list
        """
        self._km_seq = None
        self._init_seq(km_seq)

    def _init_seq(self, km_seq):
        """Inits model sequence.

        :param km_seq: Input sequence of KeyedModels
        :type km_seq: list
        :raise TypeError: Whether not every element in the input sequence
                          is tgem.embedding.KeyedModel
        """
        if not all(isinstance(e, KeyedModel) for e in km_seq):
            raise TypeError('Only tgem.embedding.KeyedModel is '
                            'supported as an Embeddding format! ')
        self._km_seq = km_seq

    @classmethod
    def from_gensim_emb_list(cls, w2v_gensim_seq):
        """Converts gensim word2vec models sequence into KeyedListModel.

        :param w2v_gensim_seq: Input word2vec gensim sequence
        :type w2v_gensim_seq: list
        :return: Loaded KeyedListModel
        :rtype: KeyedListModel
        """
        return KeyedListModel(
            km_seq=[KeyedModel.from_gensim_w2v_format(e)
                    for e in w2v_gensim_seq]
        )

    @property
    def shapes(self):
        """Returns shapes of KeyedListModel elements.

        :return: Sequence with sequence elements shapes
        :rtype: list
        """
        return [e.shape for e in self._km_seq]

    @property
    def length(self):
        """Returns length of KeyedListModel sequence.

        :return: length of KeyedListModel sequence
        :rtype: int
        """
        return len(self._km_seq)

    def get_emb(self, emb_id):
        """Gets element of KeyedListModle with given id.

        :return: A KeyedModel
        :rtype: KeyedModel
        """
        return self._km_seq[emb_id]

    def to_dict(self, nodes):
        """Converts KeyedListModel to a Dictionary.

        Nodes are keys and values are vector representations sequences
        of a given node.

        :param nodes: List of nodes
        :type nodes: list
        :return A dictionary form of KeyedListModel
        :rtype: dict
        """
        emb_dict = dict()
        for node in nodes:
            node_vectors = []
            for e in self._km_seq:
                node_vectors.append(e.get_vector(str(node)))

            emb_dict[node] = np.array(node_vectors)

        return emb_dict

    def to_aligned_array(self, nodes):
        """Converts KeyedListModel to a aligned ndarray.

        In a rows are node vector representations.
        The order from input list is preserved.

        :param nodes: List of nodes
        :type nodes: list
        :return A aligned ndarray form of KeyedListModel
        :rtype: np.ndarray
        """
        aligned_emb_arr = []
        for emb_id in range(self.length):
            e = self.get_emb(emb_id)
            aligned_emb_arr.append(
                e.to_numpy(nodes)
            )
        return aligned_emb_arr
