Dockerfile for the Feng 2014 RST Parser
=======================================

About the Dockerfile
--------------------

This Dockerfile provides a full environment for running the Feng 2014 RST Parser on bare text. It includes all dependencies and a fix for a bug in the crf wrapper blocking the parser pipeline in some Ubuntu environments (thanks Wladimir).

### Usage
<!--  todo falcon REST API  -->
```
docker build -t rstparser .
docker run -v /Path/to/text/sample/files:/samples -i -t rstparser
cd gCRF_dist/src
python parse.py /samples/F0001.txt
```


About the RST Parser
--------------------

See http://www.cs.toronto.edu/~weifeng/software.html

### Developer

* Vanessa Wei Feng
  Department of Computer Science
  University of Toronto
  Canada
  mailto: weifeng@cs.toronto.edu

### References

* Vanessa Wei Feng and Graeme Hirst, 2014. Two-pass Discourse Segmentation with Pairing and Global Features. arXiv:1407.8215v1. http://arxiv.org/abs/1407.8215
* Vanessa Wei Feng and Graeme Hirst, 2014. A Linear-Time Bottom-Up Discourse Parser with Constraints and Post-Editing. In Proceedings of the 52th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies (ACL-2014), Baltimore, USA. http://aclweb.org/anthology/P14-1048

### General information

* This RST-style discourse parser produces discourse tree structure on full-text level, given a raw text. No prior sentence splitting or any sort of preprocessing is expected. The program runs on Linux systems.
* The overall software work flow is similar to the one described in our paper (ACL 2014). However, to guarantee further efficiency, we remove the post-editing component from the workflow, and remove the set of entity-based transaction features from our feature set. Moreover, both structure and relation classification models are now implemented using CRFSuite.
