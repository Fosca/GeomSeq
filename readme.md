# GeomSeq

This is the code repository for the following article: *Mental compression of spatial sequences in human working memory using numerical and geometrical primitives*. The repository contains the scripts and functions needed to reproduce most of the analyses presented in the article.
If you have any question, please send an email to *fosca.alroumi@gmail.com*. 


## Abstract of the paper

How does the human brain store sequences of spatial locations? We propose that each sequence is internally compressed using an abstract, language-like code that captures its numerical and geometrical regularities. We exposed participants to spatial sequences of fixed length but variable regularity while their brain activity was recorded using magneto-encephalography. Using multivariate decoders, each successive location could be decoded from brain signals, and upcoming locations were anticipated prior to their actual onset. Crucially, sequences with lower complexity, defined as the minimal description length provided by the formal language led to lower error rates and to increased anticipations. Furthermore, neural codes specific to the numerical and geometrical primitives of the postulated language could be detected, both in isolation and within the sequences. These results suggest that the human brain detects sequence regularities at multiple nested levels and uses them to compress long sequences in working memory.

## Installation

In order to get the toolbox up and running, run the following commands in the MATLAB command window:

```
>> !git clone https://github.com/fosca/geomseq.git
```

## Organization of the repository

We here assume that the raw data was preprocessed according to the stages described in the *STAR methods* of the article.
* **GeomSeq_analyses**: Series of scripts that can be run as a pipeline, that reproduce the analyses and figures of the paper.
* **GeomSeq_functions**: Contains the python functions needed to perform the analyses.

## Compatibility
The code is written for python 3.7. Compatibility with previous versions of python was not tested and is therefore not granted. 

 ## Related toolboxes
Most of the MEG analyses are based on MNE <https://mne.tools/stable/index.html>.
 Dror Dotan and I also developed a related toolbox for analyzing MEG data using RSA. It was used to generate the results corresponding to Figure S2.
 
  
