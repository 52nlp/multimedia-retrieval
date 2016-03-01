# multimedia-retrieval

Contains the implementation of a multimedia retrieval framework developed for the MULTISENSOR project. The MULTISENSOR multimedia retrieval framework:
 
 - fuses multiple modalities, so as to retrieve multimedia objects in response to a multimodal query;
 - integrates high-level information, i.e. multimedia objects are enriched with high-level textual and visual concepts;
 - is language-independent.

#Description

The framework leverages 3 modalities from every multimedia object, namely visual features, visual concepts and textual concepts. Each modality provides a vector representation of the multimedia object through its corresponding features. In the framework, the similarity matrices from the 3 modalities are constructed and are fused for the computation of one relevance score vector by means of the Weighted Multimodal Contextual Similarity Matrix (W-MCSM) model.

# Version
1.0.0


 