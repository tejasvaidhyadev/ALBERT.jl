# ALBERT.jl
The Repo contains implementation of ALBERT in julia


Simply implementation of [ALBERT(A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS)](https://arxiv.org/pdf/1909.11942.pdf). This implementation is based on [Transformers.jl](https://github.com/chengchingwen/Transformers.jl/blob/master/example/BERT/pretrain.jl)

## KEYWORD IN ALBERT 


1. SOP(sentence-order prediction) loss : In Original BERT, creating is-not-next(negative) two sentences with randomly picking, however ALBERT use negative examples the same two consecutive segments but with their order swapped.
2. Cross-Layer Parameter Sharing : ALBERT use cross-layer parameter sharing in Attention and FFN(FeedForward Network) to reduce number of parameter.
3. ALBERT seperated Embedding matrix(VxD) to VxE and ExD.

 
Pre-trained BSON
==================
Pre-trained tensorflow checkpoint file by [google-research](https://github.com/google-research/ALBERT) to the Julia desired pre-trained model format(i.e. BSON) :

Version-1 of ALBERT models
- [Base](https://drive.google.com/drive/u/1/folders/1HHTlS_jBYRE4cG0elITEH7fAkiNmrEgz) from [[link](https://storage.googleapis.com/albert_models/albert_base_v1.tar.gz))]
- [Large](https://drive.google.com/drive/u/1/folders/1HHTlS_jBYRE4cG0elITEH7fAkiNmrEgz) from [[link](https://storage.googleapis.com/albert_models/albert_large_v1.tar.gz)]
- [Xlarge](https://drive.google.com/drive/u/1/folders/1HHTlS_jBYRE4cG0elITEH7fAkiNmrEgz) from [[link](https://storage.googleapis.com/albert_models/albert_xlarge_v1.tar.gz)]
- [Xxlarge](https://drive.google.com/drive/u/1/folders/1HHTlS_jBYRE4cG0elITEH7fAkiNmrEgz) from [[link](https://storage.googleapis.com/albert_models/albert_xxlarge_v1.tar.gz)]
