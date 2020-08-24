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

**Version-1 of ALBERT models**
- [Base](https://drive.google.com/drive/u/1/folders/1HHTlS_jBYRE4cG0elITEH7fAkiNmrEgz) from [[link](https://storage.googleapis.com/albert_models/albert_base_v1.tar.gz)]
- [Large](https://drive.google.com/drive/u/1/folders/1HHTlS_jBYRE4cG0elITEH7fAkiNmrEgz) from [[link](https://storage.googleapis.com/albert_models/albert_large_v1.tar.gz)]
- [Xlarge](https://drive.google.com/drive/u/1/folders/1HHTlS_jBYRE4cG0elITEH7fAkiNmrEgz) from [[link](https://storage.googleapis.com/albert_models/albert_xlarge_v1.tar.gz)]
- [Xxlarge](https://drive.google.com/drive/u/1/folders/1HHTlS_jBYRE4cG0elITEH7fAkiNmrEgz) from [[link](https://storage.googleapis.com/albert_models/albert_xxlarge_v1.tar.gz)]

**Version-2 of ALBERT models**
- [Base](https://drive.google.com/drive/u/1/folders/1DlX_WZacsjt6O8EDaawKJ-x4RWP46Xj-) 
- [Large](https://drive.google.com/drive/u/1/folders/1DlX_WZacsjt6O8EDaawKJ-x4RWP46Xj-) 
- [Xlarge](https://drive.google.com/drive/u/1/folders/1DlX_WZacsjt6O8EDaawKJ-x4RWP46Xj-)
- [Xxlarge](https://drive.google.com/drive/u/1/folders/1DlX_WZacsjt6O8EDaawKJ-x4RWP46Xj-) 

## Flie Structure
**src/albert.jl** - File contains wrapper for ALBERT transformer.It is implemented on top of Transformers.jl 

**src/alberttokenizer.jl** - File contains Albert tokenizer implemented on top of WordTokenizer to tokenize the word before feeding into wordpiece or sentence piece

**src/model.jl** - It contains model structure of original ALBERT model released by google-Research
**src/sentencepiece.jl** - Currently it contains Wordpiece model (directly taken from Transformers.jl) and planning to replace it with complete sentence piece model

**tfckpt2bsonforalbert.jl** - It is used to convert Tensorflow checkpoint file to Raw bson file 

## Status
The code is still underdevelopment 
## Checklist
- [x] file to convert Tensflowcheckpoint to bson
- [X] Space tokenizer (planning to update it as per need of sentencepiece)
- [X] SentencePiece (code can be founded in [wordtokenizer](https://github.com/JuliaText/WordTokenizers.jl))
- [X] wrapper for ALBERT Transformer 
- [X] Model file containing structure of ALBERT
   
most functions is under development and will be available soon 

sophisticated implemented and doc can be founded [here](https://github.com/JuliaText/TextAnalysis.jl/pull/203)
## Demo
[Demo](docs/Demo.ipynb) file contains the example of some of completed function.
