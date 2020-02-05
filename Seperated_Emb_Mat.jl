using Tranformers.Basic.Embed
using Transformers

abstract type abstractSEM end
struct SEM <: abstractSEM
    tok_embed1::Embed
    tok_embed2::Dense
    pos_embed::PositionEmbedding
    seg_embed::Embed
end

@functor SEM


"just a wrapper for two dense layer."
SEM(vocab_size::Int, embed::Int,hs::Int, seg::Int,maxlen::int) = SEM(
    Embed(vocab_size,embed),
    Dense(embed, hs),
    PositionEmbedding(hs,maxlen)
    Embed(seg,hidden)
    
)

function (s::SEM)(x::AbstractArray{T, N},seg::AbstractArray{T, N}) where {T, N}
    # size(x) == (dims, seq_len)
    e =s.tok_embed1(x)
    e=s.tok_embed2(e)
    e=e .+ s.postionEmbedding(x) .+ s.seg_embed(seg)
    
end

