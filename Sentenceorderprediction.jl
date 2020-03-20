#under development

using Flux
using Flux: @functor

abstract type AbstractTransformer end

struct PwFFN
    din::Dense
    dout::Dense
end


struct Transformer <: AbstractTransformer
    mh::MultiheadAttention
    mhn::LayerNorm
    pw::PwFFN
    pwn::LayerNorm
    drop::Dropout
    proj::Dense
end

@functor Transformer


"""
    Transformer(size::Int, head::Int, ps::Int;
                future::Bool = true, act = relu, pdrop = 0.1)
    Transformer(size::Int, head::Int, hs::Int, ps::Int;
                future::Bool = true, act = relu, pdrop = 0.1)

Transformer layer.

`size` is the input size. if `hs` is not specify, use `div(size, head)` as the hidden size of multi-head attention. 
`ps` is the hidden size & `act` is the activation function of the positionwise feedforward layer. 
When `future` is `false`, the k-th token can't see the j-th tokens where j > k. `pdrop` is the dropout rate.
"""
function Transformer(size::Int, head::Int, ps::Int; future::Bool = true, act = relu, pdrop = 0.1)
    rem(size, head) != 0 && error("size not divisible by head")
    Transformer(size, head, div(size, head), ps;future=future, act=act, pdrop=pdrop)
end

Transformer(size::Int, head::Int, hs::Int, ps::Int; future::Bool = true, act = relu, pdrop = 0.1) = Transformer(
    MultiheadAttention(head, size, hs, size; future=future, pdrop=pdrop),
    LayerNorm(size),
    PwFFN(size, ps, act),
    LayerNorm(size),
    Dropout(pdrop),
    Dense(hs,hs)
)

function (t::Transformer)(x::AbstractArray{T, N}, mask=nothing) where {T, N}
    h=t.mh(x,x,x,mask=mask)
    h=t.pwn(h .+ t.proj(h,h))
    h=t.pwn(h .+ t.pw(h))
end

