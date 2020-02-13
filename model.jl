const config = Dict(
  "initializer_range"            => 0.02,
  "hidden_act"                   => gelu,
  "embedding"                    => 128,
  "num_hidden_layers"            => 12,
  "hidden_size"                  => 768,
  "max_position_embeddings"      => 512,
  "type_vocab_size"              => 2,
  "vocab_size"                   => 30522,
  "num_attention_heads"          => 12,
  "intermediate_size"            => 3072,
)

#create bert model like pretrain struct
#you can define the bert model in the way you like and wrap it with TransformerModel
function create_albert()
  global config
  albert = ALBert(
    config["hidden_size"],
    config["num_attention_heads"],
    config["intermediate_size"],
    config["num_hidden_layers"];
    act = config["hidden_act"],
    embedding =config["embeddings"]
  )

  tok_emb1 = Embed(
    config["embeddings"],
    config["vocab_size"]
  )
  tok_emb2 = Dense(
    config["hidden_size"],
    config["embeddings"]
  )

  seg_emb = Embed(
    config["hidden_size"],
    config["type_vocab_size"]
  )

  posi_emb = PositionEmbedding(
    config["hidden_size"],
    config["max_position_embeddings"];
    trainable = true
  )

  emb_post = Positionwise(
    LayerNorm(
      config["hidden_size"]
    )
  )

  pooler = Dense(
    config["hidden_size"],
    config["hidden_size"],
    tanh
  )

  masklm = (
    transform = Chain(
      Dense(
        config["hidden_size"],
        config["hidden_size"],
        config["hidden_act"]
      ),
      LayerNorm(
        config["hidden_size"]
      )
    ),
    output_bias = param(randn(
      Float32,
      config["vocab_size"]
    ))
  )

  nextsentence = Chain(
    Dense(
      config["hidden_size"],
      2
    ),
    logsoftmax
  )

  emb = CompositeEmbedding(tok = tok_emb, pe = posi_emb, segment = seg_emb, postprocessor = emb_post)


  clf = (pooler = pooler, masklm = masklm, nextsentence = nextsentence)

  TransformerModel(emb, albert, clf)
end
