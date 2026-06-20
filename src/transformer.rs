use crate::tokenizer::Tokenizer;
use tch;
use tch::{Device, Kind, Tensor, nn, nn::Module, nn::Path};

pub struct Transformer {
    tokenizer: Tokenizer,
    embedding: nn::Embedding,
    positional_encoding: PositionalEncoding,
    //blocks: Vec<TransformerBlock>,
    block: TransformerBlock,
    output_projection: nn::Linear,
    // RMS Norm x = x/(x*x+e)
    // norm1: nn::LayerNorm,
    // norm2: nn::LayerNorm,
    // norm3: nn::LayerNorm,
    device: Device,
    heads: Tensor,
    dims: i64,
    dropout: f64,
    training: bool,
}

fn embedding(name: &str, vs: &Path, vocab: i64, dims: i64) -> nn::Embedding {
    tch::nn::embedding(vs / name, vocab, dims, Default::default())
}
fn linear(name: &str, vs: &Path, in_dims: i64, out_dims: i64) -> nn::Linear {
    tch::nn::linear(vs / name, in_dims, out_dims, Default::default())
}
fn norm(name: &str, vs: &Path, dims: i64) -> nn::LayerNorm {
    tch::nn::layer_norm(vs / name, vec![dims], Default::default())
}

impl Transformer {
    pub fn new(
        vs: &Path,
        device: Device,
        tokenizer: Tokenizer,
        heads: i64,
        dims: i64,
        dropout: f64,
    ) -> Self {
        let vocab = tokenizer.length;
        // TODO Bblock Vec<>
        return Transformer {
            tokenizer: tokenizer,
            embedding: embedding("embedding", vs, vocab, dims),
            positional_encoding: PositionalEncoding::new(vs, device, dims, None),
            // TODO
            // TODO
            // TODO add LayerNorm before projection
            // TODO
            // TODO
            block: TransformerBlock::new("block1", vs, device, 1, dims, dropout),
            // TODO add LayerNorms
            //blocks: Vec<TransformerBlock>,
            // TODO add LayerNorms
            //out = self.norm2(inputs + attn)
            //out = self.norm3(out + self.feedforward(out))
            output_projection: linear("output", vs, dims, dims),
            device: device,
            heads: Tensor::from(heads),
            dims: dims,
            dropout: dropout,
            training: true,
        };
    }

    pub fn forward(&self, batch: Vec<&str>) -> Tensor {
        let tokens: Tensor = self.tokenizer.encode(batch);
        let embedding: Tensor = self.embedding.forward(&tokens);
        let positions: Tensor = self.positional_encoding.forward(embedding);
        let attention = self.block.forward(positions);

        return attention;
    }
}

struct PositionalEncoding {
    embedding: nn::Embedding,
    device: Device,
}

impl PositionalEncoding {
    fn new(vs: &Path, device: Device, dims: i64, max_tokens: Option<i64>) -> Self {
        let max_tokens: i64 = max_tokens.unwrap_or(5000);
        return Self {
            device: device,
            embedding: embedding("positions", vs, max_tokens, dims),
        }
    }

    fn forward(&self, inputs: Tensor) -> Tensor {
        let size = inputs.size();
        let options = (Kind::Int64, self.device);
        let range: Tensor = Tensor::arange(size[1], options);
        let positions = self.embedding.forward(&range);//.unsqueeze(0);

        // TODO add positions
        return inputs + positions;
    }
}

struct TransformerBlock {
    device: Device,
    heads: Tensor,
    dims: i64,
    dropout: f64,
    training: bool, // TODO <- implem,ent better fn training() / eval()
    query_projection: nn::Linear,
    key_projection: nn::Linear,
    value_projection: nn::Linear,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    norm3: nn::LayerNorm,
}

impl TransformerBlock {
    pub fn new(name: &str, vs: &Path, device: Device, heads: i64, dims: i64, dropout: f64) -> Self {
        Self {
            device,
            heads: Tensor::from(1),
            dims,
            dropout,
            training: true,
            query_projection: linear("query", vs, dims, dims),
            key_projection: linear("key", vs, dims, dims),
            value_projection: linear("value", vs, dims, dims),
            norm1: norm("norm1", vs, dims),
            norm2: norm("norm2", vs, dims),
            norm3: norm("norm3", vs, dims),
        }
    }

/*
    def forward(self, inputs):
        out = self.norm1(inputs)
        out = self.qkv_projection(out) ### <-- Cache is here
        query, key, value = out.chunk(3, dim=-1)
        ## can cache all three? QK and V?
        ## @computer_vision said ONLY K and V
        attn = self.attention(query, key, value) 
        out = self.norm2(inputs + attn)
        out = self.norm3(out + self.feedforward(out))
        return out

*/
    pub fn forward(&self, input: Tensor) -> Tensor {
        // TODO LayerNorm
        let out = self.norm1.forward(&input);
        let query = self.query_projection.forward(&out);
        let key = self.key_projection.forward(&out);
        let value = self.value_projection.forward(&out);
        let attention = self.attention(query, key, value);
        let attention = self.norm2.forward(&(input + attention));
        // TODO FEED FORWARD ( 
        return attention;
        // TODO LayerNorm
        // TODO LayerNorm
    }

    fn causual_mask(&self, size: i64) -> Tensor {
        Tensor::ones(&[size, size], (Kind::Float, self.device)).tril(0)
    }

    // TODO multi-headed Attention
    // TODO LayerNorm
    fn attention(&self, query: Tensor, key: Tensor, value: Tensor) -> Tensor {
        let (B, S, F) = query.size3().unwrap_or((1,1,1));
        println!("Shape of query: {B} {S} {F}");
        let out = query.matmul(&key.transpose(1, 2));
        let out = out / self.heads.sqrt();
        let mask = self.causual_mask(S);
        let attn = out.masked_fill(&mask.eq(0.), f64::NEG_INFINITY);
        let attn = attn.softmax(-1, Kind::Float);
        let attn = attn.dropout(self.dropout, self.training);
        let attn = attn.matmul(&value);

        return attn;
    }
}
