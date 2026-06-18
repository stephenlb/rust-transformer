use crate::tokenizer::Tokenizer;
use tch;
use tch::{
    nn,
    Tensor,
    nn::Module,
    nn::Path,
    Kind,
    Device,
};

pub struct Transformer {
    tokenizer: Tokenizer,
    embedding: nn::Embedding,
    positional_encoding: nn::Embedding,
    query_projection: nn::Linear,
    key_projection: nn::Linear,
    value_projection: nn::Linear,
    //blocks: Vec<TransformerBlock>,
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

struct TransformerBlock {
    device: Device,
    heads: Tensor,
    dims: i64,
    dropout: f64,
    training: bool, // TODO <- implem,ent better fn training() / eval()
}

fn embedding(name: &str, vs: &Path, vocab: i64, dims: i64) -> nn::Embedding {
    tch::nn::embedding(vs / name, vocab, dims, Default::default())
}
fn linear(name: &str, vs: &Path, in_dims: i64, out_dims: i64) -> nn::Linear {
    tch::nn::linear(vs / name, in_dims, out_dims, Default::default())
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
        return Transformer {
            embedding: embedding("embeddign", vs, tokenizer.length, dims),
            positional_encoding: embedding("positional encoding", vs, dims, dims),
            tokenizer: tokenizer,
            query_projection: linear("query", vs, dims, dims),
            key_projection: linear("key", vs, dims, dims),
            value_projection: linear("value", vs, dims, dims),
            //blocks: Vec<TransformerBlock>,
            output_projection: linear("output", vs, dims, dims),

            device: device,
            heads: Tensor::from(heads),
            dims: dims,
            dropout: dropout,
            training: true,
        };
    }
}

impl TransformerBlock {
    pub fn new(
        vs: Path,
        device: Device,
        heads: i64,
        dims: i64,
        dropout: f64,
    ) -> Self {
        Self {
            device,
            heads: Tensor::from(1),
            dims,
            dropout,
            training: true,
        }
    }

    pub fn forward(&self) -> Tensor {
        Tensor::from_slice(&[1,2])
    }

    fn causual_mask(&self, size: i64) -> Tensor {
        Tensor::ones(&[size, size], (Kind::Float, self.device))
            .tril(0)
            .masked_fill(
                &Tensor::zeros(&[], (Kind::Float, self.device)),
                f64::NEG_INFINITY
            )
    }

    fn attention(
        &self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor {
        let out = query.matmul(&key.transpose(0, 1));
        let out = out / self.heads.sqrt();
        let mask = self.causual_mask(query.size()[1]);
        let out = out + mask;
        let out = out.softmax(-1, Kind::Float);
        let out = out.dropout(self.dropout, self.training);

        return out;
    }
}

