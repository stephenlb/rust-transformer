use crate::tokenizer::Tokenizer;
use tch;
use tch::{
    Tensor,
    nn::Module,
    nn::Path,
    Kind,
    Device,
};

pub struct Transformer {
    device: Device,
    heads: Tensor,
    tokenizer: Tokenizer,
    blocks: Vec<TransformerBlock>,
    dims: i64,
    dropout: f64,
    training: bool,
}

pub struct TransformerBlock {
    device: Device,
    heads: Tensor,
    dims: i64,
    dropout: f64,
    training: bool, // TODO <- implem,ent better fn training() / eval()
}

fn embedding(vs: &Path, vocab: i64, dims: i64) -> impl Module {
    tch::nn::embedding(vs, vocab, dims, Default::default())
}

impl TransformerBlock {
    pub fn new(
        vs: Path,
        device: Device,
        heads: Tensor,
        dims: i64,
        dropout: f64,
    ) -> Self {
        Self { device, heads, dims, dropout, training: true }
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
        let out = out.dropout(0.1, self.training);

        return out;
    }
}

