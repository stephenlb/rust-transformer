use crate::tokenizer::Tokenizer;
use tch;
use tch::{Device, Kind, Tensor, nn, nn::Module, nn::Path};

pub struct Transformer {
    tokenizer: Tokenizer,
    embedding: nn::Embedding,
    positional_encoding: PositionalEncoding,
    blocks: Vec<TransformerBlock>,
    output_projection: nn::Linear,
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
        let blocks: Vec<TransformerBlock> = (0..3).map(|n|TransformerBlock::new(&format!("mblock{n}"), vs, device, 1, dims, dropout)).collect();
        return Transformer {
            tokenizer: tokenizer,
            embedding: embedding("embedding", vs, vocab, dims),
            positional_encoding: PositionalEncoding::new(vs, device, dims, None),
            blocks: blocks,
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

        let mut out: Tensor = self.positional_encoding.forward(embedding);
        //return self.blocks[0].forward(out);
        for block in &self.blocks {
            out = block.forward(out);
        }

        let out = self.output_projection.forward(&out);
        return out;
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
    attention_projection: nn::Linear,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    norm3: nn::LayerNorm,
    linear1: nn::Linear,
    linear2: nn::Linear,
}

impl TransformerBlock {
    pub fn new(name: &str, vs: &Path, device: Device, heads: i64, dims: i64, dropout: f64) -> Self {
        Self {
            device,
            heads: Tensor::from(4),
            dims,
            dropout,
            training: true,
            query_projection: linear("query", vs, dims, dims),
            key_projection: linear("key", vs, dims, dims),
            value_projection: linear("value", vs, dims, dims),
            attention_projection: linear("attention", vs, dims, dims),
            norm1: norm("norm1", vs, dims),
            norm2: norm("norm2", vs, dims),
            norm3: norm("norm3", vs, dims),
            linear1: linear("query", vs, dims, dims),
            linear2: linear("query", vs, dims, dims),
        }
    }

    pub fn forward(&self, input: Tensor) -> Tensor {
        let out = self.norm1.forward(&input);
        let query = self.query_projection.forward(&out);
        let key = self.key_projection.forward(&out);
        let value = self.value_projection.forward(&out);
        let attention = self.attention(query, key, value);
        //return attention;
        let attention = self.norm2.forward(&(input + attention));
        let out = self.linear1.forward(&attention).gelu("tanh");
        let out = self.linear2.forward(&out);
        // TODO review do we want to (inputs + out) instead?
        let out = self.norm3.forward(&(attention + out));

        // @SquirrelSniper138
        // Standard 2-Layer FFN Structure
        // let ffn1 = self.linear1.forward(&attention).gelu("tanh");
        // let ffn2 = self.linear2.forward

        return out;
    }

    fn causual_mask(&self, size: i64) -> Tensor {
        Tensor::ones(&[size, size], (Kind::Float, self.device)).tril(0)
    }

    // TODO ✅ multi-headed Attention
    // TODO ✅ LayerNorm
    fn attention(&self, query: Tensor, key: Tensor, value: Tensor) -> Tensor {
        let (B, S, F) = query.size3().unwrap_or((1,1,1));
        let heads = i64::try_from(&self.heads).unwrap_or(1);
        let head_dims = F / heads;
        let multihead = [B, S, heads, head_dims];
        let query = query.view(multihead).transpose(1, 2);
        let key = key.view(multihead).transpose(1, 2);
        let value = value.view(multihead).transpose(1, 2);
        let out = query.matmul(&key.transpose(-2, -1));
        let out = out / self.heads.sqrt();
        let mask = self.causual_mask(S);
        let attn = out.masked_fill(&mask.eq(0.), f64::NEG_INFINITY);
        let attn = attn.softmax(-1, Kind::Float);
        let attn = attn.dropout(self.dropout, self.training);
        let attn = attn.matmul(&value)
            // @SquirrelSniper138 thank you! ❤️
            // Fix: Remove the inner transpose on value
            // exclude inner transpose for value
            //.transpose(1, 2)
            //.contiguous()
            .reshape(&[B, S, F]);
        let out = self.attention_projection.forward(&attn);
        return out;
    }
}
