use crate::tokenizer::Tokenizer;
use rand;
use tch;
use tch::IndexOp;
use tch::{Device, Kind, Tensor, nn::Module, nn::Path};
use tch::nn::{self, VarStore, OptimizerConfig};

pub struct Transformer {
    optimizer: nn::Optimizer,
    context_size: i64,
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

struct BatchPair {
    features : Vec<i64>,
    labels : Vec<i64>,
}

fn cross_entropy_for_logits(
    predictions: &Tensor,
    targets: &Tensor,
) -> Tensor {
    predictions
        .log_softmax(-1, Kind::Float)
        .nll_loss(targets)
}

impl Transformer {
    pub fn new(
        vs: &VarStore,
        device: Device,
        context_size: i64,
        tokenizer: Tokenizer,
        number_of_blocks: i64,
        heads: i64,
        dims: i64,
        dropout: f64,
    ) -> Self {
        let root = vs.root();
        let vocab = tokenizer.length;
        let learing_rate = 1e-3;
        let mut optimizer = nn::AdamW::default().build(&vs, learing_rate).unwrap();
        let blocks: Vec<TransformerBlock> = (0..number_of_blocks)
            .map(|n|TransformerBlock::new(&format!("mblock{n}"), &root, device, heads, dims, dropout))
            .collect();
        return Transformer {
            optimizer: optimizer,
            context_size: context_size,
            tokenizer: tokenizer,
            embedding: embedding("embedding", &root, vocab, dims),
            positional_encoding: PositionalEncoding::new(&root, device, dims, None),
            blocks: blocks,
            output_projection: linear("output", &root, dims, vocab),
            device: device,
            heads: Tensor::from(heads).to_device(device),
            dims: dims,
            dropout: dropout,
            training: true,
        };
    }

    pub fn predict(&self, batch: Vec<&str>) -> Tensor {
        let tokens: Tensor = self.tokenizer.encode(batch).to_device(self.device);
        self.forward(&tokens)
    }

    //[ batch [tokens] [tokens] [tokens] [tokens] ]
    pub fn forward(&self, tokens: &Tensor) -> Tensor {
        let embedding: Tensor = self.embedding.forward(&tokens);
        let mut out: Tensor = self.positional_encoding.forward(&embedding);

        for block in &self.blocks {
            out = block.forward(&out);
        }

        let out = self.output_projection.forward(&out);
        return out;
    }

    pub fn train(&mut self, data: &str, epochs: i64, batches: i64, batch_size: usize) {
        let tokens: Vec<i64> = self.tokenizer.encode_one(&data);
        let min_window: usize = 10;

        for epoch in 0..epochs {
            for batch in 0..batches {
                let window_varience: f64 = rand::random();
                let training_set: Vec<BatchPair> = (0..batch_size).map( |b| {
                    let start_varience: f64 = rand::random();
                    let window_size: usize = min_window + (window_varience * 500.0) as usize;
                    let window_start: usize = (((start_varience * tokens.len() as f64)) as i64 - window_size as i64 - 1).max(0).try_into().unwrap_or(0);

                    BatchPair {
                        features: tokens[window_start .. window_start + window_size].to_vec(),
                        labels: tokens[window_start + 1 .. window_start + window_size + 1].to_vec(),
                    }
                }).collect();

                let features: Vec<Vec<i64>> = training_set
                    .iter()
                    .map( |bp| bp.features.clone() )
                    .collect();

                let labels: Vec<Vec<i64>> = training_set
                    .iter()
                    .map( |bp| bp.labels.clone() )
                    .collect();

                let features_tensor = Tensor::from_slice2(&features).to_device(self.device);
                let labels_tensor = Tensor::from_slice2(&labels).to_device(self.device);

                self.optimizer.zero_grad();
                let out = self.forward(&features_tensor);
                //out.print();
                /*
                let (B, S, L) = out.size3().unwrap_or((1,1,1));
                let loss = cross_entropy_for_logits(
                    &out.view([B * S, L]),
                    &labels_tensor.view([B * S]),
                );
                loss.print();
                loss.backward(); // calculate gradients
                self.optimizer.step(); // optimze "learning"
                */
            }
        }

        // TODO Batching - Shuffle
        // TODO define optimizer AdamW
        // TODO Criterean cross_entropy_loss_with_logits
        // TODO write the training loop 
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

    fn forward(&self, inputs: &Tensor) -> Tensor {
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
            heads: Tensor::from(heads),
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
            linear1: linear("linear1", vs, dims, dims),
            linear2: linear("linear2", vs, dims, dims),
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let out = self.norm1.forward(input);
        let query = self.query_projection.forward(&out);
        let key = self.key_projection.forward(&out);
        let value = self.value_projection.forward(&out);
        let attention = self.attention(&query, &key, &value);
        return out;
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

    // TODO *** LEAK HERE!! maybe***
    // TODO *** LEAK HERE!! maybe***
    // TODO *** LEAK HERE!! maybe***
    // TODO *** LEAK HERE!! maybe***
    fn attention(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Tensor {
        //return Tensor::from_slice(&[1]);
        let (B, S, F) = query.size3().unwrap_or((1,1,1));
        let heads = i64::try_from(&self.heads).unwrap_or(1);
        let head_dims = F / heads;
        let multihead = [B, S, heads, head_dims];
        let query = query.view(multihead).transpose(1, 2);
        let key = key.view(multihead).transpose(1, 2);
        let value = value.view(multihead).transpose(1, 2);
        let out = query.matmul(&key.transpose(-2, -1));
        let out = out / (head_dims as f64).sqrt();
        let mask = self.causual_mask(S);
        let attn = out.masked_fill(&mask.eq(0.), f64::NEG_INFINITY);
        let attn = attn.softmax(-1, Kind::Float);
        let attn = attn.dropout(self.dropout, self.training);
        let attn = attn.matmul(&value)
            .transpose(1, 2)
            .contiguous()
            .reshape(&[B, S, F]);
        let out = self.attention_projection.forward(&attn).dropout(self.dropout, self.training);
        return out;
    }
}
