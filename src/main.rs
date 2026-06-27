mod tokenizer;
use tokenizer::Tokenizer;

mod transformer;
use transformer::Transformer;

use anyhow::Result;
use std::fs;

use tch::{Device, Reduction, Tensor, nn::Module, nn::VarStore};

/*
TODO - ✅ pretty loss output
TODO - ✅ save load
TODO - ✅ pub fn chat()
TODO - ✅ All Stephen's voice transcripts- Train on this data
TODO - ✅ Batching - Shuffle
TODO - ✅ define optimizer AdamW
TODO - ✅ Criterean cross_entropy_loss_with_logits
TODO - ✅ write the training loop 
TODO - ✅ Load Data
TODO - ✅ Tokenizer Character level
TODO - ✅ Upgraded Tokenizer ( Word-level tokens )
TODO - ✅ Decide on Architecture Abstraction approach
TODO - ✅ Embedding
TODO - ✅ Attention
TODO - ✅ Positional Encodings
TODO - ✅ Multi-headed Attention
TODO - ✅ Transformer Head
TODO - ✅ Transformer Block
TODO - ✅ Model Network Projections
TODO - ✅ Multi-headed reshape 2x
TODO - ✅ FForward ( MLP )
TODO - ✅ 3x LayerNorms
TODO - ✅ Trianing
TODO - ✅ Batch and Shuffle
TODO - Use Tiktokenizer with gpt encoding - @amzadhossain
TODO -
*/

fn main() -> Result<()> {
    let device = Device::Mps;
    let mut vs = VarStore::new(device);
    let data: String = fs::read_to_string("data/stephen.txt")?;
    let test_string: &str = "Thank you Kyle ";
    let tokenizer = Tokenizer::new(&data);
    let mut transformer = Transformer::new(&vs, device, 5000, tokenizer, 4, 4, 128, 0.1);

    // Train the model on our data
    vs.load("model.ot")?;
    transformer.train(&data, 4, 10, 256, 10);
    vs.save("model.ot")?;

    transformer.chat("hello how are you today hope you are good and", 37);
    transformer.chat("yesterday we added the norms we should probably leverage these", 37);
    transformer.chat("excited about this This is really great Let", 37);

    // TODO Start the optimization loop here
    //let out: Tensor = transformer.forward([test_string].to_vec());
    //out.print();

    //let tokens: Tensor = tokenizer.encode(test_string);
    //let tokens = tokens.to_device(device);
    //tokens.print();
    //println!("{}", data.len());

    //let decoded = tokenizer.decode(tokens);
    //println!("'{decoded}'");

    //let test: Vec<String> = tokenizer::parser(test_string);
    //println!("{:?}", test);

    //println!("{:?}", data);
    Ok(())
}

// References
// Min-GPT https://github.com/LaurentMazare/tch-rs/blob/main/examples/min-gpt/main.rs
// Llama https://github.com/LaurentMazare/tch-rs/blob/main/examples/llama/main.rs
