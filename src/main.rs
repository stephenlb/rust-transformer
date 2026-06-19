mod tokenizer;
use tokenizer::Tokenizer;

mod transformer;
use transformer::Transformer;

use anyhow::Result;
use std::fs;

use tch::{Device, Reduction, Tensor, nn::Module, nn::VarStore};

/*
TODO - ✅ Load Data
TODO - ✅ Tokenizer Character level
TODO - ✅ Upgraded Tokenizer ( Word-level tokens )
TODO - ✅ Decide on Architecture Abstraction approach
TODO - ✅ Embedding
TODO - ✅ Attention
TODO - Use Tiktokenizer with gpt encoding - @amzadhossain
TODO - Positional Encodings
TODO - Multi-headed Attention
TODO - Transformer Head
TODO - Transformer Block
TODO - Model Network Projections
TODO - Multi-headed reshape 2x
TODO - FForward ( MLP )
TODO - 3x LayerNorms
TODO - Trianing
TODO - Batch and Shuffle
TODO -
*/

fn main() -> Result<()> {
    let device = Device::Cpu;
    let vs = VarStore::new(device);
    let root = vs.root();
    // TODO use actual data
    let data: String = fs::read_to_string("data/training.txt")?;
    let test_string: &str = "Thank you Kyle ";
    let dictionary = Tokenizer::new(&data);

    let transformer = Transformer::new(&root, device, dictionary, 1, 128, 0.1);

    let out: Tensor = transformer.forward(test_string);
    out.print();

    //let tokens: Tensor = dictionary.encode(test_string);
    //let tokens = tokens.to_device(device);
    //tokens.print();
    //println!("{}", data.len());

    //let decoded = dictionary.decode(tokens);
    //println!("'{decoded}'");

    //let test: Vec<String> = tokenizer::parser(test_string);
    //println!("{:?}", test);

    //println!("{:?}", data);
    Ok(())
}

// References
// Min-GPT https://github.com/LaurentMazare/tch-rs/blob/main/examples/min-gpt/main.rs
// Llama https://github.com/LaurentMazare/tch-rs/blob/main/examples/llama/main.rs
