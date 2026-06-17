use regex::Regex;
use std::collections::HashMap;
use tch::{Tensor, Kind};

pub struct Tokenizer {
    encoder: HashMap<String, i64>,
    decoder: HashMap<i64, String>,
    pub vocab: i64,
}

pub fn parser(text: &str) -> Vec<String> {
    let matcher = Regex::new(r"\w+|.").unwrap();
    let words: Vec<String> = matcher.find_iter(text)
        .map(|word| word.as_str().to_string())
        .collect();
    return words;
}

impl Tokenizer {
    pub fn new(text: &str) -> Self {
        let mut words: Vec<String> = parser(text);
        words.sort_unstable();
        words.dedup();

        let mut encoder = HashMap::new(); // <- O(1) wow!
        let mut decoder = HashMap::new(); // <- O(1) wow!

        // Token 0 is the Padding
        encoder.insert("<padding>".to_string(), 0);
        decoder.insert(0, "<padding>".to_string());

        for (token, word) in words.iter().enumerate() {
            let token = (token + 1) as i64;
            encoder.insert(word.to_string(), token);
            decoder.insert(token, word.to_string());
        }

        Self {
            encoder,
            decoder,
            vocab: words.len() as i64,
        }
    }
    
    pub fn encode(&self, text: &str) -> Tensor {
        let tokens: Vec<i64> = parser(text)
            .iter()
            .filter_map(|c| self.encoder.get(c).copied())
            .collect();
        return Tensor::from_slice(&tokens);
    }

    //                   batch seq probs <- TODO
    //
    pub fn decode(&self, tokens: Tensor) -> String {
        let tokens: Vec<i64> = tokens.try_into().unwrap();
        tokens
            .iter()
            .map(|token| self.decoder.get(&token).unwrap().as_str().to_string())
            .collect()
    }

}

/*
    let text = "Hello, Rust!";

    // 1. Tokenize into individual characters
    let tokens: Vec<char> = text.chars().collect();
    
    // 2. Convert characters into unique numerical token IDs
    let token_ids: Vec<u32> = tokens.iter().map(|&c| c as u32).collect();

    println!("Text:   \"{}\"", text);
    println!("Tokens: {:?}", tokens);
    println!("IDs:    {:?}", token_ids);


use regex::Regex;
let text = "Hello, world! 123.";
let re = Regex::new(r"\w+|.").unwrap();
let tokens: Vec<&str> = re.find_iter(text)
    .map(|m| m.as_str())
    .collect();



*/
