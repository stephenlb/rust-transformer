use regex::Regex;
use std::collections::HashMap;
use tch::{Kind, Tensor};

pub struct Tokenizer {
    encoder: HashMap<String, i64>,
    decoder: HashMap<i64, String>,
    pub length: i64,
}

pub fn parser(text: &str) -> Vec<String> {
    let matcher = Regex::new(r"\w+|.").unwrap();
    let words: Vec<String> = matcher
        .find_iter(text)
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

        let length = decoder.len() as i64;
        Self {
            encoder,
            decoder,
            length,
        }
    }

    pub fn encode_one(&self, text: &str) -> Vec<i64> {
        let tokens: Vec<i64> = parser(text)
            .iter()
            .filter_map(|word| self.encoder.get(word).copied())
            .collect();
        tokens

    }
    pub fn encode(&self, batch: Vec<&str>) -> Tensor {
        let tokens: Vec<Vec<i64>> = batch
            .iter()
            .map(|text|
                parser(text)
                    .iter()
                    .filter_map(|word| self.encoder.get(word).copied())
                    .collect()
            )
            .collect();

        return Tensor::from_slice2(&tokens);
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
