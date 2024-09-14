use linfa::traits::{Fit, Predict};
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use linfa_nn::distance::LInfDist;
use linfa_preprocessing::tf_idf_vectorization::TfIdfVectorizer;
use ndarray::{Array1, Array2};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use regex::Regex;
use std::collections::HashMap;
use std::io::{self, BufRead, BufWriter, Write};
use tracing::debug;
use tracing_subscriber::fmt;
use tracing_subscriber::fmt::format::FmtSpan;
extern crate serde_json;
use clap::Parser;


mod word_bank;


#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Print output in grouped JSON format
    #[arg(short, long)]
    groups: bool,
}

fn format_output(args: &Args, clusters: Vec<Vec<String>>) {
    if args.groups {
        println!("{}", serde_json::to_string_pretty(&clusters).unwrap());
    } else {
        let stdout = io::stdout();
        let mut writer = BufWriter::new(stdout.lock());
        for cluster in clusters.iter() {
            for line in cluster {
                writeln!(writer, "{}", line).unwrap();
            }
            // writeln!(writer).unwrap();
        }
        writer.flush().unwrap();
    }
}



fn main() {
    setup_logging();
    let args = Args::parse();

    let stdin = io::stdin();
    let lines: Vec<String> = stdin
        .lock()
        .lines()
        .map_while(Result::ok)
        .filter(|line| !line.trim().to_string().is_empty())
        .collect();
    // dbg!(&lines);

    let sentence_strings = preprocess(&lines);
    // dbg!(sentence_strings);

    let clusters = find_clusters(sentence_strings, lines);
    format_output(&args, clusters);
}

fn default_cluster_count(records: &Array2<f64>) -> usize {
    let num_rows = records.shape()[0];
    (num_rows as f64).sqrt().ceil() as usize
}

fn find_clusters(sentence_strings: Vec<String>, lines: Vec<String>) -> Vec<Vec<String>> {
    let targets = Array1::from_shape_vec(sentence_strings.len(), sentence_strings.clone()).unwrap();
    let vectorizer = TfIdfVectorizer::default()
        .stopwords(word_bank::STOP_WORDS)
        // .n_gram_range(1,2)
        .convert_to_lowercase(false)
        .fit(&targets)
        .unwrap();
    debug!(
        "We obtain a vocabulary with {} entries",
        vectorizer.nentries()
    );

    let training_records = vectorizer.transform(&targets).to_dense();
    debug!(
        "We obtain a {}x{} matrix of counts for the vocabulary entries",
        training_records.dim().0,
        training_records.dim().1
    );

    let n_clusters = default_cluster_count(&training_records);
    let dataset = DatasetBase::from((training_records, targets));
    let rng = Xoshiro256Plus::seed_from_u64(42);
    let kmeans = KMeans::params_with(n_clusters, rng, LInfDist)
        .max_n_iterations(10)
        .tolerance(1e-4)
        .fit(&dataset)
        .expect("KMeans fitted");

    let predictions = kmeans.predict(dataset);

    // dbg!(&predictions);

    // Group lines by their cluster
    let mut clusters: Vec<Vec<String>> = vec![Vec::new(); n_clusters];
    for (line, cluster) in lines.iter().zip(predictions.targets().iter()) {
        clusters[*cluster].push(line.clone());
    }
    clusters.sort_by_key(|cluster| cluster.len());
    clusters
}

fn setup_logging() {
    fmt()
        .with_target(false)
        .with_timer(tracing_subscriber::fmt::time::uptime())
        .with_level(true)
        .with_span_events(FmtSpan::CLOSE)
        .init();
}

fn preprocess(lines: &Vec<String>) -> Vec<String> {
    let mut sentence_strings: Vec<String> = lines
        .iter()
        .map(|line| {
            line.replace("/", " ")
                .replace("\\", " ")
                .replace(".", " ")
                .replace("[", " ")
                .replace("(", " ")
                .replace("]", " ")
                .replace(")", " ")
                .replace("{", " ")
                .replace("}", " ")
                .replace("_", " ")
                .replace("-", " ")
                .to_lowercase()
        })
        .collect();

    let word_regex = Regex::new(r"\b\w\w+\b").unwrap();

    // Tokenize sentences and count word occurrences
    let mut word_counts: HashMap<String, usize> = HashMap::new();
    for line in &sentence_strings {
        for word in word_regex.find_iter(&line) {
            *word_counts.entry(word.as_str().to_string()).or_insert(0) += 1;
        }
    }

    // Filter out unique words
    sentence_strings = sentence_strings
        .iter()
        .map(|line| {
            line.split_whitespace()
                .filter_map(|word| {
                    if word_counts.get(word).unwrap_or(&0) > &1 {
                        Some(word.to_string())
                    } else {
                        None
                    }
                })
                .collect::<Vec<String>>()
                .join(" ")
        })
        .collect();

    return sentence_strings;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess() {
        let input = vec![
            "hippopotamus TURTLE hippopotamus".to_string(),
            "turtle".to_string(),
            "squirrel!".to_string(),
        ];

        let output = preprocess(&input);

        // same length
        assert_eq!(input.len(), output.len());

        // same order
        assert_eq!(
            output,
            vec![
                "hippopotamus turtle hippopotamus".to_string(),
                "turtle".to_string(),
                "".to_string(),
            ]
        );
    }
}
