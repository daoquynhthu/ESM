use std::env;
use std::process::ExitCode;

use esm_core::encoder::EncoderKind;
use esm_runner::{run_e1a, E1aConfig, StreamKind};

fn main() -> ExitCode {
    match real_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            print_usage();
            ExitCode::from(2)
        }
    }
}

fn real_main() -> Result<(), String> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return Ok(());
    }

    let cmd = args.remove(0);
    match cmd.as_str() {
        "run" => run_cmd(&args),
        _ => Err(format!("unknown command '{cmd}'")),
    }
}

fn run_cmd(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) != Some("e1a") {
        return Err("expected: run e1a".to_string());
    }

    let mut cfg = E1aConfig::default();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--stream" => {
                let value = take_value(args, i)?;
                cfg.stream = StreamKind::parse(value)
                    .ok_or_else(|| format!("unknown stream '{value}'"))?;
                i += 2;
            }
            "--encoder" => {
                let value = take_value(args, i)?;
                cfg.encoder = EncoderKind::parse(value)
                    .ok_or_else(|| format!("unknown encoder '{value}'"))?;
                i += 2;
            }
            "--steps" => {
                let value = take_value(args, i)?;
                cfg.steps = parse_u64(value, "steps")?;
                i += 2;
            }
            "--seed" => {
                let value = take_value(args, i)?;
                cfg.seed = parse_u64(value, "seed")?;
                i += 2;
            }
            "--active-bits" => {
                let value = take_value(args, i)?;
                cfg.active_bits = parse_usize(value, "active-bits")?;
                i += 2;
            }
            "--columns" => {
                let value = take_value(args, i)?;
                cfg.columns = parse_usize(value, "columns")?;
                i += 2;
            }
            "--sample-limit" => {
                let value = take_value(args, i)?;
                cfg.sample_limit = parse_usize(value, "sample-limit")?;
                i += 2;
            }
            "--lr" => {
                let value = take_value(args, i)?;
                cfg.lr = value.parse::<f32>().map_err(|_| format!("invalid lr '{value}'"))?;
                i += 2;
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
    }

    let report = run_e1a(cfg);
    println!("{}", report.to_json_pretty());
    Ok(())
}

fn take_value(args: &[String], i: usize) -> Result<&str, String> {
    args.get(i + 1)
        .map(String::as_str)
        .ok_or_else(|| format!("missing value for {}", args[i]))
}

fn parse_u64(s: &str, name: &str) -> Result<u64, String> {
    s.parse::<u64>().map_err(|_| format!("invalid {name}: '{s}'"))
}

fn parse_usize(s: &str, name: &str) -> Result<usize, String> {
    s.parse::<usize>().map_err(|_| format!("invalid {name}: '{s}'"))
}

fn print_usage() {
    eprintln!(
        "Usage:\n  esm run e1a [--stream same-token-context|role-sharing|delayed-role] \\\n                  [--encoder hash|competitive|predictive|d|...|e0|e1a|e1b|e1c] \\\n                  [--steps N] [--seed N] [--lr F]\n\nEncoders:\n  hash / a / control           Raw token/hash baseline\n  competitive / b               Sparse projection + homeostasis (v1)\n  predictive / c                Sparse + context-key role prototypes (v2)\n  d / d-full / d-no-trace / d-no-role-proto   Archived D-series\n  e0 / encoder-e0               Predictive + mean-pooled linear decoder\n  e1a / e1-attn-linear          Attention + linear readout (E1a)\n  e1b / e1-mean-mlp             Mean + one-hidden-layer MLP (E1b ablation)\n  e1c / e1-attn-mlp             Attention + one-hidden-layer MLP (E1c)\n  e2a / e2-credit-promote       Promote positive-credit features (E2a)\n  e2b / e2-credit-promote-suppress  Promote + suppress (E2b)\n  e2c / e2-no-loo               Uniform global-loss shaping (E2c)\n\nExamples:\n  esm run e1a --stream same-token-context --encoder hash --steps 10000\n  esm run e1a --stream role-sharing --encoder e1c --steps 10000 --lr 0.01\n  esm run e1a --stream role-sharing --encoder e2b --steps 10000"
    );
}
