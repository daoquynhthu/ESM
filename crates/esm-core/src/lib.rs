//! Core data types and algorithms for the Elastic Sparse Machine E-1A lab.
//!
//! This crate intentionally has no third-party dependencies. It is safe Rust only.

pub mod encoder;
pub mod event;
pub mod feature;
pub mod ids;
pub mod metrics;
pub mod rng;

pub use encoder::{DenseReport, DenseUpdateStats, EncoderConfig, EncoderKind, SparseEncoder};
pub use encoder::e::{AttentionStep, EncoderE0, EncoderE1a, EncoderE1b, EncoderE1c, EncoderE2a, EncoderE2b, EncoderE2c};
pub use event::{InputEvent, TargetEvent};
pub use feature::{FeatureId, SparseCode};
pub use metrics::{compute_embedding_role_separation, E1aMetrics, E1aReport};
