//! Experiment runners for ESM gates.

pub mod e1a;
pub mod stream;

pub use e1a::{run_e1a, E1aConfig};
pub use stream::{StreamKind, SyntheticStream};
