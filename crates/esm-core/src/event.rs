//! Event types for strict prequential experiments.

#[derive(Copy, Clone, Debug, Default)]
pub struct InputEvent {
    pub step: u64,
    pub token: u32,
    pub prev_token: u32,
    pub context_token: u32,
    pub position_mod: u32,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct TargetEvent {
    /// Diagnostic latent role. Encoders must not read this during encode.
    pub latent_role: u32,
    pub next_token: u32,
}

impl InputEvent {
    pub fn input_sketch(&self) -> [u64; 6] {
        [
            self.token as u64,
            0x100000000u64 | self.prev_token as u64,
            0x200000000u64 | self.context_token as u64,
            0x300000000u64 | self.position_mod as u64,
            0x400000000u64 | (((self.token as u64) << 32) ^ self.prev_token as u64),
            0x500000000u64 | (((self.token as u64) << 32) ^ self.context_token as u64),
        ]
    }
}
