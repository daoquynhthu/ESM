//! Deterministic synthetic streams for Gate E-1A.

use esm_core::event::{InputEvent, TargetEvent};
use esm_core::rng::XorShift64;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum StreamKind {
    SameTokenContext,
    RoleSharing,
    DelayedRole,
    DelayedCue,
    DelayedCueVerifyOnly,
}

impl StreamKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "same-token-context" | "same_token_context" | "context" => Some(Self::SameTokenContext),
            "role-sharing" | "role_sharing" | "sharing" => Some(Self::RoleSharing),
            "delayed-role" | "delayed_role" | "delayed" => Some(Self::DelayedRole),
            "delayed-cue" | "delayed_cue" | "cue" => Some(Self::DelayedCue),
            "delayed-cue-verify-only" | "verify-only" | "verifyonly" => Some(Self::DelayedCueVerifyOnly),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::SameTokenContext => "same-token-context",
            Self::RoleSharing => "role-sharing",
            Self::DelayedRole => "delayed-role",
            Self::DelayedCue => "delayed-cue",
            Self::DelayedCueVerifyOnly => "delayed-cue-verify-only",
        }
    }
}

pub trait SyntheticStream {
    fn name(&self) -> &'static str;
    fn next_sample(&mut self) -> (InputEvent, TargetEvent);
}

pub fn build_stream(kind: StreamKind, seed: u64) -> Box<dyn SyntheticStream> {
    match kind {
        StreamKind::SameTokenContext => Box::new(SameTokenContextStream::new(seed)),
        StreamKind::RoleSharing => Box::new(RoleSharingStream::new(seed)),
        StreamKind::DelayedRole => Box::new(DelayedRoleStream::new(seed)),
        StreamKind::DelayedCue => Box::new(DelayedCueStream::new(seed)),
        StreamKind::DelayedCueVerifyOnly => Box::new(DelayedCueVerifyOnlyStream::new(seed)),
    }
}

#[derive(Clone, Debug)]
struct SameTokenContextStream {
    step: u64,
    prev_token: u32,
    rng: XorShift64,
}

impl SameTokenContextStream {
    fn new(seed: u64) -> Self {
        Self { step: 0, prev_token: 0, rng: XorShift64::new(seed) }
    }
}

impl SyntheticStream for SameTokenContextStream {
    fn name(&self) -> &'static str { "same-token-context" }

    fn next_sample(&mut self) -> (InputEvent, TargetEvent) {
        let role = (self.step / 2 % 2) as u32;
        let context = if role == 0 { 10 } else { 20 };
        let token = if self.step % 2 == 0 { context } else { 42 };
        let noise = if self.rng.next_usize(16) == 0 { 7 } else { 0 };
        let input = InputEvent {
            step: self.step,
            token: token + noise,
            prev_token: self.prev_token,
            context_token: if token == 42 { context } else { 0 },
            position_mod: (self.step % 8) as u32,
        };
        let target = TargetEvent { latent_role: role, next_token: if role == 0 { 100 } else { 200 } };
        self.prev_token = input.token;
        self.step += 1;
        (input, target)
    }
}

#[derive(Clone, Debug)]
struct RoleSharingStream {
    step: u64,
    prev_token: u32,
    rng: XorShift64,
}

impl RoleSharingStream {
    fn new(seed: u64) -> Self {
        Self { step: 0, prev_token: 0, rng: XorShift64::new(seed) }
    }
}

impl SyntheticStream for RoleSharingStream {
    fn name(&self) -> &'static str { "role-sharing" }

    fn next_sample(&mut self) -> (InputEvent, TargetEvent) {
        let role = (self.step % 3) as u32;
        let base = match role { 0 => 1000, 1 => 2000, _ => 3000 };
        let token = base + self.rng.next_usize(4) as u32;
        let context = 9000 + role;
        let input = InputEvent {
            step: self.step,
            token,
            prev_token: self.prev_token,
            context_token: context,
            position_mod: (self.step % 16) as u32,
        };
        let target = TargetEvent { latent_role: role, next_token: 5000 + role };
        self.prev_token = token;
        self.step += 1;
        (input, target)
    }
}

#[derive(Clone, Debug)]
struct DelayedRoleStream {
    step: u64,
    prev_token: u32,
    current_role: u32,
    rng: XorShift64,
}

impl DelayedRoleStream {
    fn new(seed: u64) -> Self {
        Self { step: 0, prev_token: 0, current_role: 0, rng: XorShift64::new(seed) }
    }
}

impl SyntheticStream for DelayedRoleStream {
    fn name(&self) -> &'static str { "delayed-role" }

    fn next_sample(&mut self) -> (InputEvent, TargetEvent) {
        if self.step % 5 == 0 {
            self.current_role = self.rng.next_usize(2) as u32;
        }
        let phase = self.step % 5;
        let token = match phase {
            0 => if self.current_role == 0 { 31 } else { 37 },
            1 | 2 | 3 => 42,
            _ => if self.current_role == 0 { 71 } else { 73 },
        };
        let input = InputEvent {
            step: self.step,
            token,
            prev_token: self.prev_token,
            context_token: if phase == 0 { token } else { 0 },
            position_mod: phase as u32,
        };
        let target = TargetEvent {
            latent_role: self.current_role,
            next_token: if self.current_role == 0 { 700 } else { 900 },
        };
        self.prev_token = token;
        self.step += 1;
        (input, target)
    }
}

// =========================================================================
// Delayed-cue stream (E-1B bridge)
// =========================================================================

/// 6-step cycle: [CUE, FILLER×4, VERIFY]
///
/// At step 0, a cue token (100 or 101) determines the latent role.
/// Steps 1-4 are random filler tokens with random context (no role signal).
/// At step 5, a verification token (300) appears with random context.
/// The role is revealed via TargetEvent at every step (prequential protocol),
/// but the verification step carries no token/context signal for the role,
/// so the encoder must bridge the temporal gap without help from prototype
/// context→role mappings.
#[derive(Clone, Debug)]
struct DelayedCueStream {
    step: u64,
    prev_token: u32,
    current_role: u32,
    cycle_context: u32,
    rng: XorShift64,
}

impl DelayedCueStream {
    fn new(seed: u64) -> Self {
        Self { step: 0, prev_token: 0, current_role: 0, cycle_context: 0, rng: XorShift64::new(seed) }
    }
}

impl SyntheticStream for DelayedCueStream {
    fn name(&self) -> &'static str { "delayed-cue" }

    fn next_sample(&mut self) -> (InputEvent, TargetEvent) {
        let phase = self.step % 6;
        if phase == 0 {
            self.current_role = self.rng.next_usize(2) as u32;
            // Same context per cycle: shared between cue and verify
            self.cycle_context = self.rng.next_usize(4096) as u32;
        }

        let token = match phase {
            0 => if self.current_role == 0 { 100 } else { 101 },
            1 | 2 | 3 | 4 => 200 + self.rng.next_usize(8) as u32,
            _ => 300, // verification
        };

        let context_token = if phase == 0 || phase == 5 {
            // Cue and verify share the same context
            20000 + self.cycle_context
        } else {
            self.rng.next_usize(1024) as u32
        };

        let input = InputEvent {
            step: self.step,
            token,
            prev_token: self.prev_token,
            context_token,
            position_mod: phase as u32,
        };
        let target = TargetEvent {
            latent_role: self.current_role,
            next_token: 0,
        };
        self.prev_token = token;
        self.step += 1;
        (input, target)
    }
}

/// Delayed-cue stream where the role is ONLY revealed at verify step (phase 5).
/// At cue and filler steps, the role is random noise (no useful information).
/// This creates a genuine delayed credit assignment problem: the encoder must
/// learn the cue→role association through retrospective ledger credit.
struct DelayedCueVerifyOnlyStream {
    step: u64,
    prev_token: u32,
    current_role: u32,
    cycle_context: u32,
    rng: XorShift64,
}

impl DelayedCueVerifyOnlyStream {
    fn new(seed: u64) -> Self {
        Self { step: 0, prev_token: 0, current_role: 0, cycle_context: 0, rng: XorShift64::new(seed) }
    }
}

impl SyntheticStream for DelayedCueVerifyOnlyStream {
    fn name(&self) -> &'static str { "delayed-cue-verify-only" }

    fn next_sample(&mut self) -> (InputEvent, TargetEvent) {
        let phase = self.step % 6;
        if phase == 0 {
            self.current_role = self.rng.next_usize(2) as u32;
            self.cycle_context = self.rng.next_usize(4096) as u32;
        }

        let token = match phase {
            0 => 100 + self.current_role, // 100 for role 0, 101 for role 1
            1 | 2 | 3 | 4 => 200 + self.rng.next_usize(8) as u32,
            _ => 300, // verification
        };

        let context_token = if phase == 0 || phase == 5 {
            20000 + self.cycle_context
        } else {
            self.rng.next_usize(1024) as u32
        };

        // Role is ONLY revealed at verify step.
        // At cue/filler steps: random noise role (50/50).
        let reveal_role = phase == 5;
        let role = if reveal_role { self.current_role } else { self.rng.next_usize(2) as u32 };

        let input = InputEvent {
            step: self.step,
            token,
            prev_token: self.prev_token,
            context_token,
            position_mod: phase as u32,
        };
        let target = TargetEvent {
            latent_role: role,
            next_token: 0,
        };
        self.prev_token = token;
        self.step += 1;
        (input, target)
    }
}
