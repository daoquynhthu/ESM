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
    DelayedCueFixedContext,
    DelayedCueCyclingContext,
    CompositionDepth(usize),
}

impl StreamKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "same-token-context" | "same_token_context" | "context" => Some(Self::SameTokenContext),
            "role-sharing" | "role_sharing" | "sharing" => Some(Self::RoleSharing),
            "delayed-role" | "delayed_role" | "delayed" => Some(Self::DelayedRole),
            "delayed-cue" | "delayed_cue" | "cue" => Some(Self::DelayedCue),
            "delayed-cue-verify-only" | "verify-only" | "verifyonly" => Some(Self::DelayedCueVerifyOnly),
            "delayed-cue-fixed-context" | "fixed-context" | "fixedctx" => Some(Self::DelayedCueFixedContext),
            "delayed-cue-cycling-context" | "cycling-context" | "cycling" => Some(Self::DelayedCueCyclingContext),
            "composition-depth-1" | "comp1" | "d1" => Some(Self::CompositionDepth(1)),
            "composition-depth-2" | "comp2" | "d2" => Some(Self::CompositionDepth(2)),
            "composition-depth-3" | "comp3" | "d3" => Some(Self::CompositionDepth(3)),
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
            Self::DelayedCueFixedContext => "delayed-cue-fixed-context",
            Self::DelayedCueCyclingContext => "delayed-cue-cycling-context",
            Self::CompositionDepth(d) => {
                // Static lifetime; we have exactly 3 valid depths.
                match d {
                    1 => "composition-depth-1",
                    2 => "composition-depth-2",
                    _ => "composition-depth-3",
                }
            }
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
        StreamKind::DelayedCueFixedContext => Box::new(DelayedCueFixedContextStream::new(seed)),
        StreamKind::DelayedCueCyclingContext => Box::new(DelayedCueCyclingContextStream::new(seed)),
        StreamKind::CompositionDepth(d) => Box::new(CompositionDepthStream::new(seed, d)),
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

/// Delayed-cue stream with FIXED context per role and verify-only reveal.
/// Role 0 → context 30000, Role 1 → context 30001.
/// Same context at cue and verify for each role.
/// Role revealed ONLY at verify step (noise at cue/filler).
/// This enables cross-cycle ledger accumulation: the same columns fire
/// every time the same role occurs, so ledger counts accumulate.
struct DelayedCueFixedContextStream {
    step: u64,
    prev_token: u32,
    current_role: u32,
    current_filler: u32,
    rng: XorShift64,
}

impl DelayedCueFixedContextStream {
    fn new(seed: u64) -> Self {
        Self { step: 0, prev_token: 0, current_role: 0, current_filler: 0, rng: XorShift64::new(seed) }
    }
}

impl SyntheticStream for DelayedCueFixedContextStream {
    fn name(&self) -> &'static str { "delayed-cue-fixed-context" }

    fn next_sample(&mut self) -> (InputEvent, TargetEvent) {
        let phase = self.step % 6;
        if phase == 0 {
            self.current_role = self.rng.next_usize(2) as u32;
            self.current_filler = self.rng.next_usize(8) as u32;
        }

        let token = match phase {
            0 => 100 + self.current_role,
            1 | 2 | 3 | 4 => 200 + self.current_filler,
            _ => 300,
        };

        // Fixed context per role: cue and verify share the SAME role-specific context.
        // Context values are widely separated (30000 vs 31000) to avoid XOR collision
        // in the sketch hash with seed=1. Adjacent values (30000/30001) collide because
        // term.value ^ seed ^ (term_idx<<32) swaps the hash inputs between roles.
        let context_token: u32 = match phase {
            0 | 5 => 30000 + self.current_role * 1000,
            _ => 40000 + self.current_filler,
        };

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
/// Delayed-cue stream with 8 CYCLING contexts.
///
/// Each cycle gets a cycling context index (0..7) that repeats every 8 cycles.
/// Within a cycle, cue and verify share the same cycling context, but the role
/// determines the exact context value:
///   Role 0: context = 30000 + ctx_idx * 1000
///   Role 1: context = 30000 + ctx_idx * 1000 + 500
///
/// Contexts repeat every 8 cycles, enabling cross-cycle column reuse while
/// still being more challenging than the fixed-context stream.
struct DelayedCueCyclingContextStream {
    step: u64,
    prev_token: u32,
    current_role: u32,
    cycle_ctx: u32,
    cycle_idx: u64,
    rng: XorShift64,
}

impl DelayedCueCyclingContextStream {
    fn new(seed: u64) -> Self {
        Self { step: 0, prev_token: 0, current_role: 0, cycle_ctx: 0, cycle_idx: 0, rng: XorShift64::new(seed) }
    }
}

impl SyntheticStream for DelayedCueCyclingContextStream {
    fn name(&self) -> &'static str { "delayed-cue-cycling-context" }

    fn next_sample(&mut self) -> (InputEvent, TargetEvent) {
        let phase = self.step % 6;
        if phase == 0 {
            self.current_role = self.rng.next_usize(2) as u32;
            self.cycle_ctx = (self.cycle_idx % 8) as u32;
            self.cycle_idx += 1;
        }

        let token = match phase {
            0 => 100 + self.current_role,
            1 | 2 | 3 | 4 => 200 + self.rng.next_usize(8) as u32,
            _ => 300,
        };

        // Role determines exact context value within cycling context bucket.
        // 8 cycling contexts × 2 roles = 16 distinct context values.
        let context_token = match phase {
            0 | 5 => 30000 + self.cycle_ctx * 1000 + self.current_role * 500,
            _ => 40000 + self.rng.next_usize(8) as u32,
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

// =========================================================================
// Composition-depth stream (Gate E-1C)
// =========================================================================
//
// Tests whether the encoder is a single-layer threshold detector (D=1) or
// can exploit pairwise (D=2) and triple (D=3) compositional structure.
//
// D=1 (control): token=100, context=200 always. Role = random 50/50.
//   All columns fire identically every step → ~50% accuracy.
//
// D=2 (pairwise XOR): token∈{100,101}, context∈{10,20}.
//   Role = XOR(token_bit, context_bit).
//   Individual terms 50/50, token^ctx term deterministic → 100%.
//
// D=3 (triple XOR): token∈{100,101}, context∈{10,20}, prev∈{1000,2000}.
//   Role = XOR(token_bit, context_bit, prev_bit).
//   All pairwise terms 50/50, no triple hash term → ~50%.
//
// Pass condition: D=2 accuracy > D=1 accuracy (compositional disambiguation).
struct CompositionDepthStream {
    step: u64,
    prev_token: u32,
    rng: XorShift64,
    depth: usize,
}

impl CompositionDepthStream {
    fn new(seed: u64, depth: usize) -> Self {
        Self { step: 0, prev_token: 0, rng: XorShift64::new(seed), depth }
    }
}

impl SyntheticStream for CompositionDepthStream {
    fn name(&self) -> &'static str {
        match self.depth {
            1 => "composition-depth-1",
            2 => "composition-depth-2",
            _ => "composition-depth-3",
        }
    }

    fn next_sample(&mut self) -> (InputEvent, TargetEvent) {
        let (token, context_token, prev_token, role) = match self.depth {
            // D=1: no compositional signal, token+context always same
            1 => {
                let token = 100u32;
                let context = 200u32;
                let role = self.rng.next_usize(2) as u32;
                (token, context, 0u32, role)
            }
            // D=2: pairwise XOR — individual terms 50/50, token^ctx deterministic
            2 => {
                let token = if self.rng.next_usize(2) == 0 { 100 } else { 101 };
                let context = if self.rng.next_usize(2) == 0 { 10 } else { 20 };
                let token_bit = (token != 100) as u32;
                let ctx_bit = (context != 10) as u32;
                let role = token_bit ^ ctx_bit;
                (token, context, 0u32, role)
            }
            // D=3: triple XOR — all pairwise 50/50, no triple hash term
            _ => {
                let token = if self.rng.next_usize(2) == 0 { 100 } else { 101 };
                let context = if self.rng.next_usize(2) == 0 { 10 } else { 20 };
                let prev = if self.rng.next_usize(2) == 0 { 1000 } else { 2000 };
                let token_bit = (token != 100) as u32;
                let ctx_bit = (context != 10) as u32;
                let prev_bit = (prev != 1000) as u32;
                let role = token_bit ^ ctx_bit ^ prev_bit;
                (token, context, prev, role)
            }
        };

        let input = InputEvent {
            step: self.step,
            token,
            prev_token,
            context_token,
            position_mod: 0,
        };
        let target = TargetEvent { latent_role: role, next_token: 0 };
        self.step += 1;
        self.prev_token = token;
        (input, target)
    }
}

// =========================================================================
// E-1D Genesis streams
// =========================================================================

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum E1dStreamKind {
    EmptyField,
    NovelPattern,
    RareEvent,
}

impl E1dStreamKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "empty-field" | "empty" => Some(Self::EmptyField),
            "novel-pattern" | "novel" => Some(Self::NovelPattern),
            "rare-event" | "rare" => Some(Self::RareEvent),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::EmptyField => "empty-field",
            Self::NovelPattern => "novel-pattern",
            Self::RareEvent => "rare-event",
        }
    }
}

pub fn build_e1d_stream(kind: E1dStreamKind, seed: u64) -> Box<dyn SyntheticStream> {
    match kind {
        E1dStreamKind::EmptyField => Box::new(E1dEmptyFieldStream::new(seed)),
        E1dStreamKind::NovelPattern => Box::new(E1dNovelPatternStream::new(seed)),
        E1dStreamKind::RareEvent => Box::new(E1dRareEventStream::new(seed)),
    }
}

/// Every step is pure random noise. No structure to discover.
/// Tests that genesis does not waste probes on noise.
struct E1dEmptyFieldStream {
    step: u64,
    prev_token: u32,
    rng: XorShift64,
}

impl E1dEmptyFieldStream {
    fn new(seed: u64) -> Self {
        Self { step: 0, prev_token: 0, rng: XorShift64::new(seed) }
    }
}

impl SyntheticStream for E1dEmptyFieldStream {
    fn name(&self) -> &'static str { "empty-field" }

    fn next_sample(&mut self) -> (InputEvent, TargetEvent) {
        let token = self.rng.next_usize(10000) as u32;
        let context = self.rng.next_usize(10000) as u32;
        let role = self.rng.next_usize(2) as u32;
        let input = InputEvent {
            step: self.step,
            token,
            prev_token: self.prev_token,
            context_token: context,
            position_mod: (self.step % 8) as u32,
        };
        let target = TargetEvent { latent_role: role, next_token: 0 };
        self.prev_token = token;
        self.step += 1;
        (input, target)
    }
}

/// Phase 1 (burn_in steps): random noise. Phase 2: novel token→role mapping.
/// Tests that genesis creates probes when novel structure appears.
struct E1dNovelPatternStream {
    step: u64,
    prev_token: u32,
    rng: XorShift64,
}

impl E1dNovelPatternStream {
    fn new(seed: u64) -> Self {
        Self { step: 0, prev_token: 0, rng: XorShift64::new(seed) }
    }
}

impl SyntheticStream for E1dNovelPatternStream {
    fn name(&self) -> &'static str { "novel-pattern" }

    fn next_sample(&mut self) -> (InputEvent, TargetEvent) {
        let burn_in = 5000u64;
        let (token, context, role) = if self.step < burn_in {
            // Phase 1: random noise
            let token = self.rng.next_usize(10000) as u32;
            let context = self.rng.next_usize(10000) as u32;
            let role = self.rng.next_usize(2) as u32;
            (token, context, role)
        } else {
            // Phase 2: novel pattern with fixed context
            let token = if self.rng.next_usize(2) == 0 { 1000 } else { 1001 };
            let context = 5000u32;
            let role = if token == 1000 { 2 } else { 3 };
            (token, context, role)
        };
        let input = InputEvent {
            step: self.step,
            token,
            prev_token: self.prev_token,
            context_token: context,
            position_mod: (self.step % 8) as u32,
        };
        let target = TargetEvent { latent_role: role, next_token: 0 };
        self.prev_token = token;
        self.step += 1;
        (input, target)
    }
}

/// 90% role 0 (token 100), 10% role 1 (token 101), delayed-cue 6-step cycle.
/// Both cue and verify share the same context.
/// Tests that probes survive for rare events.
struct E1dRareEventStream {
    step: u64,
    prev_token: u32,
    current_role: u32,
    cycle_context: u32,
    rng: XorShift64,
}

impl E1dRareEventStream {
    fn new(seed: u64) -> Self {
        Self { step: 0, prev_token: 0, current_role: 0, cycle_context: 0, rng: XorShift64::new(seed) }
    }
}

impl SyntheticStream for E1dRareEventStream {
    fn name(&self) -> &'static str { "rare-event" }

    fn next_sample(&mut self) -> (InputEvent, TargetEvent) {
        let phase = self.step % 6;
        if phase == 0 {
            // 90% role 0, 10% role 1
            self.current_role = if self.rng.next_usize(10) == 0 { 1 } else { 0 };
            self.cycle_context = self.rng.next_usize(4096) as u32;
        }

        let token = match phase {
            0 => 100 + self.current_role,
            1 | 2 | 3 | 4 => 200 + self.rng.next_usize(8) as u32,
            _ => 300,
        };
        let context_token = match phase {
            0 | 5 => 20000 + self.cycle_context,
            _ => self.rng.next_usize(1024) as u32,
        };

        let input = InputEvent {
            step: self.step,
            token,
            prev_token: self.prev_token,
            context_token,
            position_mod: phase as u32,
        };
        let target = TargetEvent { latent_role: self.current_role, next_token: 0 };
        self.prev_token = token;
        self.step += 1;
        (input, target)
    }
}
