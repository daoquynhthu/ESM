use crate::feature::FeatureId;
use std::collections::HashSet;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ElementPhase {
    Probe,
    Active,
    Retired,
    Quarantined,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ParentStatus {
    NoAdequateParent,
    WeakParent,
    StableConflictParent,
    StableCompatibleParent,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum GenesisKind {
    Round0,
    Composition,
    WeakParentRefinement,
    ClaimMismatch,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Lineage {
    Root,
    WeakParentRefinement,
    CompositionProbe,
    Fork,
}

#[derive(Clone, Debug)]
pub struct RoleObservation {
    pub role: usize,
    pub match_quality: f32,
}

#[derive(Clone, Debug)]
pub struct Element {
    pub id: u64,
    pub phase: ElementPhase,
    pub features: Vec<FeatureId>,
    pub feature_set: HashSet<FeatureId>,
    pub role_counts: Vec<u32>,
    pub total_observations: u32,
    pub correct_predictions: u32,
    pub utility: f32,
    pub plasticity: f32,
    pub resistance: f32,
    pub rent_paid: f32,
    pub age: u64,
    pub last_activation: u64,
    pub source_round: usize,
    pub genesis_kind: GenesisKind,
    pub lineage: Lineage,
}

impl Element {
    pub fn new_probe(
        id: u64,
        features: Vec<FeatureId>,
        source_round: usize,
        genesis_kind: GenesisKind,
        lineage: Lineage,
        num_roles: usize,
    ) -> Self {
        let feature_set: HashSet<FeatureId> = features.iter().copied().collect();
        Self {
            id,
            phase: ElementPhase::Probe,
            features,
            feature_set,
            role_counts: vec![0; num_roles],
            total_observations: 0,
            correct_predictions: 0,
            utility: 0.0,
            plasticity: 1.0,
            resistance: 0.1,
            rent_paid: 0.0,
            age: 0,
            last_activation: 0,
            source_round,
            genesis_kind,
            lineage,
        }
    }

    /// How many of this element's features overlap with the current code.
    pub fn overlap_count(&self, code_features: &[FeatureId]) -> usize {
        code_features.iter().filter(|f| self.feature_set.contains(f)).count()
    }

    /// Dominant role based on role_counts (most frequently observed role).
    pub fn dominant_role(&self) -> Option<(usize, u32, u32)> {
        if self.role_counts.is_empty() {
            return None;
        }
        let mut best_role = 0usize;
        let mut best = 0u32;
        let mut second = 0u32;
        for (role, count) in self.role_counts.iter().copied().enumerate() {
            if count > best {
                second = best;
                best = count;
                best_role = role;
            } else if count > second {
                second = count;
            }
        }
        if best == 0 { None } else { Some((best_role, best, second)) }
    }

    /// Vote for a role using this element's knowledge.
    /// Returns (role, confidence) where confidence = fraction of features that overlap.
    pub fn vote(&self, code_features: &[FeatureId]) -> Option<(usize, f32)> {
        let overlap = self.overlap_count(code_features);
        if overlap == 0 {
            return None;
        }
        let confidence = overlap as f32 / self.features.len().max(1) as f32;
        self.dominant_role().map(|(r, _, _)| (r, confidence))
    }

    /// Mark this element as having been activated at the given step.
    pub fn record_activation(&mut self, step: u64) {
        self.last_activation = step;
    }

    /// Record an observation (target role) and optionally mark correct/incorrect
    /// based on what this element would have predicted.
    pub fn observe(&mut self, actual_role: usize) {
        if actual_role >= self.role_counts.len() {
            return;
        }
        let predicted = self.dominant_role().map(|(r, _, _)| r);
        self.role_counts[actual_role] = self.role_counts[actual_role].saturating_add(1);
        self.total_observations += 1;
        if let Some(pred) = predicted {
            if pred == actual_role {
                self.correct_predictions += 1;
            }
        }
    }

    /// Update utility based on correct prediction ratio and rent cost.
    pub fn refresh_utility(&mut self) {
        if self.total_observations == 0 {
            self.utility = 0.0;
            return;
        }
        let accuracy = self.correct_predictions as f32 / self.total_observations as f32;
        let baseline = 1.0 / self.role_counts.len().max(2) as f32;
        // Utility = accuracy above baseline, discounted by rent
        let raw = (accuracy - baseline).max(0.0) * 2.0;
        self.utility = raw - self.rent_paid * 0.1;
    }

    /// Pay one step of rent. Returns true if element should be retired.
    pub fn pay_rent(&mut self, rent_per_step: f32) -> bool {
        self.age += 1;
        self.rent_paid += rent_per_step;
        // Retire if rent has exceeded accumulated utility for too long
        self.rent_paid > 5.0 && self.rent_paid > self.utility * 3.0 + 1.0
    }

    /// Gradual plasticity decay: probe → active transition.
    pub fn decay_plasticity(&mut self) {
        if self.phase == ElementPhase::Probe && self.age > 10 && self.utility > 0.3 {
            self.phase = ElementPhase::Active;
        }
        if self.phase == ElementPhase::Active && self.age > 100 {
            self.plasticity = (self.plasticity * 0.999).max(0.1);
            self.resistance = (self.resistance + 0.001).min(1.0);
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenesisConfig {
    /// Maximum number of elements (all phases, excluding Retired).
    pub max_elements: usize,
    /// Maximum number of Probe-phase elements.
    pub max_probes: usize,
    /// Genesis probes created per step (hard cap).
    pub probes_per_step: usize,
    /// Rent deducted per step from each element.
    pub rent_per_step: f32,
    /// Utility below this triggers retirement.
    pub utility_floor: f32,
    /// Parent coverage below this → NoAdequateParent.
    pub parent_coverage_floor: f32,
    /// Parent utility below this → NoAdequateParent.
    pub parent_utility_floor: f32,
    /// Surprise (prediction NLL) above this contributes to genesis pressure.
    pub surprise_floor: f32,
    /// Fraction of active budget reserved for probes.
    pub probe_exploration_fraction: f32,
    /// Minimum overlap for an element to be considered "covering" current input.
    pub coverage_overlap_min: f32,
}

impl Default for GenesisConfig {
    fn default() -> Self {
        Self {
            max_elements: 1024,
            max_probes: 128,
            probes_per_step: 2,
            rent_per_step: 0.01,
            utility_floor: 0.05,
            parent_coverage_floor: 0.3,
            parent_utility_floor: 0.1,
            surprise_floor: 0.5,
            probe_exploration_fraction: 0.1,
            coverage_overlap_min: 0.3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ElementStore {
    pub elements: Vec<Element>,
    next_id: u64,
    pub config: GenesisConfig,
    // Stats
    pub total_genesis_attempts: u64,
    pub total_retired: u64,
    pub total_promoted: u64,
}

impl ElementStore {
    pub fn new(config: GenesisConfig) -> Self {
        Self {
            elements: Vec::with_capacity(config.max_elements),
            next_id: 1,
            config,
            total_genesis_attempts: 0,
            total_retired: 0,
            total_promoted: 0,
        }
    }

    pub fn probe_count(&self) -> usize {
        self.elements.iter().filter(|e| e.phase == ElementPhase::Probe).count()
    }

    pub fn active_count(&self) -> usize {
        self.elements.iter().filter(|e| e.phase == ElementPhase::Active).count()
    }

    pub fn total_count(&self) -> usize {
        self.elements.len()
    }

    pub fn can_genesis(&self) -> bool {
        self.probe_count() < self.config.max_probes
            && self.total_count() < self.config.max_elements
    }

    /// Create a new probe element from the given features.
    pub fn create_probe(
        &mut self,
        features: Vec<FeatureId>,
        source_round: usize,
        genesis_kind: GenesisKind,
        lineage: Lineage,
        num_roles: usize,
    ) -> Option<u64> {
        if !self.can_genesis() {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        let element = Element::new_probe(id, features, source_round, genesis_kind, lineage, num_roles);
        self.elements.push(element);
        self.total_genesis_attempts += 1;
        Some(id)
    }

    /// Compute parent status: how well existing elements cover `code_features`.
    pub fn parent_status(&self, code_features: &[FeatureId]) -> ParentStatus {
        let active_elements: Vec<_> = self.elements.iter()
            .filter(|e| e.phase == ElementPhase::Active || e.phase == ElementPhase::Probe)
            .collect();

        if active_elements.is_empty() {
            return ParentStatus::NoAdequateParent;
        }

        let mut best_coverage = 0.0f32;
        let mut best_utility = 0.0f32;

        for elem in &active_elements {
            let overlap = elem.overlap_count(code_features);
            let coverage = overlap as f32 / code_features.len().max(1) as f32;
            if coverage > best_coverage {
                best_coverage = coverage;
                best_utility = elem.utility;
            } else if coverage == best_coverage && elem.utility > best_utility {
                best_utility = elem.utility;
            }
        }

        if best_coverage < self.config.parent_coverage_floor
            || best_utility < self.config.parent_utility_floor
        {
            ParentStatus::NoAdequateParent
        } else if best_coverage < 0.6 || best_utility < 0.3 {
            ParentStatus::WeakParent
        } else {
            // For now, default to StableCompatibleParent; StableConflictParent
            // requires interference tracking (future layer).
            ParentStatus::StableCompatibleParent
        }
    }

    ///  Run one step of lifecycle for all elements.
    ///  - Pay rent
    ///  - Retire elements whose rent > utility
    ///  - Decay plasticity for mature elements
    ///  - Promote probes with sufficient utility
    pub fn step_lifecycle(&mut self) {
        let mut to_remove: Vec<usize> = Vec::new();

        for (i, elem) in self.elements.iter_mut().enumerate() {
            if elem.phase == ElementPhase::Retired || elem.phase == ElementPhase::Quarantined {
                continue;
            }

            elem.refresh_utility();
            if elem.pay_rent(self.config.rent_per_step) {
                elem.phase = ElementPhase::Retired;
                to_remove.push(i);
                self.total_retired += 1;
                continue;
            }

            elem.decay_plasticity();
            if elem.phase == ElementPhase::Probe && elem.utility > self.config.utility_floor * 3.0 {
                elem.phase = ElementPhase::Active;
                self.total_promoted += 1;
            }
        }

        // Remove retired elements (keep elements list compact)
        for i in to_remove.into_iter().rev() {
            self.elements.swap_remove(i);
        }
    }

    /// Observe the actual role for all elements that overlap with `code_features`.
    pub fn observe_active_elements(&mut self, code_features: &[FeatureId], actual_role: usize, step: u64) {
        for elem in self.elements.iter_mut() {
            if elem.phase == ElementPhase::Retired || elem.phase == ElementPhase::Quarantined {
                continue;
            }
            if elem.overlap_count(code_features) > 0 {
                elem.observe(actual_role);
                elem.record_activation(step);
            }
        }
    }

    ///  Collect votes from all elements that overlap with `code_features`.
    ///  Returns (vote_counts_per_role, total_weight).
    pub fn collect_votes(&self, code_features: &[FeatureId]) -> (Vec<u32>, f32) {
        let mut votes = vec![0u32; 2]; // will resize if needed
        let mut total_weight = 0.0f32;

        for elem in self.elements.iter() {
            if elem.phase == ElementPhase::Retired || elem.phase == ElementPhase::Quarantined {
                continue;
            }
            if let Some((role, confidence)) = elem.vote(code_features) {
                let needed = role + 1;
                if votes.len() < needed {
                    votes.resize(needed, 0);
                }
                votes[role] = votes[role].saturating_add((confidence * 10.0).max(1.0) as u32);
                total_weight += confidence;
            }
        }

        (votes, total_weight)
    }
}
