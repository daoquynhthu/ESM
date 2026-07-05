//! Stable integer IDs for future ESM stores.
//!
//! These are deliberately plain integer wrappers. Core structures should point to each other by ID,
//! not by Rust references, `Rc`, or `RefCell`.

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ElementId(pub u32);

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SegmentId(pub u32);

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LinkId(pub u32);

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ClaimId(pub u32);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Phase {
    Probe,
    Active,
    Frozen,
    Dormant,
    Retired,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Bucket {
    Low,
    Unknown,
    Medium,
    High,
}
