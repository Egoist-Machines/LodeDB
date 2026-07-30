//! Write-time admission policy — the seam a caller uses to constrain what may enter the graph.
//!
//! Graphiti's `add_episode` decides what to write with an LLM, and its graph store accepts
//! whatever comes back. This crate deliberately has no LLM (see the crate-level *Scope* note),
//! which leaves the topology defenceless against a caller's extraction bugs: today
//! [`TemporalGraph::add_fact`](crate::TemporalGraph::add_fact) will happily write
//! `lives_in(a_city, a_person)` with the endpoints reversed, and nothing ever notices.
//!
//! A [`FactPolicy`] is consulted at the write boundary and may refuse. It is a data-integrity
//! constraint of the same kind as the bi-temporal frame — not an inference step — so it does not
//! cross the crate's scope line: a policy sees rows, never prompts, and calls no model.
//!
//! [`Schema`](crate::Schema) is the policy this crate ships. The trait is public because the
//! interesting constraints are caller-specific: a tenant boundary, an id namespace, a
//! "relations may only be asserted between entities that share a property" rule. Such a policy
//! needs to *read* the graph, so both candidate types expose the reads a decision plausibly
//! needs rather than handing over only the row being written.

use crate::error::GraphError;
use crate::model::{AsOf, Direction, Entity, Fact};
use crate::topology::TopologyStore;

/// Why a [`FactPolicy`] refused a write.
///
/// Deliberately NOT [`GraphError`]: a policy that could return the whole error enum could
/// return [`GraphError::Topology`] or [`GraphError::Internal`], which mean "the store failed"
/// and "this crate has a bug". A third-party policy must not be able to impersonate either, so
/// its only channel is a reason string, which the crate maps to
/// [`GraphError::PolicyViolation`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyRejection {
    reason: String,
}

impl PolicyRejection {
    /// Refuse the write, explaining why. The reason reaches the caller verbatim inside
    /// [`GraphError::PolicyViolation`], so write it for whoever reads the log.
    pub fn new(reason: impl Into<String>) -> Self {
        PolicyRejection { reason: reason.into() }
    }

    /// The explanation given at construction.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl std::fmt::Display for PolicyRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.reason)
    }
}

impl std::error::Error for PolicyRejection {}

impl From<PolicyRejection> for GraphError {
    fn from(r: PolicyRejection) -> Self {
        GraphError::PolicyViolation(r.reason)
    }
}

/// The result of an admission decision.
pub type PolicyResult = std::result::Result<(), PolicyRejection>;

/// A fact about to be written, with its endpoints already resolved.
///
/// `#[non_exhaustive]` so later fields are not a breaking change; construct nothing yourself —
/// the graph builds it.
#[non_exhaustive]
pub struct FactCandidate<'a> {
    fact: &'a Fact,
    src: &'a Entity,
    dst: &'a Entity,
    invalidates: &'a [String],
    store: &'a TopologyStore,
}

impl<'a> FactCandidate<'a> {
    pub(crate) fn new(
        fact: &'a Fact,
        src: &'a Entity,
        dst: &'a Entity,
        invalidates: &'a [String],
        store: &'a TopologyStore,
    ) -> Self {
        FactCandidate { fact, src, dst, invalidates, store }
    }

    /// The fact as it would be stored, with its four timestamps already set.
    pub fn fact(&self) -> &Fact {
        self.fact
    }

    /// The relation name — the same string as `self.fact().relation`, offered directly because
    /// it is what most policies branch on.
    pub fn relation(&self) -> &str {
        &self.fact.relation
    }

    /// The stored source entity. Its `entity_type` is the kind a domain check reads.
    pub fn src(&self) -> &Entity {
        self.src
    }

    /// The stored target entity.
    pub fn dst(&self) -> &Entity {
        self.dst
    }

    /// Ids of the facts this write would supersede (the `invalidates` argument).
    pub fn invalidates(&self) -> &[String] {
        self.invalidates
    }

    /// Read the facts already incident to `node_ids` — the escape hatch for policies that must
    /// look at the neighbourhood, such as a cardinality or an acyclicity rule. Reads the
    /// authoritative topology, so it sees no part of the write in progress.
    pub fn facts_for(
        &self,
        node_ids: &[String],
        direction: Direction,
        relation: Option<&str>,
        as_of: AsOf,
    ) -> crate::error::Result<Vec<Fact>> {
        self.store.facts_for(node_ids, direction, relation, as_of)
    }

    /// Read a stored entity by id.
    pub fn entity(&self, id: &str) -> crate::error::Result<Option<Entity>> {
        self.store.get_entity(id)
    }
}

/// An entity about to be written, and the row it would replace.
#[non_exhaustive]
pub struct EntityCandidate<'a> {
    entity: &'a Entity,
    existing: Option<&'a Entity>,
    store: &'a TopologyStore,
}

impl<'a> EntityCandidate<'a> {
    pub(crate) fn new(
        entity: &'a Entity,
        existing: Option<&'a Entity>,
        store: &'a TopologyStore,
    ) -> Self {
        EntityCandidate { entity, existing, store }
    }

    /// The entity as it would be stored.
    pub fn entity(&self) -> &Entity {
        self.entity
    }

    /// The kind being asserted — `self.entity().entity_type`.
    pub fn kind(&self) -> &str {
        &self.entity.entity_type
    }

    /// The row this upsert would replace, if the id already exists. `None` on a first write.
    ///
    /// A policy that wants to freeze an entity's kind compares this against [`Self::kind`]:
    /// the crate itself permits retyping, because the store has always permitted it
    /// (`ON CONFLICT(id) DO UPDATE SET type=excluded.type`) and existing databases depend on it.
    pub fn existing(&self) -> Option<&Entity> {
        self.existing
    }

    /// Read the facts already incident to `node_ids` — so a policy can refuse a retype that
    /// would strand facts admitted under the old kind.
    pub fn facts_for(
        &self,
        node_ids: &[String],
        direction: Direction,
        relation: Option<&str>,
        as_of: AsOf,
    ) -> crate::error::Result<Vec<Fact>> {
        self.store.facts_for(node_ids, direction, relation, as_of)
    }
}

/// A caller-supplied admission rule, consulted before a write reaches the store.
///
/// `Send + Sync` for the same reason [`Embedder`](crate::Embedder) is: the graph owns it behind
/// a box and must stay movable across threads.
///
/// Both methods are consulted only when a policy is installed
/// ([`TemporalGraph::set_fact_policy`](crate::TemporalGraph::set_fact_policy)); a graph with no
/// policy behaves exactly as it did before this trait existed.
pub trait FactPolicy: Send + Sync {
    /// Admit or refuse a fact. Called after the fact's shape has been validated and its
    /// endpoints loaded, and before anything is written or the index is marked dirty — so a
    /// refusal costs no SQL write and no reindex.
    fn admit_fact(&self, candidate: &FactCandidate<'_>) -> PolicyResult;

    /// Admit or refuse an entity upsert. Defaulted to permit, because a fact-only policy is a
    /// reasonable thing to write — though note that a domain/range rule is only as trustworthy
    /// as the kinds it reads, so a policy that constrains facts usually wants to constrain
    /// kinds too.
    fn admit_entity(&self, candidate: &EntityCandidate<'_>) -> PolicyResult {
        let _ = candidate;
        Ok(())
    }
}
