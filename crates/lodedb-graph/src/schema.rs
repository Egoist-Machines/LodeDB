//! A closed-world entity/relation schema — Graphiti's `edge_type_map`, formalised and enforced.
//!
//! Graphiti carries the same information as
//! `edge_type_map: dict[tuple[str, str], list[str]]` — (source kind, target kind) to the
//! relation names permitted between them. It flattens that map into **prompt context**
//! (`graphiti_core/utils/maintenance/edge_operations.py:146-197`) and uses it to select which
//! attribute model applies to an extracted edge (`:478`); its graph store never constrains what
//! the model returns. This port kept Graphiti's bi-temporal core and dropped its edge typing
//! entirely, leaving `Entity::entity_type` and `Fact::relation` as unchecked free strings.
//!
//! [`Schema`] restores it on the other side of the boundary: the same vocabulary, enforced as a
//! [`FactPolicy`] at the write funnel, so a caller's extraction bug cannot corrupt the topology.
//! Nothing here is model-facing — no prompt, no grammar, no ontology format is parsed.
//!
//! ```ignore
//! let schema = Schema::try_from(
//!     SchemaDef::new()
//!         .kind(KindDef::root("Person"))
//!         .kind(KindDef::root("Place"))
//!         // Land IS-A Place, so a `Place` range admits a `Land` without restating it.
//!         .kind(KindDef::sub("Land", ["Place"]))
//!         .relation(RelationDef::new("lives_in", ["Person"], ["Place"])),
//! )?;
//! graph.set_fact_policy(Some(Box::new(schema)));
//! ```
//!
//! ## What it does and does not govern
//!
//! Admission is **write-time and non-retroactive**. Installing a schema never re-examines rows
//! already stored, and a legal retype does not re-check facts already incident to the entity —
//! matching Graphiti, whose store never re-derives. A retroactive rule would also let a
//! tightened schema brick `open()`, which re-walks every stored row to repair the index.
//!
//! Subsumption governs the **write** side only. `entities.type` is one string per row, so a
//! `Place`-typed enumeration still will not return a `Land` even though the schema knows
//! `Land` is a `Place`. A subsumption-aware read verb is an additive follow-on, not part of
//! this type.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{GraphError, Result};
use crate::policy::{EntityCandidate, FactCandidate, FactPolicy, PolicyRejection, PolicyResult};

/// One declared entity kind, and the kinds it is a sub-kind of.
///
/// Multiple parents are permitted. The hierarchy has to live here because a `lodedb-graph`
/// entity carries a single `entity_type` string, unlike a Graphiti node, which carries a *list*
/// of labels and so gets subsumption for free (`nodes.py` appends the universal `'Entity'`
/// label, and `edge_operations.py:464-478` cross-products source × target labels against the
/// map). Flattening that to one string is where the hierarchy would otherwise be lost.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KindDef {
    /// The kind name, matched against `Entity::entity_type` verbatim.
    pub name: String,
    /// Direct super-kinds. Every name here must itself be declared as a kind.
    #[serde(default)]
    pub subclass_of: Vec<String>,
}

impl KindDef {
    /// A kind with no super-kind.
    pub fn root(name: impl Into<String>) -> Self {
        KindDef { name: name.into(), subclass_of: Vec::new() }
    }

    /// A kind that is a sub-kind of each name in `parents`.
    pub fn sub<I, S>(name: impl Into<String>, parents: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        KindDef {
            name: name.into(),
            subclass_of: parents.into_iter().map(Into::into).collect(),
        }
    }
}

/// One permitted (source kinds) × relation × (target kinds) rectangle.
///
/// Several `RelationDef`s may share a `name`; admission is a disjunction over them, never a
/// merge. That is what makes a Graphiti `edge_type_map` expressible exactly: one rectangle per
/// map key, so declaring `(Person, r, Org)` and `(Org, r, Person)` does **not** thereby admit
/// `(Person, r, Person)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationDef {
    /// The relation name, matched against `Fact::relation` verbatim.
    pub name: String,
    /// Permitted source kinds. A sub-kind of any listed kind is permitted.
    pub domain: Vec<String>,
    /// Permitted target kinds. A sub-kind of any listed kind is permitted.
    pub range: Vec<String>,
}

impl RelationDef {
    /// A rectangle. Both ends must be non-empty — see [`Schema`] on why there is no wildcard.
    pub fn new<D, R, S1, S2>(name: impl Into<String>, domain: D, range: R) -> Self
    where
        D: IntoIterator<Item = S1>,
        R: IntoIterator<Item = S2>,
        S1: Into<String>,
        S2: Into<String>,
    {
        RelationDef {
            name: name.into(),
            domain: domain.into_iter().map(Into::into).collect(),
            range: range.into_iter().map(Into::into).collect(),
        }
    }
}

/// The declarative form of a schema — what a caller writes, and what crosses a binding as JSON.
///
/// Compile it with [`Schema::try_from`], which validates it and precomputes the subsumption
/// closure. Building is infallible so declaration order never matters; every rule is checked
/// once, at compile.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaDef {
    #[serde(default)]
    pub kinds: Vec<KindDef>,
    #[serde(default)]
    pub relations: Vec<RelationDef>,
}

impl SchemaDef {
    /// An empty declaration. A schema with no kinds admits nothing, which is a usable starting
    /// point but never a useful policy.
    pub fn new() -> Self {
        SchemaDef::default()
    }

    /// Declare a kind.
    pub fn kind(mut self, kind: KindDef) -> Self {
        self.kinds.push(kind);
        self
    }

    /// Declare a permitted rectangle.
    pub fn relation(mut self, relation: RelationDef) -> Self {
        self.relations.push(relation);
        self
    }
}

/// The permitted (domain, range) rectangles declared for one relation name. Several may share
/// a name; admission is a disjunction over them.
type Rectangles = Vec<(BTreeSet<String>, BTreeSet<String>)>;

/// A compiled, enforceable schema. Build with [`Schema::try_from`] on a [`SchemaDef`].
///
/// ## Closed world, no wildcards
///
/// A kind is declared **only** by appearing in `kinds`. Naming a kind in a `domain`, a `range`
/// or a `subclass_of` does not declare it — that is a compile error, which is what stops a
/// mistyped `"Persn"` in a range from silently becoming an admissible entity kind.
///
/// There is likewise no wildcard and no "empty means unrestricted": an empty `domain` or
/// `range` is a compile error. "Any kind" is spelled by declaring a root kind and naming it,
/// exactly as Graphiti's implicit `('Entity', 'Entity')` signature does. So no blank field can
/// silently open a gate.
///
/// The empty kind `""` *is* declarable, because `entities.type` defaults to the empty string
/// and real pre-schema rows carry it; a graph with such rows needs a way to say so.
#[derive(Debug, Clone)]
pub struct Schema {
    /// Every declared kind.
    kinds: BTreeSet<String>,
    /// kind -> {itself} ∪ all transitive super-kinds. Precomputed, so admission is set
    /// membership rather than a walk — the same trick `temporal::encode_ts` uses to turn an
    /// as-of query into a string comparison.
    ancestors: BTreeMap<String, BTreeSet<String>>,
    /// relation name -> its rectangles, as (domain, range) pairs.
    relations: BTreeMap<String, Rectangles>,
}

impl TryFrom<SchemaDef> for Schema {
    type Error = GraphError;

    fn try_from(def: SchemaDef) -> Result<Self> {
        let mut kinds: BTreeSet<String> = BTreeSet::new();
        for k in &def.kinds {
            if !kinds.insert(k.name.clone()) {
                return Err(GraphError::InvalidArgument(format!(
                    "schema declares kind {:?} more than once",
                    k.name
                )));
            }
        }

        // Direct parents, checked against the declared set so a typo cannot invent a kind.
        let mut parents: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for k in &def.kinds {
            for p in &k.subclass_of {
                if !kinds.contains(p) {
                    return Err(GraphError::InvalidArgument(format!(
                        "kind {:?} is declared a subclass of undeclared kind {:?}",
                        k.name, p
                    )));
                }
                if *p == k.name {
                    return Err(GraphError::InvalidArgument(format!(
                        "kind {:?} is declared a subclass of itself",
                        k.name
                    )));
                }
            }
            parents.insert(k.name.clone(), k.subclass_of.clone());
        }

        // Reflexive-transitive closure, with an explicit visited set so a cycle is REJECTED
        // rather than looped. A cyclic hierarchy is a declaration bug and must be loud.
        let mut ancestors: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for name in &kinds {
            let mut seen: BTreeSet<String> = BTreeSet::new();
            let mut stack: Vec<String> = vec![name.clone()];
            while let Some(cur) = stack.pop() {
                if !seen.insert(cur.clone()) {
                    continue;
                }
                if let Some(ps) = parents.get(&cur) {
                    for p in ps {
                        if p == name && seen.contains(name) && cur != *name {
                            return Err(GraphError::InvalidArgument(format!(
                                "schema kind hierarchy is cyclic at {name:?}"
                            )));
                        }
                        stack.push(p.clone());
                    }
                }
            }
            ancestors.insert(name.clone(), seen);
        }
        // A cycle not caught above (A -> B -> A reached from a third kind) shows up as a kind
        // being its own strict ancestor via a parent chain; verify directly.
        for (name, anc) in &ancestors {
            for a in anc {
                if a != name {
                    if let Some(up) = ancestors.get(a) {
                        if up.contains(name) {
                            return Err(GraphError::InvalidArgument(format!(
                                "schema kind hierarchy is cyclic between {name:?} and {a:?}"
                            )));
                        }
                    }
                }
            }
        }

        let mut relations: BTreeMap<String, Rectangles> = BTreeMap::new();
        for r in &def.relations {
            if r.name.trim().is_empty() {
                return Err(GraphError::InvalidArgument(
                    "schema declares a relation with an empty name".into(),
                ));
            }
            if r.domain.is_empty() || r.range.is_empty() {
                return Err(GraphError::InvalidArgument(format!(
                    "relation {:?} must declare a non-empty domain and range; there is no \
                     wildcard — name a declared root kind instead",
                    r.name
                )));
            }
            for side in [&r.domain, &r.range] {
                for k in side {
                    if !kinds.contains(k) {
                        return Err(GraphError::InvalidArgument(format!(
                            "relation {:?} names undeclared kind {:?}",
                            r.name, k
                        )));
                    }
                }
            }
            relations.entry(r.name.clone()).or_default().push((
                r.domain.iter().cloned().collect(),
                r.range.iter().cloned().collect(),
            ));
        }

        Ok(Schema { kinds, ancestors, relations })
    }
}

impl Schema {
    /// Whether `name` is a declared kind.
    pub fn declares_kind(&self, name: &str) -> bool {
        self.kinds.contains(name)
    }

    /// Whether `name` is a declared relation.
    pub fn declares_relation(&self, name: &str) -> bool {
        self.relations.contains_key(name)
    }

    /// Whether `relation` may hold from `src_kind` to `dst_kind`, honouring subsumption.
    ///
    /// Public because a caller's extraction layer wants to *filter* candidates before writing,
    /// not merely catch the error afterwards — the difference between an integrity constraint
    /// and an error generator.
    pub fn permits(&self, src_kind: &str, relation: &str, dst_kind: &str) -> bool {
        let Some(rects) = self.relations.get(relation) else {
            return false;
        };
        let Some(src_anc) = self.ancestors.get(src_kind) else {
            return false;
        };
        let Some(dst_anc) = self.ancestors.get(dst_kind) else {
            return false;
        };
        rects.iter().any(|(domain, range)| {
            !src_anc.is_disjoint(domain) && !dst_anc.is_disjoint(range)
        })
    }

    /// Every declared kind, sorted.
    pub fn kinds(&self) -> impl Iterator<Item = &str> {
        self.kinds.iter().map(String::as_str)
    }

    /// Every declared relation name, sorted.
    pub fn relations(&self) -> impl Iterator<Item = &str> {
        self.relations.keys().map(String::as_str)
    }
}

impl FactPolicy for Schema {
    fn admit_fact(&self, candidate: &FactCandidate<'_>) -> PolicyResult {
        let relation = candidate.relation();
        let src_kind = &candidate.src().entity_type;
        let dst_kind = &candidate.dst().entity_type;
        if !self.declares_relation(relation) {
            return Err(PolicyRejection::new(format!(
                "relation {relation:?} is not declared by the schema"
            )));
        }
        if !self.declares_kind(src_kind) {
            return Err(PolicyRejection::new(format!(
                "source entity {:?} has undeclared kind {src_kind:?}",
                candidate.src().id
            )));
        }
        if !self.declares_kind(dst_kind) {
            return Err(PolicyRejection::new(format!(
                "target entity {:?} has undeclared kind {dst_kind:?}",
                candidate.dst().id
            )));
        }
        if !self.permits(src_kind, relation, dst_kind) {
            return Err(PolicyRejection::new(format!(
                "relation {relation:?} is not permitted from kind {src_kind:?} to kind \
                 {dst_kind:?}"
            )));
        }
        Ok(())
    }

    fn admit_entity(&self, candidate: &EntityCandidate<'_>) -> PolicyResult {
        let kind = candidate.kind();
        if self.declares_kind(kind) {
            Ok(())
        } else {
            Err(PolicyRejection::new(format!(
                "entity {:?} has undeclared kind {kind:?}",
                candidate.entity().id
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn places() -> Schema {
        Schema::try_from(
            SchemaDef::new()
                .kind(KindDef::root("Person"))
                .kind(KindDef::root("Place"))
                .kind(KindDef::sub("Land", ["Place"]))
                .kind(KindDef::root("Group"))
                .relation(RelationDef::new("lives_in", ["Person"], ["Place"]))
                .relation(RelationDef::new("member_of", ["Person"], ["Group"])),
        )
        .expect("schema compiles")
    }

    #[test]
    fn a_declared_rectangle_is_permitted_and_its_converse_is_not() {
        let s = places();
        assert!(s.permits("Person", "lives_in", "Place"));
        assert!(!s.permits("Place", "lives_in", "Person"));
    }

    #[test]
    fn a_subkind_satisfies_its_parents_range_without_restating_it() {
        let s = places();
        // Land was never named in lives_in's range; it qualifies by being a Place.
        assert!(s.permits("Person", "lives_in", "Land"));
    }

    #[test]
    fn subsumption_does_not_run_downhill() {
        let s = places();
        // A Place is not a Land, so a range of Land must not admit a bare Place.
        let s2 = Schema::try_from(
            SchemaDef::new()
                .kind(KindDef::root("Person"))
                .kind(KindDef::root("Place"))
                .kind(KindDef::sub("Land", ["Place"]))
                .relation(RelationDef::new("stewards", ["Person"], ["Land"])),
        )
        .unwrap();
        assert!(s2.permits("Person", "stewards", "Land"));
        assert!(!s2.permits("Person", "stewards", "Place"));
        let _ = s;
    }

    #[test]
    fn rectangles_sharing_a_name_are_disjoined_never_merged() {
        let s = Schema::try_from(
            SchemaDef::new()
                .kind(KindDef::root("Person"))
                .kind(KindDef::root("Org"))
                .relation(RelationDef::new("pays", ["Person"], ["Org"]))
                .relation(RelationDef::new("pays", ["Org"], ["Person"])),
        )
        .unwrap();
        assert!(s.permits("Person", "pays", "Org"));
        assert!(s.permits("Org", "pays", "Person"));
        // The merge would have admitted these; the disjunction must not.
        assert!(!s.permits("Person", "pays", "Person"));
        assert!(!s.permits("Org", "pays", "Org"));
    }

    #[test]
    fn an_undeclared_kind_in_a_range_is_a_compile_error() {
        let err = Schema::try_from(
            SchemaDef::new()
                .kind(KindDef::root("Person"))
                .relation(RelationDef::new("lives_in", ["Person"], ["Persn"])),
        );
        assert!(matches!(err, Err(GraphError::InvalidArgument(_))), "a typo must not declare a kind");
    }

    #[test]
    fn an_empty_endpoint_list_is_a_compile_error_not_a_wildcard() {
        let empty: [&str; 0] = [];
        let err = Schema::try_from(
            SchemaDef::new()
                .kind(KindDef::root("Person"))
                .relation(RelationDef::new("knows", ["Person"], empty)),
        );
        assert!(matches!(err, Err(GraphError::InvalidArgument(_))));
    }

    #[test]
    fn a_cyclic_hierarchy_is_rejected_rather_than_looping() {
        let err = Schema::try_from(
            SchemaDef::new()
                .kind(KindDef::sub("A", ["B"]))
                .kind(KindDef::sub("B", ["A"])),
        );
        assert!(matches!(err, Err(GraphError::InvalidArgument(_))), "got {err:?}");
    }

    #[test]
    fn the_empty_kind_is_declarable_because_stored_rows_carry_it() {
        let s = Schema::try_from(
            SchemaDef::new()
                .kind(KindDef::root(""))
                .kind(KindDef::root("Person"))
                .relation(RelationDef::new("knows", ["Person"], ["Person"])),
        )
        .expect("the empty kind is legal");
        assert!(s.declares_kind(""));
        assert!(!s.permits("", "knows", "Person"));
    }

    #[test]
    fn a_schema_def_round_trips_through_json() {
        let def = SchemaDef::new()
            .kind(KindDef::sub("Land", ["Place"]))
            .kind(KindDef::root("Place"))
            .relation(RelationDef::new("part_of", ["Land"], ["Land"]));
        let json = serde_json::to_string(&def).unwrap();
        let back: SchemaDef = serde_json::from_str(&json).unwrap();
        assert_eq!(def, back);
    }
}
