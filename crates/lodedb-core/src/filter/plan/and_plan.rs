use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde_json::Value;

use crate::error::CoreError;
use crate::filter::doc_set::DocSet;
use crate::filter::field_index::FieldIndex;
use crate::filter::predicate::compare_ordered;

use super::{
    count_entry, exact_matching_count, invalid, materialize_entry_probe, materialize_probe,
    probe_entry, probe_info, probe_operator, MetadataFilterPlan, MetadataPredicatePlan, ProbeInfo,
    ProbeSide, BLIND_DEFER_DEN, BLIND_DEFER_NUM, KNOCKOUT_VS_DENYLIST, REFINE_COST_RATIO,
};

#[derive(Debug, Clone, Copy)]
enum AndConjunct<'a> {
    Entry { key: &'a str, spec: &'a Value },
    Filter(&'a Value),
}

#[derive(Debug, Clone, Copy)]
struct ConjunctStats<'a> {
    conjunct: AndConjunct<'a>,
    matching_count: Option<usize>,
    matching_probe: Option<ProbeInfo>,
    failing_count: Option<usize>,
    failing_probe: Option<ProbeInfo>,
    matching_probe_exact: bool,
}

pub(super) fn plan_conjunctive_filter(
    metadata_filter: &Value,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<Option<MetadataFilterPlan>, CoreError> {
    let Some(conjuncts) = conjunctive_terms(metadata_filter)? else {
        return Ok(None);
    };
    if conjuncts.len() < 2 {
        return Ok(None);
    }

    let stats = conjuncts
        .iter()
        .map(|conjunct| conjunct_stats(*conjunct, fields, all_docs))
        .collect::<Result<Vec<_>, _>>()?;

    if let Some(plan) = plan_selective_anchor(&stats, fields, all_docs)? {
        return Ok(Some(plan));
    }
    if let Some(plan) = plan_exact_failing_denylist(&stats, fields, all_docs)? {
        return Ok(Some(plan));
    }
    if let Some(plan) = plan_all_unselective_denylist(&stats, fields, all_docs)? {
        return Ok(Some(plan));
    }
    if let Some(plan) = plan_complement_knockout(&stats, fields, all_docs)? {
        return Ok(Some(plan));
    }

    Ok(None)
}

fn conjunctive_terms(filter: &Value) -> Result<Option<Vec<AndConjunct<'_>>>, CoreError> {
    let node = filter
        .as_object()
        .ok_or_else(|| invalid("validated filter must be an object"))?;
    if node.len() == 1 {
        if let Some(spec) = node.get("$and") {
            let subs = spec
                .as_array()
                .ok_or_else(|| invalid("validated $and must be a list"))?;
            return Ok(Some(subs.iter().map(AndConjunct::Filter).collect()));
        }
        return Ok(None);
    }
    Ok(Some(
        node.iter()
            .map(|(key, spec)| AndConjunct::Entry {
                key: key.as_str(),
                spec,
            })
            .collect(),
    ))
}

fn conjunct_stats<'a>(
    conjunct: AndConjunct<'a>,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<ConjunctStats<'a>, CoreError> {
    let matching_count = conjunct_matching_count(conjunct, fields, all_docs)?;
    Ok(ConjunctStats {
        conjunct,
        matching_count,
        matching_probe: conjunct_probe(conjunct, ProbeSide::Matching, fields, all_docs)?,
        failing_count: matching_count.map(|count| all_docs.len().saturating_sub(count)),
        failing_probe: conjunct_probe(conjunct, ProbeSide::Failing, fields, all_docs)?,
        matching_probe_exact: conjunct_matching_probe_is_exact(conjunct, fields, all_docs)?,
    })
}

fn plan_selective_anchor(
    stats: &[ConjunctStats<'_>],
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<Option<MetadataFilterPlan>, CoreError> {
    let Some((anchor_index, anchor)) = stats
        .iter()
        .enumerate()
        .filter(|(_, stat)| {
            stat.matching_probe.is_some()
                && legacy_single_conjunct_materializes(stat, all_docs.len())
        })
        .min_by_key(|(_, stat)| stat.matching_probe.expect("filtered above").count)
    else {
        return Ok(None);
    };

    let anchor_count = anchor.matching_probe.expect("filtered above").count;
    let alternative = stats
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != anchor_index)
        .map(|(_, stat)| matching_materialization_cost(stat, all_docs.len()))
        .min()
        .unwrap_or(all_docs.len());

    if anchor_count.saturating_mul(REFINE_COST_RATIO) >= alternative {
        return Ok(None);
    }

    Ok(Some(MetadataFilterPlan::FilteredCandidates(
        materialize_conjunct_probe(anchor.conjunct, ProbeSide::Matching, fields, all_docs)?,
    )))
}

fn plan_all_unselective_denylist(
    stats: &[ConjunctStats<'_>],
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<Option<MetadataFilterPlan>, CoreError> {
    if !stats
        .iter()
        .all(|stat| failing_refinement_is_small(stat, all_docs.len()))
    {
        return Ok(None);
    }

    let mut failing_candidates = DocSet::new();
    for stat in stats {
        failing_candidates.extend(materialize_conjunct_probe(
            stat.conjunct,
            ProbeSide::Failing,
            fields,
            all_docs,
        )?);
    }
    Ok(Some(MetadataFilterPlan::Predicate(MetadataPredicatePlan {
        matching_count: None,
        failing_count: None,
        failing_candidates: Some(failing_candidates),
        failing_exact: false,
    })))
}

fn plan_exact_failing_denylist(
    stats: &[ConjunctStats<'_>],
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<Option<MetadataFilterPlan>, CoreError> {
    let total = all_docs.len();
    let mut choices = Vec::new();
    for (anchor_index, anchor) in stats.iter().enumerate() {
        if !anchor.matching_probe_exact {
            continue;
        }
        let Some(anchor_count) = anchor.matching_probe.map(|probe| probe.count) else {
            continue;
        };
        let mut failing_bound = total.saturating_sub(anchor_count);
        let mut bounded = true;
        for (index, stat) in stats.iter().enumerate() {
            if index == anchor_index {
                continue;
            }
            let Some(probe) = stat.failing_probe else {
                bounded = false;
                break;
            };
            failing_bound = failing_bound.saturating_add(probe.count);
        }
        if !bounded {
            continue;
        }
        if failing_bound.saturating_mul(KNOCKOUT_VS_DENYLIST) < anchor_count {
            choices.push((failing_bound, std::cmp::Reverse(anchor_count), anchor_index));
        }
    }
    choices.sort();

    for (_, _, anchor_index) in choices {
        let anchor = &stats[anchor_index];
        let Some(anchor_posting) =
            matching_posting_for_conjunct(anchor.conjunct, fields, all_docs)?
        else {
            continue;
        };
        // complement(anchor) and every entry-shaped conjunct's failing set are
        // already exact failures of the whole AND, so re-proving them through
        // the full predicate is pure waste (measured 423 ms on a 95%-anchor
        // recall shape refining ~15k members). Only superset-shaped failing
        // materializations (a multi-branch $or's cheapest branch, $not) need
        // refinement, and only against their own conjunct: failing one
        // conjunct fails the AND, and a member failing a different conjunct
        // is already covered by that conjunct's exact set.
        let mut failing = complement_posting(all_docs, anchor_posting.as_doc_set());
        for stat in stats
            .iter()
            .enumerate()
            .filter_map(|(index, stat)| (index != anchor_index).then_some(stat))
        {
            let superset = materialize_conjunct_probe(
                stat.conjunct,
                ProbeSide::Failing,
                fields,
                all_docs,
            )?;
            if conjunct_failing_materialization_is_exact(stat.conjunct) {
                failing.extend(superset);
                continue;
            }
            for document_id in superset {
                if !indexed_conjunct_matches_document(&document_id, stat.conjunct, fields)? {
                    failing.insert(document_id);
                }
            }
        }
        return Ok(Some(MetadataFilterPlan::Predicate(MetadataPredicatePlan {
            matching_count: Some(total.saturating_sub(failing.len())),
            failing_count: Some(failing.len()),
            failing_candidates: Some(failing),
            failing_exact: true,
        })));
    }

    Ok(None)
}

fn plan_complement_knockout(
    stats: &[ConjunctStats<'_>],
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<Option<MetadataFilterPlan>, CoreError> {
    let Some((anchor_index, anchor)) = stats
        .iter()
        .enumerate()
        .filter(|(_, stat)| stat.matching_probe.is_some() && stat.matching_probe_exact)
        .min_by_key(|(_, stat)| stat.matching_probe.expect("filtered above").count)
    else {
        return Ok(None);
    };

    if stats
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != anchor_index)
        .any(|(_, stat)| !failing_refinement_is_small(stat, all_docs.len()))
    {
        return Ok(None);
    }

    let anchor_docs =
        materialize_conjunct_probe(anchor.conjunct, ProbeSide::Matching, fields, all_docs)?;
    let mut failing_candidates = DocSet::new();
    for stat in stats
        .iter()
        .enumerate()
        .filter_map(|(index, stat)| (index != anchor_index).then_some(stat))
    {
        failing_candidates.extend(materialize_conjunct_probe(
            stat.conjunct,
            ProbeSide::Failing,
            fields,
            all_docs,
        )?);
    }

    Ok(Some(MetadataFilterPlan::AnchoredDenylist {
        anchor: anchor_docs,
        failing_candidates,
    }))
}

fn matching_materialization_cost(stat: &ConjunctStats<'_>, total: usize) -> usize {
    stat.matching_count
        .or(stat.matching_probe.map(|probe| probe.count))
        .unwrap_or(total)
}

fn legacy_single_conjunct_materializes(stat: &ConjunctStats<'_>, total: usize) -> bool {
    if stat.matching_count == Some(0) || stat.matching_probe.is_some_and(|probe| probe.count == 0) {
        return true;
    }
    if stat.matching_count == Some(total)
        || stat.failing_probe.is_some_and(|probe| probe.count == 0)
    {
        return false;
    }
    if let Some(failing_probe) = stat.failing_probe {
        let matching_cost = matching_materialization_cost(stat, total);
        if failing_probe.count.saturating_mul(REFINE_COST_RATIO) < matching_cost {
            return false;
        }
    }
    if let Some(count) = stat.matching_count {
        if count.saturating_mul(BLIND_DEFER_DEN) > total.saturating_mul(BLIND_DEFER_NUM) {
            return false;
        }
    }
    true
}

fn failing_side_is_small(count: usize, total: usize) -> bool {
    // BLIND_DEFER_NUM / BLIND_DEFER_DEN says blind deferral starts winning once
    // the matching side is above 2/3 of the corpus. The equivalent failing-side
    // bound for an AND denylist is below the remaining 1/3.
    count == 0
        || count.saturating_mul(BLIND_DEFER_DEN)
            < total.saturating_mul(BLIND_DEFER_DEN - BLIND_DEFER_NUM)
}

fn failing_refinement_is_small(stat: &ConjunctStats<'_>, total: usize) -> bool {
    let Some(probe) = stat.failing_probe else {
        return false;
    };
    let count = probe.count.max(stat.failing_count.unwrap_or(0));
    failing_side_is_small(count, total)
}

fn conjunct_matching_count(
    conjunct: AndConjunct<'_>,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<Option<usize>, CoreError> {
    match conjunct {
        AndConjunct::Entry { key, spec } => count_entry(key, spec, fields, all_docs),
        AndConjunct::Filter(filter) => exact_matching_count(filter, fields, all_docs),
    }
}

fn conjunct_probe(
    conjunct: AndConjunct<'_>,
    side: ProbeSide,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<Option<ProbeInfo>, CoreError> {
    match conjunct {
        AndConjunct::Entry { key, spec } => probe_entry(key, spec, side, fields, all_docs),
        AndConjunct::Filter(filter) => probe_info(filter, side, fields, all_docs),
    }
}

fn materialize_conjunct_probe(
    conjunct: AndConjunct<'_>,
    side: ProbeSide,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<DocSet, CoreError> {
    match conjunct {
        AndConjunct::Entry { key, spec } => {
            materialize_entry_probe(key, spec, side, fields, all_docs)?
                .ok_or_else(|| invalid("validated conjunct probe disappeared"))
        }
        AndConjunct::Filter(filter) => materialize_probe(filter, side, fields, all_docs),
    }
}

fn conjunct_matching_probe_is_exact(
    conjunct: AndConjunct<'_>,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<bool, CoreError> {
    match conjunct {
        AndConjunct::Entry { key, spec } => {
            entry_matching_probe_is_exact(key, spec, fields, all_docs)
        }
        AndConjunct::Filter(filter) => filter_matching_probe_is_exact(filter, fields, all_docs),
    }
}

fn filter_matching_probe_is_exact(
    filter: &Value,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<bool, CoreError> {
    let node = filter
        .as_object()
        .ok_or_else(|| invalid("validated filter must be an object"))?;
    if node.is_empty() {
        return Ok(true);
    }
    if node.len() != 1 {
        return Ok(false);
    }
    let (key, spec) = node.iter().next().expect("one entry checked above");
    entry_matching_probe_is_exact(key, spec, fields, all_docs)
}

fn entry_matching_probe_is_exact(
    key: &str,
    spec: &Value,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<bool, CoreError> {
    match key {
        "$and" => {
            let subs = spec
                .as_array()
                .ok_or_else(|| invalid("validated $and must be a list"))?;
            if subs.len() <= 1 {
                return subs.first().map_or(Ok(true), |sub| {
                    filter_matching_probe_is_exact(sub, fields, all_docs)
                });
            }
            Ok(false)
        }
        "$or" => {
            let subs = spec
                .as_array()
                .ok_or_else(|| invalid("validated $or must be a list"))?;
            // A multi-branch $or's matching probe SUMS branch counts, and
            // overlapping postings overcount, so the sum is a bound, never
            // an exact cardinality; treating it as exact mis-prices anchors.
            if subs.len() != 1 {
                return Ok(false);
            }
            filter_matching_probe_is_exact(&subs[0], fields, all_docs)
        }
        "$not" => Ok(false),
        field => field_matching_probe_is_exact(field, spec, fields, all_docs),
    }
}

fn field_matching_probe_is_exact(
    field: &str,
    spec: &Value,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<bool, CoreError> {
    let empty = FieldIndex::default();
    let index = fields.get(field).unwrap_or(&empty);
    if spec.as_str().is_some() {
        return Ok(true);
    }
    let operators = spec
        .as_object()
        .ok_or_else(|| invalid("validated field spec must be a string or operator map"))?;
    if operators.len() != 1 {
        return Ok(false);
    }
    let (op, operand) = operators.iter().next().expect("one operator checked above");
    Ok(probe_operator(op, operand, index, ProbeSide::Matching, all_docs.len())?.is_some())
}

enum MatchingPosting<'a> {
    Borrowed(&'a DocSet),
    Owned(DocSet),
}

impl<'a> MatchingPosting<'a> {
    fn as_doc_set(&self) -> &DocSet {
        match self {
            Self::Borrowed(docs) => docs,
            Self::Owned(docs) => docs,
        }
    }
}

fn matching_posting_for_conjunct<'a>(
    conjunct: AndConjunct<'_>,
    fields: &'a BTreeMap<String, FieldIndex>,
    all_docs: &'a DocSet,
) -> Result<Option<MatchingPosting<'a>>, CoreError> {
    match conjunct {
        AndConjunct::Entry { key, spec } => matching_posting_for_entry(key, spec, fields, all_docs),
        AndConjunct::Filter(filter) => matching_posting_for_filter(filter, fields, all_docs),
    }
}

fn matching_posting_for_filter<'a>(
    filter: &Value,
    fields: &'a BTreeMap<String, FieldIndex>,
    all_docs: &'a DocSet,
) -> Result<Option<MatchingPosting<'a>>, CoreError> {
    let node = filter
        .as_object()
        .ok_or_else(|| invalid("validated filter must be an object"))?;
    if node.is_empty() {
        return Ok(Some(MatchingPosting::Borrowed(all_docs)));
    }
    if node.len() != 1 {
        return Ok(None);
    }
    let (key, spec) = node.iter().next().expect("one entry checked above");
    matching_posting_for_entry(key, spec, fields, all_docs)
}

fn matching_posting_for_entry<'a>(
    key: &str,
    spec: &Value,
    fields: &'a BTreeMap<String, FieldIndex>,
    all_docs: &'a DocSet,
) -> Result<Option<MatchingPosting<'a>>, CoreError> {
    match key {
        "$and" => {
            let subs = spec
                .as_array()
                .ok_or_else(|| invalid("validated $and must be a list"))?;
            match subs.as_slice() {
                [] => Ok(Some(MatchingPosting::Borrowed(all_docs))),
                [sub] => matching_posting_for_filter(sub, fields, all_docs),
                _ => Ok(None),
            }
        }
        "$or" => {
            let subs = spec
                .as_array()
                .ok_or_else(|| invalid("validated $or must be a list"))?;
            let mut docs = DocSet::new();
            for sub in subs {
                let Some(posting) = matching_posting_for_filter(sub, fields, all_docs)? else {
                    return Ok(None);
                };
                docs.extend(posting.as_doc_set().iter().cloned());
            }
            Ok(Some(MatchingPosting::Owned(docs)))
        }
        "$not" => Ok(None),
        field => matching_posting_for_field(field, spec, fields, all_docs),
    }
}

fn matching_posting_for_field<'a>(
    field: &str,
    spec: &Value,
    fields: &'a BTreeMap<String, FieldIndex>,
    all_docs: &'a DocSet,
) -> Result<Option<MatchingPosting<'a>>, CoreError> {
    let index = fields.get(field);
    if let Some(expected) = spec.as_str() {
        return Ok(Some(match index.and_then(|index| index.value_docs.get(expected)) {
            Some(docs) => MatchingPosting::Borrowed(docs),
            None => MatchingPosting::Owned(DocSet::new()),
        }));
    }
    let operators = spec
        .as_object()
        .ok_or_else(|| invalid("validated field spec must be a string or operator map"))?;
    if operators.len() != 1 {
        return Ok(None);
    }
    let (op, operand) = operators.iter().next().expect("one operator checked above");
    Ok(match op.as_str() {
        "$eq" => {
            let value = operand
                .as_str()
                .ok_or_else(|| invalid("validated $eq operand must be a string"))?;
            Some(match index.and_then(|index| index.value_docs.get(value)) {
                Some(docs) => MatchingPosting::Borrowed(docs),
                None => MatchingPosting::Owned(DocSet::new()),
            })
        }
        "$in" => {
            let mut docs = DocSet::new();
            let values = operand
                .as_array()
                .ok_or_else(|| invalid("validated $in operand must be a list"))?;
            if let Some(index) = index {
                for value in values {
                    let value = value
                        .as_str()
                        .ok_or_else(|| invalid("validated $in values must be strings"))?;
                    if let Some(value_docs) = index.value_docs.get(value) {
                        docs.extend(value_docs.iter().cloned());
                    }
                }
            }
            Some(MatchingPosting::Owned(docs))
        }
        "$exists" => {
            let exists = operand
                .as_bool()
                .ok_or_else(|| invalid("validated $exists operand must be a boolean"))?;
            if exists {
                Some(match index {
                    Some(index) => MatchingPosting::Borrowed(&index.docs),
                    None => MatchingPosting::Owned(DocSet::new()),
                })
            } else if index.is_some_and(|index| index.docs.len() == all_docs.len()) {
                Some(MatchingPosting::Owned(DocSet::new()))
            } else {
                None
            }
        }
        "$ne" => {
            let value = operand
                .as_str()
                .ok_or_else(|| invalid("validated $ne operand must be a string"))?;
            if index
                .and_then(|index| index.value_docs.get(value))
                .is_some_and(|docs| docs.len() == all_docs.len())
            {
                Some(MatchingPosting::Owned(DocSet::new()))
            } else {
                None
            }
        }
        "$nin" => {
            let values = operand
                .as_array()
                .ok_or_else(|| invalid("validated $nin operand must be a list"))?;
            let mut excluded = DocSet::new();
            if let Some(index) = index {
                for value in values {
                    let value = value
                        .as_str()
                        .ok_or_else(|| invalid("validated $nin values must be strings"))?;
                    if let Some(value_docs) = index.value_docs.get(value) {
                        excluded.extend(value_docs.iter().cloned());
                    }
                }
            }
            if excluded.len() == all_docs.len() {
                Some(MatchingPosting::Owned(DocSet::new()))
            } else {
                None
            }
        }
        ordered if crate::filter::ast::is_ordered_operator(ordered) => {
            let operand = operand
                .as_str()
                .ok_or_else(|| invalid("validated ordered operand must be a string"))?;
            Some(MatchingPosting::Owned(index.map_or_else(DocSet::new, |index| {
                index.resolve_ordered(ordered, operand)
            })))
        }
        _ => return Err(invalid("unsupported validated operator")),
    })
}

fn complement_posting(all_docs: &DocSet, posting: &DocSet) -> DocSet {
    let mut complement = DocSet::new();
    let mut posting_iter = posting.iter();
    let mut current = posting_iter.next();
    for document_id in all_docs {
        while let Some(posted_id) = current {
            match posted_id.as_str().cmp(document_id.as_str()) {
                Ordering::Less => current = posting_iter.next(),
                Ordering::Equal => break,
                Ordering::Greater => break,
            }
        }
        if current.is_some_and(|posted_id| posted_id == document_id) {
            current = posting_iter.next();
        } else {
            complement.insert(document_id.clone());
        }
    }
    complement
}

/// True when a conjunct's failing-side materialization is the exact failing
/// set, not a superset. Field entries qualify: each operator's failing
/// materialization is a posting set or a proven-empty set, and an operator
/// map unions exact per-operator sets. A multi-branch $or materializes only
/// its cheapest branch's failing side (a superset), and $not swaps to a
/// bounding matching probe, so both need refinement.
fn conjunct_failing_materialization_is_exact(conjunct: AndConjunct<'_>) -> bool {
    match conjunct {
        AndConjunct::Entry { key, spec } => entry_failing_materialization_is_exact(key, spec),
        AndConjunct::Filter(filter) => filter_failing_materialization_is_exact(filter),
    }
}

fn filter_failing_materialization_is_exact(filter: &Value) -> bool {
    filter.as_object().is_some_and(|node| {
        node.iter()
            .all(|(key, spec)| entry_failing_materialization_is_exact(key, spec))
    })
}

fn entry_failing_materialization_is_exact(key: &str, spec: &Value) -> bool {
    match key {
        "$and" => spec
            .as_array()
            .is_some_and(|subs| subs.iter().all(filter_failing_materialization_is_exact)),
        "$or" => spec.as_array().is_some_and(|subs| {
            subs.len() == 1 && filter_failing_materialization_is_exact(&subs[0])
        }),
        "$not" => false,
        _ => true,
    }
}

/// Evaluates one conjunct for one document via the field indexes.
fn indexed_conjunct_matches_document(
    document_id: &str,
    conjunct: AndConjunct<'_>,
    fields: &BTreeMap<String, FieldIndex>,
) -> Result<bool, CoreError> {
    match conjunct {
        AndConjunct::Entry { key, spec } => {
            indexed_entry_matches_document(document_id, key, spec, fields)
        }
        AndConjunct::Filter(filter) => indexed_filter_matches_document(document_id, filter, fields),
    }
}

fn indexed_filter_matches_document(
    document_id: &str,
    filter: &Value,
    fields: &BTreeMap<String, FieldIndex>,
) -> Result<bool, CoreError> {
    let node = filter
        .as_object()
        .ok_or_else(|| invalid("validated filter must be an object"))?;
    for (key, spec) in node {
        if !indexed_entry_matches_document(document_id, key, spec, fields)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn indexed_entry_matches_document(
    document_id: &str,
    key: &str,
    spec: &Value,
    fields: &BTreeMap<String, FieldIndex>,
) -> Result<bool, CoreError> {
    Ok(match key {
        "$and" => spec
            .as_array()
            .ok_or_else(|| invalid("validated $and must be a list"))?
            .iter()
            .map(|sub| indexed_filter_matches_document(document_id, sub, fields))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .all(|matched| matched),
        "$or" => spec
            .as_array()
            .ok_or_else(|| invalid("validated $or must be a list"))?
            .iter()
            .map(|sub| indexed_filter_matches_document(document_id, sub, fields))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .any(|matched| matched),
        "$not" => !indexed_filter_matches_document(document_id, spec, fields)?,
        field => indexed_field_matches_document(document_id, field, spec, fields)?,
    })
}

fn indexed_field_matches_document(
    document_id: &str,
    field: &str,
    spec: &Value,
    fields: &BTreeMap<String, FieldIndex>,
) -> Result<bool, CoreError> {
    if let Some(expected) = spec.as_str() {
        return Ok(indexed_value_matches(document_id, field, expected, fields));
    }
    let operators = spec
        .as_object()
        .ok_or_else(|| invalid("validated field spec must be a string or operator map"))?;
    for (op, operand) in operators {
        let matched = match op.as_str() {
            "$eq" => {
                let value = operand
                    .as_str()
                    .ok_or_else(|| invalid("validated $eq operand must be a string"))?;
                indexed_value_matches(document_id, field, value, fields)
            }
            "$ne" => {
                let value = operand
                    .as_str()
                    .ok_or_else(|| invalid("validated $ne operand must be a string"))?;
                !indexed_value_matches(document_id, field, value, fields)
            }
            "$in" => {
                let values = operand
                    .as_array()
                    .ok_or_else(|| invalid("validated $in operand must be a list"))?;
                values.iter().try_fold(false, |matched, value| {
                    let value = value
                        .as_str()
                        .ok_or_else(|| invalid("validated $in values must be strings"))?;
                    Ok(matched || indexed_value_matches(document_id, field, value, fields))
                })?
            }
            "$nin" => {
                let values = operand
                    .as_array()
                    .ok_or_else(|| invalid("validated $nin operand must be a list"))?;
                !values.iter().try_fold(false, |matched, value| {
                    let value = value
                        .as_str()
                        .ok_or_else(|| invalid("validated $nin values must be strings"))?;
                    Ok(matched || indexed_value_matches(document_id, field, value, fields))
                })?
            }
            "$exists" => {
                let exists = operand
                    .as_bool()
                    .ok_or_else(|| invalid("validated $exists operand must be a boolean"))?;
                fields
                    .get(field)
                    .is_some_and(|index| index.docs.contains(document_id))
                    == exists
            }
            ordered if crate::filter::ast::is_ordered_operator(ordered) => {
                let operand = operand
                    .as_str()
                    .ok_or_else(|| invalid("validated ordered operand must be a string"))?;
                fields.get(field).is_some_and(|index| {
                    index.value_docs.iter().any(|(value, docs)| {
                        docs.contains(document_id) && compare_ordered(value, ordered, operand)
                    })
                })
            }
            _ => return Err(invalid("unsupported validated operator")),
        };
        if !matched {
            return Ok(false);
        }
    }
    Ok(true)
}

fn indexed_value_matches(
    document_id: &str,
    field: &str,
    expected: &str,
    fields: &BTreeMap<String, FieldIndex>,
) -> bool {
    fields
        .get(field)
        .and_then(|index| index.value_docs.get(expected))
        .is_some_and(|docs| docs.contains(document_id))
}
