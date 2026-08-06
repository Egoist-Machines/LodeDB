//! Metadata filter planning over index cardinalities.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::error::{CoreError, CoreErrorCode};
use crate::filter::ast::is_ordered_operator;
use crate::filter::doc_set::DocSet;
use crate::filter::field_index::FieldIndex;
use crate::filter::resolve::resolve_filter;

/// A planned metadata filter execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetadataFilterPlan {
    /// The matching document-id set is selective enough to materialize directly.
    Materialized(DocSet),
    /// The query path should keep the filter as a per-document predicate.
    Predicate(MetadataPredicatePlan),
}

/// Deferred predicate metadata used by engine query paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataPredicatePlan {
    /// Exact matching document count when known without refining candidates.
    pub matching_count: Option<usize>,
    /// Exact failing document count when known without refining candidates.
    pub failing_count: Option<usize>,
    /// A cheap superset of failing documents. The engine refines it with the
    /// predicate against stored metadata, which keeps complement clauses bounded
    /// by the indexed field rather than by the corpus.
    pub failing_candidates: Option<DocSet>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeSide {
    Matching,
    Failing,
}

#[derive(Debug, Clone, Copy)]
struct ProbeInfo {
    count: usize,
}

/// Plans a validated metadata filter without materializing unselective matches.
pub fn plan_metadata_filter(
    metadata_filter: &Value,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<MetadataFilterPlan, CoreError> {
    let total = all_docs.len();
    let matching_count = exact_matching_count(metadata_filter, fields, all_docs)?;
    let matching_probe = probe_info(metadata_filter, ProbeSide::Matching, fields, all_docs)?;
    let failing_probe = probe_info(metadata_filter, ProbeSide::Failing, fields, all_docs)?;

    if matching_count == Some(0) || matching_probe.is_some_and(|probe| probe.count == 0) {
        return Ok(MetadataFilterPlan::Materialized(DocSet::new()));
    }
    if matching_count == Some(total) || failing_probe.is_some_and(|probe| probe.count == 0) {
        return Ok(MetadataFilterPlan::Predicate(MetadataPredicatePlan {
            matching_count: Some(total),
            failing_count: Some(0),
            failing_candidates: Some(DocSet::new()),
        }));
    }

    if let Some(count) = matching_count {
        if count.saturating_mul(2) > total {
            if failing_probe.is_some() {
                return Ok(MetadataFilterPlan::Predicate(MetadataPredicatePlan {
                    matching_count: None,
                    failing_count: None,
                    failing_candidates: Some(materialize_probe(
                        metadata_filter,
                        ProbeSide::Failing,
                        fields,
                        all_docs,
                    )?),
                }));
            }
            return Ok(MetadataFilterPlan::Predicate(MetadataPredicatePlan {
                matching_count: Some(count),
                failing_count: Some(total.saturating_sub(count)),
                failing_candidates: None,
            }));
        }
    }

    if let Some(failing_probe) = failing_probe {
        let matching_cost = matching_probe.map_or(usize::MAX, |probe| probe.count);
        if failing_probe.count < matching_cost {
            return Ok(MetadataFilterPlan::Predicate(MetadataPredicatePlan {
                matching_count: None,
                failing_count: None,
                failing_candidates: Some(materialize_probe(
                    metadata_filter,
                    ProbeSide::Failing,
                    fields,
                    all_docs,
                )?),
            }));
        }
    }

    Ok(MetadataFilterPlan::Materialized(resolve_filter(
        metadata_filter,
        fields,
        all_docs,
    )?))
}

/// Returns the best cheap matching-cardinality bound for id-bounded planning.
pub fn estimate_matching_cardinality(
    metadata_filter: &Value,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<usize, CoreError> {
    if let Some(count) = exact_matching_count(metadata_filter, fields, all_docs)? {
        return Ok(count);
    }
    Ok(
        probe_info(metadata_filter, ProbeSide::Matching, fields, all_docs)?
            .map_or(all_docs.len(), |probe| probe.count),
    )
}

fn exact_matching_count(
    metadata_filter: &Value,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<Option<usize>, CoreError> {
    count_node(metadata_filter, fields, all_docs)
}

fn count_node(
    filter: &Value,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<Option<usize>, CoreError> {
    let node = filter
        .as_object()
        .ok_or_else(|| invalid("validated filter must be an object"))?;
    count_and(
        node.iter()
            .map(|(key, spec)| count_entry(key, spec, fields, all_docs))
            .collect::<Result<Vec<_>, _>>()?,
        all_docs.len(),
    )
}

fn count_entry(
    key: &str,
    spec: &Value,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<Option<usize>, CoreError> {
    match key {
        "$and" => {
            let subs = spec
                .as_array()
                .ok_or_else(|| invalid("validated $and must be a list"))?;
            count_and(
                subs.iter()
                    .map(|sub| count_node(sub, fields, all_docs))
                    .collect::<Result<Vec<_>, _>>()?,
                all_docs.len(),
            )
        }
        "$or" => {
            let subs = spec
                .as_array()
                .ok_or_else(|| invalid("validated $or must be a list"))?;
            count_or(
                subs.iter()
                    .map(|sub| count_node(sub, fields, all_docs))
                    .collect::<Result<Vec<_>, _>>()?,
                all_docs.len(),
            )
        }
        "$not" => {
            Ok(count_node(spec, fields, all_docs)?
                .map(|count| all_docs.len().saturating_sub(count)))
        }
        field => count_field(field, spec, fields, all_docs),
    }
}

fn count_and(counts: Vec<Option<usize>>, total: usize) -> Result<Option<usize>, CoreError> {
    if counts.is_empty() {
        return Ok(Some(total));
    }
    if counts.contains(&Some(0)) {
        return Ok(Some(0));
    }
    if counts.iter().all(|count| *count == Some(total)) {
        return Ok(Some(total));
    }
    if counts.len() == 1 {
        return Ok(counts[0]);
    }
    Ok(None)
}

fn count_or(counts: Vec<Option<usize>>, total: usize) -> Result<Option<usize>, CoreError> {
    if counts.is_empty() {
        return Ok(Some(0));
    }
    if counts.contains(&Some(total)) {
        return Ok(Some(total));
    }
    if counts.iter().all(|count| *count == Some(0)) {
        return Ok(Some(0));
    }
    if counts.len() == 1 {
        return Ok(counts[0]);
    }
    Ok(None)
}

fn count_field(
    field: &str,
    spec: &Value,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<Option<usize>, CoreError> {
    let empty = FieldIndex::default();
    let index = fields.get(field).unwrap_or(&empty);
    if let Some(expected) = spec.as_str() {
        return Ok(Some(index.value_docs.get(expected).map_or(0, DocSet::len)));
    }
    let operators = spec
        .as_object()
        .ok_or_else(|| invalid("validated field spec must be a string or operator map"))?;
    count_and(
        operators
            .iter()
            .map(|(op, operand)| count_operator(op, operand, index, all_docs.len()))
            .collect::<Result<Vec<_>, _>>()?,
        all_docs.len(),
    )
}

fn count_operator(
    op: &str,
    operand: &Value,
    index: &FieldIndex,
    total: usize,
) -> Result<Option<usize>, CoreError> {
    match op {
        "$eq" => {
            let value = operand
                .as_str()
                .ok_or_else(|| invalid("validated $eq operand must be a string"))?;
            Ok(Some(index.value_docs.get(value).map_or(0, DocSet::len)))
        }
        "$ne" => {
            let value = operand
                .as_str()
                .ok_or_else(|| invalid("validated $ne operand must be a string"))?;
            Ok(Some(total.saturating_sub(
                index.value_docs.get(value).map_or(0, DocSet::len),
            )))
        }
        "$in" => {
            let docs = union_values(index, operand, "validated $in operand must be a list")?;
            Ok(Some(docs.len()))
        }
        "$nin" => {
            let docs = union_values(index, operand, "validated $nin operand must be a list")?;
            Ok(Some(total.saturating_sub(docs.len())))
        }
        "$exists" => {
            let exists = operand
                .as_bool()
                .ok_or_else(|| invalid("validated $exists operand must be a boolean"))?;
            if exists {
                Ok(Some(index.docs.len()))
            } else {
                Ok(Some(total.saturating_sub(index.docs.len())))
            }
        }
        ordered if is_ordered_operator(ordered) => {
            let operand = operand
                .as_str()
                .ok_or_else(|| invalid("validated ordered operand must be a string"))?;
            Ok(Some(index.count_ordered(ordered, operand)))
        }
        _ => Err(invalid("unsupported validated operator")),
    }
}

fn probe_info(
    filter: &Value,
    side: ProbeSide,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<Option<ProbeInfo>, CoreError> {
    let node = filter
        .as_object()
        .ok_or_else(|| invalid("validated filter must be an object"))?;
    combine_and_probes(
        node.iter()
            .map(|(key, spec)| probe_entry(key, spec, side, fields, all_docs))
            .collect::<Result<Vec<_>, _>>()?,
        side,
        all_docs.len(),
    )
}

fn probe_entry(
    key: &str,
    spec: &Value,
    side: ProbeSide,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<Option<ProbeInfo>, CoreError> {
    match key {
        "$and" => {
            let subs = spec
                .as_array()
                .ok_or_else(|| invalid("validated $and must be a list"))?;
            combine_and_probes(
                subs.iter()
                    .map(|sub| probe_info(sub, side, fields, all_docs))
                    .collect::<Result<Vec<_>, _>>()?,
                side,
                all_docs.len(),
            )
        }
        "$or" => {
            let subs = spec
                .as_array()
                .ok_or_else(|| invalid("validated $or must be a list"))?;
            combine_or_probes(
                subs.iter()
                    .map(|sub| probe_info(sub, side, fields, all_docs))
                    .collect::<Result<Vec<_>, _>>()?,
                side,
                all_docs.len(),
            )
        }
        "$not" => probe_info(
            spec,
            match side {
                ProbeSide::Matching => ProbeSide::Failing,
                ProbeSide::Failing => ProbeSide::Matching,
            },
            fields,
            all_docs,
        ),
        field => probe_field(field, spec, side, fields, all_docs),
    }
}

fn combine_and_probes(
    probes: Vec<Option<ProbeInfo>>,
    side: ProbeSide,
    total: usize,
) -> Result<Option<ProbeInfo>, CoreError> {
    if probes.is_empty() {
        return Ok(Some(ProbeInfo {
            count: match side {
                ProbeSide::Matching => total,
                ProbeSide::Failing => 0,
            },
        }));
    }
    match side {
        ProbeSide::Matching => Ok(probes.into_iter().flatten().min_by_key(|probe| probe.count)),
        ProbeSide::Failing => {
            if probes.iter().any(Option::is_none) {
                return Ok(None);
            }
            Ok(Some(ProbeInfo {
                count: probes
                    .into_iter()
                    .flatten()
                    .map(|probe| probe.count)
                    .sum::<usize>()
                    .min(total),
            }))
        }
    }
}

fn combine_or_probes(
    probes: Vec<Option<ProbeInfo>>,
    side: ProbeSide,
    total: usize,
) -> Result<Option<ProbeInfo>, CoreError> {
    if probes.is_empty() {
        return Ok(Some(ProbeInfo {
            count: match side {
                ProbeSide::Matching => 0,
                ProbeSide::Failing => total,
            },
        }));
    }
    match side {
        ProbeSide::Matching => {
            if probes.iter().any(Option::is_none) {
                return Ok(None);
            }
            Ok(Some(ProbeInfo {
                count: probes
                    .into_iter()
                    .flatten()
                    .map(|probe| probe.count)
                    .sum::<usize>()
                    .min(total),
            }))
        }
        ProbeSide::Failing => Ok(probes.into_iter().flatten().min_by_key(|probe| probe.count)),
    }
}

fn probe_field(
    field: &str,
    spec: &Value,
    side: ProbeSide,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<Option<ProbeInfo>, CoreError> {
    let empty = FieldIndex::default();
    let index = fields.get(field).unwrap_or(&empty);
    if let Some(expected) = spec.as_str() {
        return Ok(probe_eq(index, expected, side, all_docs.len()));
    }
    let operators = spec
        .as_object()
        .ok_or_else(|| invalid("validated field spec must be a string or operator map"))?;
    combine_and_probes(
        operators
            .iter()
            .map(|(op, operand)| probe_operator(op, operand, index, side, all_docs.len()))
            .collect::<Result<Vec<_>, _>>()?,
        side,
        all_docs.len(),
    )
}

fn probe_operator(
    op: &str,
    operand: &Value,
    index: &FieldIndex,
    side: ProbeSide,
    total: usize,
) -> Result<Option<ProbeInfo>, CoreError> {
    match op {
        "$eq" => {
            let value = operand
                .as_str()
                .ok_or_else(|| invalid("validated $eq operand must be a string"))?;
            Ok(probe_eq(index, value, side, total))
        }
        "$ne" => {
            let value = operand
                .as_str()
                .ok_or_else(|| invalid("validated $ne operand must be a string"))?;
            Ok(match side {
                ProbeSide::Matching => {
                    if index.value_docs.get(value).map_or(0, DocSet::len) == total {
                        Some(ProbeInfo { count: 0 })
                    } else {
                        None
                    }
                }
                ProbeSide::Failing => Some(ProbeInfo {
                    count: index.value_docs.get(value).map_or(0, DocSet::len),
                }),
            })
        }
        "$in" => {
            let docs = union_values(index, operand, "validated $in operand must be a list")?;
            Ok(match side {
                ProbeSide::Matching => Some(ProbeInfo { count: docs.len() }),
                ProbeSide::Failing => {
                    if docs.len() == total {
                        Some(ProbeInfo { count: 0 })
                    } else {
                        None
                    }
                }
            })
        }
        "$nin" => {
            let docs = union_values(index, operand, "validated $nin operand must be a list")?;
            Ok(match side {
                ProbeSide::Matching => {
                    if docs.len() == total {
                        Some(ProbeInfo { count: 0 })
                    } else {
                        None
                    }
                }
                ProbeSide::Failing => Some(ProbeInfo { count: docs.len() }),
            })
        }
        "$exists" => {
            let exists = operand
                .as_bool()
                .ok_or_else(|| invalid("validated $exists operand must be a boolean"))?;
            Ok(match (exists, side) {
                (true, ProbeSide::Matching) => Some(ProbeInfo {
                    count: index.docs.len(),
                }),
                (true, ProbeSide::Failing) => {
                    if index.docs.len() == total {
                        Some(ProbeInfo { count: 0 })
                    } else {
                        None
                    }
                }
                (false, ProbeSide::Matching) => {
                    if index.docs.len() == total {
                        Some(ProbeInfo { count: 0 })
                    } else {
                        None
                    }
                }
                (false, ProbeSide::Failing) => Some(ProbeInfo {
                    count: index.docs.len(),
                }),
            })
        }
        ordered if is_ordered_operator(ordered) => {
            let operand = operand
                .as_str()
                .ok_or_else(|| invalid("validated ordered operand must be a string"))?;
            let count = index.count_ordered(ordered, operand);
            Ok(match side {
                ProbeSide::Matching => Some(ProbeInfo { count }),
                ProbeSide::Failing => {
                    if count == total {
                        Some(ProbeInfo { count: 0 })
                    } else {
                        None
                    }
                }
            })
        }
        _ => Err(invalid("unsupported validated operator")),
    }
}

fn probe_eq(index: &FieldIndex, value: &str, side: ProbeSide, total: usize) -> Option<ProbeInfo> {
    let count = index.value_docs.get(value).map_or(0, DocSet::len);
    match side {
        ProbeSide::Matching => Some(ProbeInfo { count }),
        ProbeSide::Failing => {
            if count == total {
                Some(ProbeInfo { count: 0 })
            } else {
                None
            }
        }
    }
}

fn materialize_probe(
    filter: &Value,
    side: ProbeSide,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<DocSet, CoreError> {
    let node = filter
        .as_object()
        .ok_or_else(|| invalid("validated filter must be an object"))?;
    let entries = node
        .iter()
        .map(|(key, spec)| (key.as_str(), spec))
        .collect::<Vec<_>>();
    materialize_and_entries(&entries, side, fields, all_docs)
}

fn materialize_entry_probe(
    key: &str,
    spec: &Value,
    side: ProbeSide,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<Option<DocSet>, CoreError> {
    match key {
        "$and" => {
            let subs = spec
                .as_array()
                .ok_or_else(|| invalid("validated $and must be a list"))?;
            materialize_and_filters(subs, side, fields, all_docs).map(Some)
        }
        "$or" => {
            let subs = spec
                .as_array()
                .ok_or_else(|| invalid("validated $or must be a list"))?;
            materialize_or_filters(subs, side, fields, all_docs).map(Some)
        }
        "$not" => materialize_probe(
            spec,
            match side {
                ProbeSide::Matching => ProbeSide::Failing,
                ProbeSide::Failing => ProbeSide::Matching,
            },
            fields,
            all_docs,
        )
        .map(Some),
        field => materialize_field_probe(field, spec, side, fields, all_docs).map(Some),
    }
}

fn materialize_and_entries(
    entries: &[(&str, &Value)],
    side: ProbeSide,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<DocSet, CoreError> {
    match side {
        ProbeSide::Matching => {
            let (key, spec) = entries
                .iter()
                .filter_map(|(key, spec)| {
                    probe_entry(key, spec, side, fields, all_docs)
                        .ok()
                        .flatten()
                        .map(|probe| (probe.count, *key, *spec))
                })
                .min_by_key(|(count, _, _)| *count)
                .map(|(_, key, spec)| (key, spec))
                .ok_or_else(|| invalid("validated $and has no matching probe"))?;
            materialize_entry_probe(key, spec, side, fields, all_docs)?
                .ok_or_else(|| invalid("validated $and matching probe disappeared"))
        }
        ProbeSide::Failing => {
            let mut docs = DocSet::new();
            for (key, spec) in entries {
                if probe_entry(key, spec, side, fields, all_docs)?.is_none() {
                    return Err(invalid("validated $and has no failing probe"));
                }
                let probe = materialize_entry_probe(key, spec, side, fields, all_docs)?
                    .ok_or_else(|| invalid("validated $and failing probe disappeared"))?;
                docs.extend(probe);
            }
            Ok(docs)
        }
    }
}

fn materialize_and_filters(
    filters: &[Value],
    side: ProbeSide,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<DocSet, CoreError> {
    match side {
        ProbeSide::Matching => {
            let filter = filters
                .iter()
                .filter_map(|filter| {
                    probe_info(filter, side, fields, all_docs)
                        .ok()
                        .flatten()
                        .map(|probe| (probe.count, filter))
                })
                .min_by_key(|(count, _)| *count)
                .map(|(_, filter)| filter)
                .ok_or_else(|| invalid("validated $and has no matching probe"))?;
            materialize_probe(filter, side, fields, all_docs)
        }
        ProbeSide::Failing => {
            let mut docs = DocSet::new();
            for filter in filters {
                if probe_info(filter, side, fields, all_docs)?.is_none() {
                    return Err(invalid("validated $and has no failing probe"));
                }
                docs.extend(materialize_probe(filter, side, fields, all_docs)?);
            }
            Ok(docs)
        }
    }
}

fn materialize_or_filters(
    filters: &[Value],
    side: ProbeSide,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<DocSet, CoreError> {
    match side {
        ProbeSide::Matching => {
            let mut docs = DocSet::new();
            for filter in filters {
                if probe_info(filter, side, fields, all_docs)?.is_none() {
                    return Err(invalid("validated $or has no matching probe"));
                }
                docs.extend(materialize_probe(filter, side, fields, all_docs)?);
            }
            Ok(docs)
        }
        ProbeSide::Failing => {
            let filter = filters
                .iter()
                .filter_map(|filter| {
                    probe_info(filter, side, fields, all_docs)
                        .ok()
                        .flatten()
                        .map(|probe| (probe.count, filter))
                })
                .min_by_key(|(count, _)| *count)
                .map(|(_, filter)| filter)
                .ok_or_else(|| invalid("validated $or has no failing probe"))?;
            materialize_probe(filter, side, fields, all_docs)
        }
    }
}

fn materialize_field_probe(
    field: &str,
    spec: &Value,
    side: ProbeSide,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<DocSet, CoreError> {
    let empty = FieldIndex::default();
    let index = fields.get(field).unwrap_or(&empty);
    if let Some(expected) = spec.as_str() {
        return materialize_eq(index, expected, side, all_docs.len());
    }
    let operators = spec
        .as_object()
        .ok_or_else(|| invalid("validated field spec must be a string or operator map"))?;
    match side {
        ProbeSide::Matching => {
            let (op, operand) = operators
                .iter()
                .filter_map(|(op, operand)| {
                    probe_operator(op, operand, index, side, all_docs.len())
                        .ok()
                        .flatten()
                        .map(|probe| (probe.count, op.as_str(), operand))
                })
                .min_by_key(|(count, _, _)| *count)
                .map(|(_, op, operand)| (op, operand))
                .ok_or_else(|| invalid("validated field has no matching probe"))?;
            materialize_operator_probe(op, operand, index, side, all_docs.len())?
                .ok_or_else(|| invalid("validated field matching probe disappeared"))
        }
        ProbeSide::Failing => {
            let mut docs = DocSet::new();
            for (op, operand) in operators {
                if probe_operator(op, operand, index, side, all_docs.len())?.is_none() {
                    return Err(invalid("validated field has no failing probe"));
                }
                let probe = materialize_operator_probe(op, operand, index, side, all_docs.len())?
                    .ok_or_else(|| invalid("validated field failing probe disappeared"))?;
                docs.extend(probe);
            }
            Ok(docs)
        }
    }
}

fn materialize_operator_probe(
    op: &str,
    operand: &Value,
    index: &FieldIndex,
    side: ProbeSide,
    total: usize,
) -> Result<Option<DocSet>, CoreError> {
    match op {
        "$eq" => {
            let value = operand
                .as_str()
                .ok_or_else(|| invalid("validated $eq operand must be a string"))?;
            materialize_eq(index, value, side, total).map(Some)
        }
        "$ne" => {
            let value = operand
                .as_str()
                .ok_or_else(|| invalid("validated $ne operand must be a string"))?;
            match side {
                ProbeSide::Matching => {
                    if index.value_docs.get(value).map_or(0, DocSet::len) == total {
                        Ok(Some(DocSet::new()))
                    } else {
                        Ok(None)
                    }
                }
                ProbeSide::Failing => Ok(Some(
                    index.value_docs.get(value).cloned().unwrap_or_default(),
                )),
            }
        }
        "$in" => {
            let docs = union_values(index, operand, "validated $in operand must be a list")?;
            match side {
                ProbeSide::Matching => Ok(Some(docs)),
                ProbeSide::Failing => {
                    if docs.len() == total {
                        Ok(Some(DocSet::new()))
                    } else {
                        Ok(None)
                    }
                }
            }
        }
        "$nin" => {
            let docs = union_values(index, operand, "validated $nin operand must be a list")?;
            match side {
                ProbeSide::Matching => {
                    if docs.len() == total {
                        Ok(Some(DocSet::new()))
                    } else {
                        Ok(None)
                    }
                }
                ProbeSide::Failing => Ok(Some(docs)),
            }
        }
        "$exists" => {
            let exists = operand
                .as_bool()
                .ok_or_else(|| invalid("validated $exists operand must be a boolean"))?;
            Ok(match (exists, side) {
                (true, ProbeSide::Matching) => Some(index.docs.clone()),
                (true, ProbeSide::Failing) => {
                    if index.docs.len() == total {
                        Some(DocSet::new())
                    } else {
                        None
                    }
                }
                (false, ProbeSide::Matching) => {
                    if index.docs.len() == total {
                        Some(DocSet::new())
                    } else {
                        None
                    }
                }
                (false, ProbeSide::Failing) => Some(index.docs.clone()),
            })
        }
        ordered if is_ordered_operator(ordered) => {
            let operand = operand
                .as_str()
                .ok_or_else(|| invalid("validated ordered operand must be a string"))?;
            let docs = index.resolve_ordered(ordered, operand);
            Ok(match side {
                ProbeSide::Matching => Some(docs),
                ProbeSide::Failing => {
                    if docs.len() == total {
                        Some(DocSet::new())
                    } else {
                        None
                    }
                }
            })
        }
        _ => Err(invalid("unsupported validated operator")),
    }
}

fn materialize_eq(
    index: &FieldIndex,
    value: &str,
    side: ProbeSide,
    total: usize,
) -> Result<DocSet, CoreError> {
    let docs = index.value_docs.get(value).cloned().unwrap_or_default();
    match side {
        ProbeSide::Matching => Ok(docs),
        ProbeSide::Failing => {
            if docs.len() == total {
                Ok(DocSet::new())
            } else {
                Err(invalid("validated equality has no failing probe"))
            }
        }
    }
}

fn union_values(
    index: &FieldIndex,
    operand: &Value,
    list_error: &str,
) -> Result<DocSet, CoreError> {
    let values = operand.as_array().ok_or_else(|| invalid(list_error))?;
    let mut docs = DocSet::new();
    for value in values {
        let value = value
            .as_str()
            .ok_or_else(|| invalid("validated $in/$nin values must be strings"))?;
        if let Some(value_docs) = index.value_docs.get(value) {
            docs.extend(value_docs.iter().cloned());
        }
    }
    Ok(docs)
}

fn invalid(message: impl Into<String>) -> CoreError {
    CoreError::new(CoreErrorCode::InvalidArgument, message)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;

    use super::{estimate_matching_cardinality, plan_metadata_filter, MetadataFilterPlan};
    use crate::filter::field_index::build_field_indexes;
    use crate::filter::validate::validate_metadata_filter;

    fn indexes(
        rows: &[(&str, &[(&str, &str)])],
    ) -> (
        BTreeMap<String, crate::filter::FieldIndex>,
        crate::filter::DocSet,
    ) {
        let metadata = rows
            .iter()
            .map(|(id, entries)| {
                (
                    (*id).to_string(),
                    entries
                        .iter()
                        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
                        .collect(),
                )
            })
            .collect();
        build_field_indexes(&metadata)
    }

    #[test]
    fn complement_or_uses_failing_candidates() {
        let (fields, all_docs) = indexes(&[
            ("a", &[("expires", "10")]),
            ("b", &[]),
            ("c", &[]),
            ("d", &[("expires", "30")]),
        ]);
        let filter = validate_metadata_filter(&json!({
            "$or": [
                {"expires": {"$exists": false}},
                {"expires": {"$gt": "20"}}
            ]
        }))
        .unwrap();

        let plan = plan_metadata_filter(&filter, &fields, &all_docs).unwrap();
        match plan {
            MetadataFilterPlan::Predicate(predicate) => {
                assert_eq!(
                    predicate.failing_candidates.unwrap(),
                    ["a".to_string(), "d".to_string()].into_iter().collect()
                );
            }
            MetadataFilterPlan::Materialized(_) => panic!("complement OR should defer"),
        }
    }

    #[test]
    fn planner_flips_at_half_for_exact_positive_clause() {
        let rows = (0..10)
            .map(|i| {
                let entries: &[(&str, &str)] = if i < 5 { &[("flag", "yes")] } else { &[] };
                (format!("d{i}"), entries.to_vec())
            })
            .collect::<Vec<_>>();
        let borrowed = rows
            .iter()
            .map(|(id, entries)| {
                (
                    id.as_str(),
                    entries
                        .iter()
                        .map(|(key, value)| (*key, *value))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        let borrowed = borrowed
            .iter()
            .map(|(id, entries)| (*id, entries.as_slice()))
            .collect::<Vec<_>>();
        let (fields, all_docs) = indexes(&borrowed);
        let filter = validate_metadata_filter(&json!({"flag": {"$exists": true}})).unwrap();
        assert!(matches!(
            plan_metadata_filter(&filter, &fields, &all_docs).unwrap(),
            MetadataFilterPlan::Materialized(_)
        ));

        let rows = (0..10)
            .map(|i| {
                let entries: &[(&str, &str)] = if i < 6 { &[("flag", "yes")] } else { &[] };
                (format!("d{i}"), entries.to_vec())
            })
            .collect::<Vec<_>>();
        let borrowed = rows
            .iter()
            .map(|(id, entries)| {
                (
                    id.as_str(),
                    entries
                        .iter()
                        .map(|(key, value)| (*key, *value))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        let borrowed = borrowed
            .iter()
            .map(|(id, entries)| (*id, entries.as_slice()))
            .collect::<Vec<_>>();
        let (fields, all_docs) = indexes(&borrowed);
        assert!(matches!(
            plan_metadata_filter(&filter, &fields, &all_docs).unwrap(),
            MetadataFilterPlan::Predicate(_)
        ));
    }

    #[test]
    fn estimates_document_id_bound_without_materializing_missing_docs() {
        let (fields, all_docs) = indexes(&[
            ("a", &[("expires", "10")]),
            ("b", &[]),
            ("c", &[]),
            ("d", &[]),
        ]);
        let filter = validate_metadata_filter(&json!({"expires": {"$exists": false}})).unwrap();
        assert_eq!(
            estimate_matching_cardinality(&filter, &fields, &all_docs).unwrap(),
            3
        );
    }
}
