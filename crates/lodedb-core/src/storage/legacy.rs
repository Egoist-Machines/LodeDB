use crate::storage::text_store;
use crate::storage::util::{read_json, CoreResult};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct LegacyStore {
    pub payload: Value,
    pub raw_text: BTreeMap<String, String>,
}

pub fn load_top_level_json(persistence_dir: &Path, index_key: &str) -> CoreResult<LegacyStore> {
    let payload = read_json(
        &persistence_dir.join(format!("{index_key}.json")),
        "legacy top-level state snapshot",
    )?;
    let raw_text =
        text_store::read_legacy_text_sidecar(&persistence_dir.join(format!("{index_key}.tvtext")))?;
    Ok(LegacyStore { payload, raw_text })
}
