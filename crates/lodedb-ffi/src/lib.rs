//! C ABI for the native LodeDB core.

use lodedb_core::engine::{CoreAppender, CoreEngine};
use lodedb_core::engine::{IngestPlan, QueryPlan};
use lodedb_core::types::{
    CoreDocument, CoreIndexCreateRequest, CoreMetadata, CoreOpenOptions, CoreVectorDocument,
};
use lodedb_core::{CoreError, CoreErrorCode};
use std::collections::BTreeMap;
use std::ffi::{c_char, CString};
use std::os::raw::c_float;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;

const ABI_VERSION: u32 = lodedb_core::NATIVE_CORE_ABI_VERSION;

#[repr(C)]
pub struct LodeError {
    size: u32,
    version: u32,
    code: u32,
    message: *mut c_char,
}

#[repr(C)]
pub struct LodeEngine {
    engine: CoreEngine,
}

#[repr(C)]
pub struct LodeAppender {
    appender: CoreAppender,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LodeStringView {
    size: u32,
    version: u32,
    data: *const c_char,
    len: usize,
}

#[repr(C)]
pub struct LodeOwnedString {
    size: u32,
    version: u32,
    data: *mut c_char,
    len: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LodeMetadataPair {
    size: u32,
    version: u32,
    key: LodeStringView,
    value: LodeStringView,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LodeVectorDocument {
    size: u32,
    version: u32,
    document_id: LodeStringView,
    vector: *const c_float,
    vector_len: usize,
    metadata: *const LodeMetadataPair,
    metadata_len: usize,
    text: LodeStringView,
    has_text: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LodeSearchRequest {
    size: u32,
    version: u32,
    index_id: LodeStringView,
    query: *const c_float,
    query_len: usize,
    top_k: usize,
}

#[repr(C)]
pub struct LodeSearchHit {
    size: u32,
    version: u32,
    document_id: *mut c_char,
    chunk_id: *mut c_char,
    score: c_float,
}

#[repr(C)]
pub struct LodeSearchResults {
    size: u32,
    version: u32,
    hits: *mut LodeSearchHit,
    hits_len: usize,
    total_considered: usize,
}

#[no_mangle]
pub extern "C" fn lodedb_abi_version() -> u32 {
    ABI_VERSION
}

/// Frees an error allocated by this library.
///
/// # Safety
///
/// `error` must be null or a pointer previously returned through a `LodeError **`
/// out-parameter by this library. It must not be freed more than once.
#[no_mangle]
pub unsafe extern "C" fn lodedb_error_free(error: *mut LodeError) {
    if error.is_null() {
        return;
    }
    let error = Box::from_raw(error);
    if !error.message.is_null() {
        let _ = CString::from_raw(error.message);
    }
}

/// Frees an owned string allocated by this library.
///
/// # Safety
///
/// `text` must be null or a pointer returned by an FFI function that documents
/// `LodeOwnedString` ownership. It must not be used after this call or freed
/// more than once.
#[no_mangle]
pub unsafe extern "C" fn lodedb_owned_string_free(text: *mut LodeOwnedString) {
    if text.is_null() {
        return;
    }
    let text = Box::from_raw(text);
    if !text.data.is_null() {
        let _ = CString::from_raw(text.data);
    }
}

/// Frees search results allocated by this library.
///
/// # Safety
///
/// `results` must be null or a pointer returned by `lodedb_engine_query_vector`.
/// It must not be used after this call and must not be freed more than once.
#[no_mangle]
pub unsafe extern "C" fn lodedb_search_results_free(results: *mut LodeSearchResults) {
    if results.is_null() {
        return;
    }
    let results = Box::from_raw(results);
    if !results.hits.is_null() {
        let hits = Vec::from_raw_parts(results.hits, results.hits_len, results.hits_len);
        for hit in hits {
            if !hit.document_id.is_null() {
                let _ = CString::from_raw(hit.document_id);
            }
            if !hit.chunk_id.is_null() {
                let _ = CString::from_raw(hit.chunk_id);
            }
        }
    }
}

/// Allocates a new in-memory engine handle.
///
/// # Safety
///
/// `out` must be a valid writable pointer to a `LodeEngine *`. `error` may be
/// null or a valid writable pointer to a `LodeError *`.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_new_in_memory(
    out: *mut *mut LodeEngine,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let handle = Box::new(LodeEngine {
            engine: CoreEngine::new_in_memory(),
        });
        unsafe {
            *out = Box::into_raw(handle);
        }
        Ok(())
    })
}

/// Frees an engine handle allocated by this library.
///
/// # Safety
///
/// `engine` must be null or a pointer returned by `lodedb_engine_new_in_memory`.
/// It must not be used after this call and must not be freed more than once.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_free(engine: *mut LodeEngine) {
    if !engine.is_null() {
        let _ = Box::from_raw(engine);
    }
}

/// Creates a vector index from a minimal JSON create request.
///
/// The JSON object carries only the distinguishing fields — `index_id`,
/// `vector_dim`, optional `bit_width` (defaults to 4), optional `model` (the
/// reopen-time embedder-guard identity), and optional `ann` tuning — and the core
/// supplies the identity defaults (name, provider, task, route/storage profile,
/// and `index_key`/`client_id_hash` from `index_id`). This is the single create
/// entry point: an exact index simply omits `ann`, so bindings never hand-copy the
/// native-default identity literals.
///
/// # Safety
///
/// `engine`, `options_json`, and `error` must be valid for the duration of the
/// call. `options_json` must reference valid UTF-8 that deserializes to
/// `CoreIndexCreateRequest`. `error` may be null or writable.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_create_index_json(
    engine: *mut LodeEngine,
    options_json: LodeStringView,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        let engine = engine_mut(engine)?;
        let request = read_json_view::<CoreIndexCreateRequest>(options_json)?;
        engine.create_index_with_options(request.into_options())
    })
}

/// Upserts contiguous f32 vector documents.
///
/// # Safety
///
/// `engine` must be valid. `index_id` and every string view in `documents` must
/// reference valid UTF-8 for the duration of the call. `documents` must either be
/// null with length zero or point to `documents_len` initialized records; each
/// vector and metadata slice must satisfy the same pointer/length rule.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_upsert_vectors(
    engine: *mut LodeEngine,
    index_id: LodeStringView,
    documents: *const LodeVectorDocument,
    documents_len: usize,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        let engine = engine_mut(engine)?;
        let index_id = read_string(index_id)?;
        let documents = read_vector_documents(documents, documents_len)?;
        engine.upsert_vectors(&index_id, &documents).map(|_| ())
    })
}

/// Searches one contiguous f32 query vector.
///
/// # Safety
///
/// `engine`, `request`, and `out` must be valid for the duration of the call.
/// The request's string and vector pointers must satisfy their length fields.
/// The returned results are Rust-owned and must be released with
/// `lodedb_search_results_free`.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_query_vector(
    engine: *const LodeEngine,
    request: *const LodeSearchRequest,
    out: *mut *mut LodeSearchResults,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        if request.is_null() {
            return invalid("request is null");
        }
        let engine = engine_ref(engine)?;
        let request = unsafe { *request };
        validate_abi_header(
            request.size,
            request.version,
            std::mem::size_of::<LodeSearchRequest>(),
            "search request",
        )?;
        let index_id = read_string(request.index_id)?;
        let query = read_f32_slice(request.query, request.query_len)?;
        let results = engine.query_vector(&index_id, query, request.top_k, None)?;
        let total_considered = results.total_considered;
        let mut hits = results
            .hits
            .into_iter()
            .map(|hit| {
                Ok(LodeSearchHit {
                    size: std::mem::size_of::<LodeSearchHit>() as u32,
                    version: ABI_VERSION,
                    document_id: c_string(hit.document_id)?.into_raw(),
                    chunk_id: c_string(hit.chunk_id)?.into_raw(),
                    score: hit.score,
                })
            })
            .collect::<Result<Vec<_>, CoreError>>()?;
        let result = Box::new(LodeSearchResults {
            size: std::mem::size_of::<LodeSearchResults>() as u32,
            version: ABI_VERSION,
            hits: hits.as_mut_ptr(),
            hits_len: hits.len(),
            total_considered,
        });
        std::mem::forget(hits);
        unsafe {
            *out = Box::into_raw(result);
        }
        Ok(())
    })
}

/// Plans a text upsert from a JSON array of `CoreDocument` objects.
///
/// Embeddings stay in the caller. The returned JSON is an `IngestPlan` and must
/// be released with `lodedb_owned_string_free`.
///
/// # Safety
///
/// `engine`, `index_id`, `documents_json`, and `out` must be valid for the
/// duration of the call. String views must contain valid UTF-8 bytes.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_prepare_text_upsert_json(
    engine: *mut LodeEngine,
    index_id: LodeStringView,
    documents_json: LodeStringView,
    store_text: u8,
    index_text: u8,
    chunk_character_limit: usize,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let engine = engine_mut(engine)?;
        let index_id = read_string(index_id)?;
        let documents = read_json_view::<Vec<CoreDocument>>(documents_json)?;
        let plan = engine.prepare_text_upsert(
            &index_id,
            &documents,
            store_text != 0,
            index_text != 0,
            chunk_character_limit,
        )?;
        let result = owned_json(&plan)?;
        unsafe {
            *out = Box::into_raw(result);
        }
        Ok(())
    })
}

/// Applies a JSON `IngestPlan` with caller-provided embeddings JSON.
///
/// The returned JSON is a `TextApplyResult` and must be released with
/// `lodedb_owned_string_free`.
///
/// # Safety
///
/// `engine`, `plan_json`, `embeddings_json`, and `out` must be valid for the
/// duration of the call. String views must contain valid UTF-8 bytes.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_apply_text_upsert_json(
    engine: *mut LodeEngine,
    plan_json: LodeStringView,
    embeddings_json: LodeStringView,
    embedding_time_ms: f64,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let engine = engine_mut(engine)?;
        let plan = read_json_view::<IngestPlan>(plan_json)?;
        let embeddings = read_json_view::<Vec<Vec<f32>>>(embeddings_json)?;
        let result = engine.apply_text_upsert(&plan, &embeddings, embedding_time_ms)?;
        let result = owned_json(&result)?;
        unsafe {
            *out = Box::into_raw(result);
        }
        Ok(())
    })
}

/// Prepares a text query as JSON while embeddings stay in the caller.
///
/// The returned JSON is a `QueryPlan` and must be released with
/// `lodedb_owned_string_free`.
///
/// # Safety
///
/// `engine`, `query`, `mode`, and `out` must be valid for the duration of the
/// call. String views must contain valid UTF-8 bytes.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_prepare_query_text_json(
    engine: *const LodeEngine,
    query: LodeStringView,
    mode: LodeStringView,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let engine = engine_ref(engine)?;
        let query = read_string(query)?;
        let mode = read_string(mode)?;
        let plan = engine.prepare_query_text(&query, &mode)?;
        let result = owned_json(&plan)?;
        unsafe {
            *out = Box::into_raw(result);
        }
        Ok(())
    })
}

/// Searches text using a JSON `QueryPlan` and optional caller-provided embedding.
///
/// The returned JSON is `CoreSearchResults` and must be released with
/// `lodedb_owned_string_free`.
///
/// # Safety
///
/// `engine`, `index_id`, `query_plan_json`, `out`, and any present optional JSON
/// views must be valid for the duration of the call. String views must contain
/// valid UTF-8 bytes.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_search_embedded_text_json(
    engine: *const LodeEngine,
    index_id: LodeStringView,
    query_plan_json: LodeStringView,
    query_embedding_json: LodeStringView,
    has_query_embedding: u8,
    top_k: usize,
    filter_json: LodeStringView,
    has_filter: u8,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let engine = engine_ref(engine)?;
        let index_id = read_string(index_id)?;
        let query_plan = read_json_view::<QueryPlan>(query_plan_json)?;
        let query_embedding = if has_query_embedding == 0 {
            None
        } else {
            Some(read_json_view::<Vec<f32>>(query_embedding_json)?)
        };
        let filter = if has_filter == 0 {
            None
        } else {
            Some(read_json_view::<serde_json::Value>(filter_json)?)
        };
        let results = engine.search_embedded_text(
            &index_id,
            &query_plan,
            query_embedding.as_deref(),
            top_k,
            filter.as_ref(),
        )?;
        let result = owned_json(&results)?;
        unsafe {
            *out = Box::into_raw(result);
        }
        Ok(())
    })
}

/// Upserts vector documents from a JSON array of `CoreVectorDocument` objects.
///
/// This is the JSON-shaped sibling of `lodedb_engine_upsert_vectors`; bindings
/// that already marshal documents as JSON (and treat ingest as a cold path) use
/// this to avoid hand-building the nested `LodeVectorDocument` C structs. Each
/// object is `{ "document_id", "vector": [f32...], "metadata": {..}, "text"? }`.
///
/// # Safety
///
/// `engine`, `index_id`, and `documents_json` must be valid for the duration of
/// the call. String views must contain valid UTF-8 bytes.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_upsert_vectors_json(
    engine: *mut LodeEngine,
    index_id: LodeStringView,
    documents_json: LodeStringView,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        let engine = engine_mut(engine)?;
        let index_id = read_string(index_id)?;
        let documents = read_json_view::<Vec<CoreVectorDocument>>(documents_json)?;
        engine.upsert_vectors(&index_id, &documents).map(|_| ())
    })
}

/// Searches one contiguous f32 query vector and returns `CoreSearchResults` JSON.
///
/// Unlike `lodedb_engine_query_vector`, this returns the full result set as JSON
/// (including per-hit metadata) and accepts an optional metadata filter, matching
/// the text-search path. The returned JSON must be released with
/// `lodedb_owned_string_free`.
///
/// # Safety
///
/// `engine`, `index_id`, `out`, and (when `has_filter` is non-zero) `filter_json`
/// must be valid for the duration of the call. `query` must either be null with
/// `query_len` zero or point to `query_len` initialized f32 values. String views
/// must contain valid UTF-8 bytes.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_query_vector_json(
    engine: *const LodeEngine,
    index_id: LodeStringView,
    query: *const c_float,
    query_len: usize,
    top_k: usize,
    filter_json: LodeStringView,
    has_filter: u8,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let engine = engine_ref(engine)?;
        let index_id = read_string(index_id)?;
        let query = read_f32_slice(query, query_len)?;
        let filter = if has_filter == 0 {
            None
        } else {
            Some(read_json_view::<serde_json::Value>(filter_json)?)
        };
        let results = engine.query_vector(&index_id, query, top_k, filter.as_ref())?;
        let result = owned_json(&results)?;
        unsafe {
            *out = Box::into_raw(result);
        }
        Ok(())
    })
}

// ---- Durable storage + CRUD (Phase 1) ----

/// Opens a writable persistent engine from a JSON `CoreOpenOptions`.
///
/// The returned handle owns a durable engine (WAL/generation storage under
/// `options.path`) and must be released with `lodedb_engine_free`.
///
/// # Safety
///
/// `options_json` and `out` must be valid for the duration of the call;
/// `options_json` must contain valid UTF-8 JSON.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_open_json(
    options_json: LodeStringView,
    out: *mut *mut LodeEngine,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let options = read_json_view::<CoreOpenOptions>(options_json)?;
        let engine = CoreEngine::open(options)?;
        let handle = Box::new(LodeEngine { engine });
        unsafe {
            *out = Box::into_raw(handle);
        }
        Ok(())
    })
}

/// Opens a lock-free read-only generation snapshot from a JSON `CoreOpenOptions`.
///
/// WAL tails are ignored; the snapshot reflects the last committed generation. The
/// returned handle must be released with `lodedb_engine_free`.
///
/// # Safety
///
/// `options_json` and `out` must be valid for the duration of the call;
/// `options_json` must contain valid UTF-8 JSON.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_open_readonly_json(
    options_json: LodeStringView,
    out: *mut *mut LodeEngine,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let options = read_json_view::<CoreOpenOptions>(options_json)?;
        let path = options.path.clone();
        let engine = CoreEngine::open_readonly(path, options)?;
        let handle = Box::new(LodeEngine { engine });
        unsafe {
            *out = Box::into_raw(handle);
        }
        Ok(())
    })
}

/// Flushes pending writes to durable storage (a generation commit / checkpoint).
///
/// # Safety
///
/// `engine` must be a valid writable engine pointer; `error` may be null or writable.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_persist(
    engine: *mut LodeEngine,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || engine_mut(engine)?.persist())
}

/// Closes the engine's writable generation (a final checkpoint). Idempotent.
///
/// # Safety
///
/// `engine` must be a valid engine pointer; `error` may be null or writable.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_close(
    engine: *mut LodeEngine,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || engine_mut(engine)?.close())
}

/// Overlays the current WAL tail into a read-only handle's in-memory view without
/// checkpointing, giving reader freshness and read-your-writes. A no-op for a
/// writable handle (which folds the WAL on open). After this returns,
/// `lodedb_engine_applied_lsn` reflects the durable base plus every WAL record
/// currently on disk.
///
/// # Safety
///
/// `engine` must be a valid engine pointer; `error` may be null or writable.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_refresh(
    engine: *mut LodeEngine,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || engine_mut(engine)?.refresh())
}

/// Writes the highest LSN reflected in `index_id`'s in-memory view to `out_lsn`
/// (the committed base, this handle's own writer mutations, and any WAL records a
/// refresh folded). Compare it to an appender's returned LSN for read-your-writes.
///
/// # Safety
///
/// `engine`, `index_id`, and `out_lsn` must be valid for the duration of the call.
/// `index_id` must contain valid UTF-8 bytes.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_applied_lsn(
    engine: *const LodeEngine,
    index_id: LodeStringView,
    out_lsn: *mut u64,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out_value(out_lsn)?;
        let engine = engine_ref(engine)?;
        let index_id = read_string(index_id)?;
        let lsn = engine.applied_lsn(&index_id)?;
        unsafe {
            *out_lsn = lsn;
        }
        Ok(())
    })
}

/// Deletes documents by id; returns a `CoreMutationResult` JSON to release with
/// `lodedb_owned_string_free`.
///
/// # Safety
///
/// `engine`, `index_id`, `document_ids_json`, and `out` must be valid for the
/// duration of the call. String views must contain valid UTF-8 bytes.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_delete_documents_json(
    engine: *mut LodeEngine,
    index_id: LodeStringView,
    document_ids_json: LodeStringView,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let engine = engine_mut(engine)?;
        let index_id = read_string(index_id)?;
        let ids = read_json_view::<Vec<String>>(document_ids_json)?;
        let result = engine.delete_documents(&index_id, &ids)?;
        let result = owned_json(&result)?;
        unsafe {
            *out = Box::into_raw(result);
        }
        Ok(())
    })
}

/// Updates a document's metadata and/or retained text; returns a
/// `CoreMutationResult` JSON.
///
/// `has_metadata`/`has_text` carry `Option` semantics: when `has_text` is non-zero,
/// `text_json` is parsed as an `Option<String>` (JSON `null` clears the stored text,
/// a string replaces it); when zero, the text is left unchanged. Same for metadata.
///
/// # Safety
///
/// `engine`, `index_id`, `document_id`, `out`, and any present optional JSON views
/// must be valid for the duration of the call. String views must contain valid UTF-8.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_update_document_payload_json(
    engine: *mut LodeEngine,
    index_id: LodeStringView,
    document_id: LodeStringView,
    metadata_json: LodeStringView,
    has_metadata: u8,
    text_json: LodeStringView,
    has_text: u8,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let engine = engine_mut(engine)?;
        let index_id = read_string(index_id)?;
        let document_id = read_string(document_id)?;
        let metadata = if has_metadata == 0 {
            None
        } else {
            Some(read_json_view::<CoreMetadata>(metadata_json)?)
        };
        let text = if has_text == 0 {
            None
        } else {
            Some(read_json_view::<Option<String>>(text_json)?)
        };
        let result = engine.update_document_payload(&index_id, &document_id, metadata, text)?;
        let result = owned_json(&result)?;
        unsafe {
            *out = Box::into_raw(result);
        }
        Ok(())
    })
}

/// Returns metrics-only `CoreEngineStats` JSON for an index.
///
/// # Safety
///
/// `engine`, `index_id`, and `out` must be valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_stats_json(
    engine: *const LodeEngine,
    index_id: LodeStringView,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let engine = engine_ref(engine)?;
        let index_id = read_string(index_id)?;
        let stats = engine.stats(&index_id)?;
        let result = owned_json(&stats)?;
        unsafe {
            *out = Box::into_raw(result);
        }
        Ok(())
    })
}

/// Returns a single document's payload-free record as JSON (`null` if absent).
///
/// # Safety
///
/// `engine`, `index_id`, `document_id`, and `out` must be valid for the duration of
/// the call. String views must contain valid UTF-8 bytes.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_get_document_json(
    engine: *const LodeEngine,
    index_id: LodeStringView,
    document_id: LodeStringView,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let engine = engine_ref(engine)?;
        let index_id = read_string(index_id)?;
        let document_id = read_string(document_id)?;
        let record = engine.get_document(&index_id, &document_id)?;
        let result = owned_json(&record)?;
        unsafe {
            *out = Box::into_raw(result);
        }
        Ok(())
    })
}

/// Returns a document's retained text as JSON `Option<String>` (`null` if none).
///
/// # Safety
///
/// `engine`, `index_id`, `document_id`, and `out` must be valid for the duration of
/// the call. String views must contain valid UTF-8 bytes.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_get_document_text_json(
    engine: *const LodeEngine,
    index_id: LodeStringView,
    document_id: LodeStringView,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let engine = engine_ref(engine)?;
        let index_id = read_string(index_id)?;
        let document_id = read_string(document_id)?;
        let text = engine.get_document_text(&index_id, &document_id)?;
        let result = owned_json(&text)?;
        unsafe {
            *out = Box::into_raw(result);
        }
        Ok(())
    })
}

/// Returns a JSON object mapping document id to retained text for the given ids.
///
/// # Safety
///
/// `engine`, `index_id`, `document_ids_json`, and `out` must be valid for the
/// duration of the call. String views must contain valid UTF-8 bytes.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_get_document_texts_json(
    engine: *const LodeEngine,
    index_id: LodeStringView,
    document_ids_json: LodeStringView,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let engine = engine_ref(engine)?;
        let index_id = read_string(index_id)?;
        let ids = read_json_view::<Vec<String>>(document_ids_json)?;
        let texts = engine.get_document_texts(&index_id, &ids)?;
        let result = owned_json(&texts)?;
        unsafe {
            *out = Box::into_raw(result);
        }
        Ok(())
    })
}

/// Lists payload-free document records as a JSON array, with an optional metadata
/// filter, an `after` id cursor, and a `limit` (each gated by its `has_*` flag).
///
/// # Safety
///
/// `engine`, `index_id`, `out`, and any present optional views must be valid for the
/// duration of the call. String views must contain valid UTF-8 bytes.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_list_documents_json(
    engine: *const LodeEngine,
    index_id: LodeStringView,
    filter_json: LodeStringView,
    has_filter: u8,
    after: LodeStringView,
    has_after: u8,
    limit: usize,
    has_limit: u8,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let engine = engine_ref(engine)?;
        let index_id = read_string(index_id)?;
        let filter = if has_filter == 0 {
            None
        } else {
            Some(read_json_view::<serde_json::Value>(filter_json)?)
        };
        let after_str = if has_after == 0 {
            None
        } else {
            Some(read_string(after)?)
        };
        let limit_opt = if has_limit == 0 { None } else { Some(limit) };
        let documents =
            engine.list_documents(&index_id, filter.as_ref(), after_str.as_deref(), limit_opt)?;
        let result = owned_json(&documents)?;
        unsafe {
            *out = Box::into_raw(result);
        }
        Ok(())
    })
}

/// Returns the engine's loaded index ids as a JSON array of strings.
///
/// # Safety
///
/// `engine` and `out` must be valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_index_ids_json(
    engine: *const LodeEngine,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let engine = engine_ref(engine)?;
        let ids = engine.index_ids();
        let result = owned_json(&ids)?;
        unsafe {
            *out = Box::into_raw(result);
        }
        Ok(())
    })
}

// ---- Batch search + late interaction (Phase 2) ----

/// Batched vector search. `queries_json` is a JSON `[[f32]]`; returns a JSON array
/// of `CoreSearchResults`, one per query, to release with `lodedb_owned_string_free`.
///
/// # Safety
///
/// `engine`, `index_id`, `queries_json`, `out`, and any present filter must be valid
/// for the duration of the call. String views must contain valid UTF-8 bytes.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_query_vectors_batch_json(
    engine: *const LodeEngine,
    index_id: LodeStringView,
    queries_json: LodeStringView,
    top_k: usize,
    filter_json: LodeStringView,
    has_filter: u8,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let engine = engine_ref(engine)?;
        let index_id = read_string(index_id)?;
        let queries = read_json_view::<Vec<Vec<f32>>>(queries_json)?;
        let filter = optional_filter(filter_json, has_filter)?;
        let results = engine.query_vectors_batch(&index_id, &queries, top_k, filter.as_ref())?;
        write_owned_json(out, &results)
    })
}

/// Batched text search. `query_plans_json` is a JSON `[QueryPlan]`; when
/// `has_query_embeddings` is set, `query_embeddings_json` is a JSON `[[f32]]` aligned
/// to the plans. Returns a JSON array of `CoreSearchResults`.
///
/// # Safety
///
/// `engine`, `index_id`, `query_plans_json`, `out`, and any present optional views
/// must be valid for the duration of the call. String views must contain valid UTF-8.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_search_embedded_text_batch_json(
    engine: *const LodeEngine,
    index_id: LodeStringView,
    query_plans_json: LodeStringView,
    query_embeddings_json: LodeStringView,
    has_query_embeddings: u8,
    top_k: usize,
    filter_json: LodeStringView,
    has_filter: u8,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let engine = engine_ref(engine)?;
        let index_id = read_string(index_id)?;
        let query_plans = read_json_view::<Vec<QueryPlan>>(query_plans_json)?;
        let query_embeddings = if has_query_embeddings == 0 {
            None
        } else {
            Some(read_json_view::<Vec<Vec<f32>>>(query_embeddings_json)?)
        };
        let filter = optional_filter(filter_json, has_filter)?;
        let results = engine.search_embedded_text_batch(
            &index_id,
            &query_plans,
            query_embeddings.as_deref(),
            top_k,
            filter.as_ref(),
        )?;
        write_owned_json(out, &results)
    })
}

/// Late-interaction MaxSim query. `query` is a contiguous `(n_query * dim)` f32
/// matrix (row-major, L2-normalized). Returns `CoreSearchResults` JSON.
///
/// # Safety
///
/// `engine`, `index_id`, `out`, and any present filter must be valid for the call.
/// `query` must point to `query_len` initialized f32 values (or be null with len 0).
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_query_multivector_json(
    engine: *const LodeEngine,
    index_id: LodeStringView,
    query: *const c_float,
    query_len: usize,
    n_query: usize,
    top_k: usize,
    filter_json: LodeStringView,
    has_filter: u8,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let engine = engine_ref(engine)?;
        let index_id = read_string(index_id)?;
        let query = read_f32_slice(query, query_len)?;
        let filter = optional_filter(filter_json, has_filter)?;
        let results = engine.query_multivector(&index_id, query, n_query, top_k, filter.as_ref())?;
        write_owned_json(out, &results)
    })
}

/// Sidecar describing one document's stored multi-vector patch matrix, matching the
/// PyO3 multivector upsert contract.
#[derive(serde::Deserialize)]
struct MultiVecSidecar {
    document_id: String,
    #[serde(default)]
    metadata: BTreeMap<String, String>,
    dtype: String,
    patch_count: usize,
    nbytes: usize,
}

/// Upserts multi-vector (late-interaction) documents. `vectors` is a contiguous
/// `(rows * dim)` f32 matrix of per-document anchor vectors; `patch_bytes` is the
/// concatenation of each document's encoded patch matrix (split by the sidecar
/// `nbytes`); `sidecar_json` is a JSON array of `{document_id, metadata?, dtype,
/// patch_count, nbytes}`. Returns a `CoreMutationResult` JSON.
///
/// # Safety
///
/// `engine`, `index_id`, `sidecar_json`, and `out` must be valid for the call.
/// `vectors` / `patch_bytes` must point to the stated number of elements (or be null
/// with length 0). String views must contain valid UTF-8 bytes.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_upsert_multivector_json(
    engine: *mut LodeEngine,
    index_id: LodeStringView,
    vectors: *const c_float,
    rows: usize,
    dim: usize,
    patch_bytes: *const u8,
    patch_bytes_len: usize,
    sidecar_json: LodeStringView,
    out: *mut *mut LodeOwnedString,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let engine = engine_mut(engine)?;
        let index_id = read_string(index_id)?;
        let vectors = read_f32_slice(vectors, rows.saturating_mul(dim))?;
        let all_bytes = read_u8_slice(patch_bytes, patch_bytes_len)?;
        let sidecars = read_json_view::<Vec<MultiVecSidecar>>(sidecar_json)?;
        if sidecars.len() != rows {
            return invalid("sidecar count does not match vector rows");
        }
        if rows > 0 && dim == 0 {
            return invalid("vectors must have a non-zero dimension");
        }
        let total: usize = sidecars.iter().map(|sidecar| sidecar.nbytes).sum();
        if total != all_bytes.len() {
            return invalid("patch_bytes length does not match sidecar nbytes sum");
        }
        let mut offset = 0usize;
        let mut documents: Vec<CoreVectorDocument> = Vec::with_capacity(rows);
        for (sidecar, vector) in sidecars.into_iter().zip(vectors.chunks(dim.max(1))) {
            // Fail closed on malformed patch layouts rather than persisting bytes the
            // MaxSim decoder would later silently truncate or misread.
            if sidecar.patch_count == 0 {
                return invalid("patch_count must be positive");
            }
            if sidecar.nbytes != expected_patch_nbytes(&sidecar.dtype, sidecar.patch_count, dim)? {
                return invalid("patch nbytes does not match dtype, patch_count, and dim");
            }
            let stop = offset + sidecar.nbytes;
            let patch_matrix = lodedb_core::storage::multivec_store::MultiVecRecord {
                dtype: sidecar.dtype,
                patch_count: sidecar.patch_count,
                bytes: all_bytes[offset..stop].to_vec(),
            };
            offset = stop;
            documents.push(CoreVectorDocument {
                document_id: sidecar.document_id,
                vector: vector.to_vec(),
                metadata: sidecar.metadata,
                text: None,
                patch_matrix: Some(patch_matrix),
            });
        }
        let result = engine.upsert_vectors(&index_id, &documents)?;
        write_owned_json(out, &result)
    })
}

// ---- Concurrent multi-writer append ----

/// Opens a shared-lock appender over the single index at `options.path`.
///
/// Many processes can hold an appender at once and durably log vector-in records
/// to the store's WAL concurrently; the next exclusive writer folds them into the
/// index on open. The store must be in WAL commit mode and hold exactly one
/// index. The returned handle must be released with `lodedb_appender_free`.
///
/// # Safety
///
/// `options_json` and `out` must be valid for the duration of the call;
/// `options_json` must contain valid UTF-8 JSON.
#[no_mangle]
pub unsafe extern "C" fn lodedb_appender_open_json(
    options_json: LodeStringView,
    out: *mut *mut LodeAppender,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out(out)?;
        let options = read_json_view::<CoreOpenOptions>(options_json)?;
        let appender = CoreAppender::open(options)?;
        let handle = Box::new(LodeAppender { appender });
        unsafe {
            *out = Box::into_raw(handle);
        }
        Ok(())
    })
}

/// Frees an appender handle allocated by this library.
///
/// # Safety
///
/// `appender` must be null or a pointer returned by `lodedb_appender_open_json`.
/// It must not be used after this call and must not be freed more than once.
#[no_mangle]
pub unsafe extern "C" fn lodedb_appender_free(appender: *mut LodeAppender) {
    if !appender.is_null() {
        let _ = Box::from_raw(appender);
    }
}

/// Durably appends one `upsert_vectors` record from a JSON `CoreVectorDocument`
/// array (vector plus metadata; raw text is not logged) and writes the assigned
/// LSN to `out_lsn`.
///
/// # Safety
///
/// `appender`, `documents_json`, and `out_lsn` must be valid for the duration of
/// the call. `documents_json` must contain valid UTF-8 JSON.
#[no_mangle]
pub unsafe extern "C" fn lodedb_appender_append_vectors_json(
    appender: *const LodeAppender,
    documents_json: LodeStringView,
    out_lsn: *mut u64,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out_value(out_lsn)?;
        let appender = appender_ref(appender)?;
        let documents = read_json_view::<Vec<CoreVectorDocument>>(documents_json)?;
        let lsn = appender.append_vectors(&documents)?;
        unsafe {
            *out_lsn = lsn;
        }
        Ok(())
    })
}

/// Durably appends one `delete_documents` record from a JSON array of document
/// ids and writes the assigned LSN to `out_lsn`.
///
/// # Safety
///
/// `appender`, `document_ids_json`, and `out_lsn` must be valid for the duration
/// of the call. `document_ids_json` must contain valid UTF-8 JSON.
#[no_mangle]
pub unsafe extern "C" fn lodedb_appender_append_deletes_json(
    appender: *const LodeAppender,
    document_ids_json: LodeStringView,
    out_lsn: *mut u64,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        require_out_value(out_lsn)?;
        let appender = appender_ref(appender)?;
        let ids = read_json_view::<Vec<String>>(document_ids_json)?;
        let lsn = appender.append_deletes(&ids)?;
        unsafe {
            *out_lsn = lsn;
        }
        Ok(())
    })
}

fn ffi_result(error: *mut *mut LodeError, f: impl FnOnce() -> Result<(), CoreError>) -> u32 {
    clear_error(error);
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        Ok(Err(err)) => {
            set_error(error, &err);
            err.code().ffi_status_code()
        }
        Err(_) => {
            let err = CoreError::new(CoreErrorCode::Internal, "native core panic was caught");
            set_error(error, &err);
            err.code().ffi_status_code()
        }
    }
}

fn clear_error(error: *mut *mut LodeError) {
    if !error.is_null() {
        unsafe {
            *error = ptr::null_mut();
        }
    }
}

fn set_error(error: *mut *mut LodeError, core: &CoreError) {
    if error.is_null() {
        return;
    }
    let message = c_string(core.message())
        .unwrap_or_else(|_| CString::new("native core error").expect("static string"));
    let ffi_error = Box::new(LodeError {
        size: std::mem::size_of::<LodeError>() as u32,
        version: ABI_VERSION,
        code: core.code().ffi_status_code(),
        message: message.into_raw(),
    });
    unsafe {
        *error = Box::into_raw(ffi_error);
    }
}

fn require_out<T>(out: *mut *mut T) -> Result<(), CoreError> {
    if out.is_null() {
        return invalid("output pointer is null");
    }
    Ok(())
}

fn require_out_value(out: *mut u64) -> Result<(), CoreError> {
    if out.is_null() {
        return invalid("output pointer is null");
    }
    Ok(())
}

fn engine_mut<'a>(engine: *mut LodeEngine) -> Result<&'a mut CoreEngine, CoreError> {
    if engine.is_null() {
        return invalid("engine pointer is null");
    }
    Ok(unsafe { &mut (*engine).engine })
}

fn engine_ref<'a>(engine: *const LodeEngine) -> Result<&'a CoreEngine, CoreError> {
    if engine.is_null() {
        return invalid("engine pointer is null");
    }
    Ok(unsafe { &(*engine).engine })
}

fn appender_ref<'a>(appender: *const LodeAppender) -> Result<&'a CoreAppender, CoreError> {
    if appender.is_null() {
        return invalid("appender pointer is null");
    }
    Ok(unsafe { &(*appender).appender })
}

fn read_string(view: LodeStringView) -> Result<String, CoreError> {
    validate_abi_header(
        view.size,
        view.version,
        std::mem::size_of::<LodeStringView>(),
        "string view",
    )?;
    if view.data.is_null() {
        if view.len == 0 {
            return Ok(String::new());
        }
        return invalid("string data pointer is null");
    }
    let bytes = unsafe { slice::from_raw_parts(view.data.cast::<u8>(), view.len) };
    std::str::from_utf8(bytes)
        .map(str::to_string)
        .map_err(|_| CoreError::new(CoreErrorCode::InvalidArgument, "string is not valid UTF-8"))
}

fn read_json_view<T: serde::de::DeserializeOwned>(view: LodeStringView) -> Result<T, CoreError> {
    let text = read_string(view)?;
    serde_json::from_str(&text).map_err(|error| {
        CoreError::new(
            CoreErrorCode::InvalidArgument,
            format!("invalid JSON payload: {error}"),
        )
    })
}

fn owned_json<T: serde::Serialize>(value: &T) -> Result<Box<LodeOwnedString>, CoreError> {
    let text = serde_json::to_string(value).map_err(|error| {
        CoreError::new(
            CoreErrorCode::Internal,
            format!("failed to serialize core value: {error}"),
        )
    })?;
    owned_string(text)
}

fn owned_string(text: String) -> Result<Box<LodeOwnedString>, CoreError> {
    let len = text.len();
    Ok(Box::new(LodeOwnedString {
        size: std::mem::size_of::<LodeOwnedString>() as u32,
        version: ABI_VERSION,
        data: c_string(text)?.into_raw(),
        len,
    }))
}

fn read_f32_slice<'a>(data: *const c_float, len: usize) -> Result<&'a [f32], CoreError> {
    if data.is_null() && len > 0 {
        return invalid("f32 data pointer is null");
    }
    Ok(unsafe { slice::from_raw_parts(data, len) })
}

fn read_u8_slice<'a>(data: *const u8, len: usize) -> Result<&'a [u8], CoreError> {
    if data.is_null() && len > 0 {
        return invalid("byte data pointer is null");
    }
    Ok(unsafe { slice::from_raw_parts(data, len) })
}

/// Expected encoded byte length of one document's patch matrix, matching
/// `MultiVecRecord::decode`: little-endian f4/f2 for float32/float16, or per-patch
/// f32 scale (4 bytes) plus `dim` int8 codes for int8.
fn expected_patch_nbytes(dtype: &str, patch_count: usize, dim: usize) -> Result<usize, CoreError> {
    let size = match dtype {
        "float32" => patch_count.checked_mul(dim).and_then(|n| n.checked_mul(4)),
        "float16" => patch_count.checked_mul(dim).and_then(|n| n.checked_mul(2)),
        "int8" => dim.checked_add(4).and_then(|per| patch_count.checked_mul(per)),
        other => {
            return Err(CoreError::new(
                CoreErrorCode::InvalidArgument,
                format!("unsupported patch dtype '{other}'"),
            ))
        }
    };
    size.ok_or_else(|| CoreError::new(CoreErrorCode::InvalidArgument, "patch matrix size overflow"))
}

fn optional_filter(
    filter_json: LodeStringView,
    has_filter: u8,
) -> Result<Option<serde_json::Value>, CoreError> {
    if has_filter == 0 {
        Ok(None)
    } else {
        Ok(Some(read_json_view::<serde_json::Value>(filter_json)?))
    }
}

/// Serializes `value` to JSON and writes it to the owned-string out-parameter.
///
/// # Safety
///
/// `out` must be a valid writable pointer (callers pass `require_out`-checked `out`s).
fn write_owned_json<T: serde::Serialize>(
    out: *mut *mut LodeOwnedString,
    value: &T,
) -> Result<(), CoreError> {
    let result = owned_json(value)?;
    unsafe {
        *out = Box::into_raw(result);
    }
    Ok(())
}

fn read_vector_documents(
    documents: *const LodeVectorDocument,
    documents_len: usize,
) -> Result<Vec<CoreVectorDocument>, CoreError> {
    if documents.is_null() && documents_len > 0 {
        return invalid("documents pointer is null");
    }
    let documents = unsafe { slice::from_raw_parts(documents, documents_len) };
    documents
        .iter()
        .map(|document| {
            validate_abi_header(
                document.size,
                document.version,
                std::mem::size_of::<LodeVectorDocument>(),
                "vector document",
            )?;
            let vector = read_f32_slice(document.vector, document.vector_len)?.to_vec();
            Ok(CoreVectorDocument {
                document_id: read_string(document.document_id)?,
                vector,
                metadata: read_metadata(document.metadata, document.metadata_len)?,
                text: if document.has_text == 0 {
                    None
                } else {
                    Some(read_string(document.text)?)
                },
                // Plain vector documents carry no late-interaction multivector;
                // the multivec record is populated only on the multivector path.
                patch_matrix: None,
            })
        })
        .collect()
}

fn read_metadata(
    metadata: *const LodeMetadataPair,
    metadata_len: usize,
) -> Result<BTreeMap<String, String>, CoreError> {
    if metadata.is_null() && metadata_len > 0 {
        return invalid("metadata pointer is null");
    }
    let pairs = unsafe { slice::from_raw_parts(metadata, metadata_len) };
    pairs
        .iter()
        .map(|pair| {
            validate_abi_header(
                pair.size,
                pair.version,
                std::mem::size_of::<LodeMetadataPair>(),
                "metadata pair",
            )?;
            Ok((read_string(pair.key)?, read_string(pair.value)?))
        })
        .collect()
}

fn c_string(text: impl AsRef<str>) -> Result<CString, CoreError> {
    CString::new(text.as_ref())
        .map_err(|_| CoreError::new(CoreErrorCode::InvalidArgument, "string contains NUL byte"))
}

fn invalid<T>(message: impl Into<String>) -> Result<T, CoreError> {
    Err(CoreError::new(CoreErrorCode::InvalidArgument, message))
}

fn validate_abi_header(
    size: u32,
    version: u32,
    expected_size: usize,
    context: &str,
) -> Result<(), CoreError> {
    if usize::try_from(size).ok() != Some(expected_size) {
        return invalid(format!("{context} ABI size mismatch"));
    }
    if version != ABI_VERSION {
        return invalid(format!("{context} ABI version mismatch"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_struct_versions_start_each_public_struct() {
        assert_eq!(std::mem::offset_of!(LodeError, size), 0);
        assert_eq!(std::mem::offset_of!(LodeError, version), 4);
        assert_eq!(std::mem::offset_of!(LodeOwnedString, size), 0);
        assert_eq!(std::mem::offset_of!(LodeOwnedString, version), 4);
        assert_eq!(std::mem::offset_of!(LodeSearchRequest, size), 0);
        assert_eq!(std::mem::offset_of!(LodeSearchRequest, version), 4);
        assert_eq!(std::mem::offset_of!(LodeSearchHit, size), 0);
        assert_eq!(std::mem::offset_of!(LodeSearchResults, size), 0);
    }

    #[test]
    fn invalid_search_request_version_returns_ffi_error() {
        let mut error: *mut LodeError = ptr::null_mut();
        let mut engine: *mut LodeEngine = ptr::null_mut();
        let index_id = string_view("default");

        unsafe {
            assert_eq!(lodedb_engine_new_in_memory(&mut engine, &mut error), 0);
            assert_eq!(create_default_index(engine, &mut error), 0);
        }

        let query = [1.0_f32, 0.0_f32];
        let request = LodeSearchRequest {
            size: std::mem::size_of::<LodeSearchRequest>() as u32,
            version: ABI_VERSION + 1,
            index_id,
            query: query.as_ptr(),
            query_len: query.len(),
            top_k: 1,
        };
        let mut results: *mut LodeSearchResults = ptr::null_mut();
        let status =
            unsafe { lodedb_engine_query_vector(engine, &request, &mut results, &mut error) };

        assert_eq!(status, CoreErrorCode::InvalidArgument.ffi_status_code());
        assert!(results.is_null());
        assert!(!error.is_null());
        let message = unsafe { std::ffi::CStr::from_ptr((*error).message) }
            .to_string_lossy()
            .into_owned();
        assert!(message.contains("ABI version mismatch"));
        unsafe {
            lodedb_error_free(error);
            lodedb_engine_free(engine);
        }
    }

    #[test]
    fn invalid_vector_document_size_returns_ffi_error() {
        let mut error: *mut LodeError = ptr::null_mut();
        let mut engine: *mut LodeEngine = ptr::null_mut();
        let index_id = string_view("default");
        unsafe {
            assert_eq!(lodedb_engine_new_in_memory(&mut engine, &mut error), 0);
            assert_eq!(create_default_index(engine, &mut error), 0);
        }
        let vector = [1.0_f32, 0.0_f32];
        let document = LodeVectorDocument {
            size: 0,
            version: ABI_VERSION,
            document_id: string_view("doc-a"),
            vector: vector.as_ptr(),
            vector_len: vector.len(),
            metadata: ptr::null(),
            metadata_len: 0,
            text: string_view(""),
            has_text: 0,
        };

        let status =
            unsafe { lodedb_engine_upsert_vectors(engine, index_id, &document, 1, &mut error) };

        assert_eq!(status, CoreErrorCode::InvalidArgument.ffi_status_code());
        assert!(!error.is_null());
        let message = unsafe { std::ffi::CStr::from_ptr((*error).message) }
            .to_string_lossy()
            .into_owned();
        assert!(message.contains("ABI size mismatch"));
        unsafe {
            lodedb_error_free(error);
            lodedb_engine_free(engine);
        }
    }

    fn string_view(text: &'static str) -> LodeStringView {
        LodeStringView {
            size: std::mem::size_of::<LodeStringView>() as u32,
            version: ABI_VERSION,
            data: text.as_ptr().cast::<c_char>(),
            len: text.len(),
        }
    }

    /// Creates the shared `default` (dim 8, bit width 4) exact index through the
    /// JSON create ABI, the single create entry point.
    unsafe fn create_default_index(engine: *mut LodeEngine, error: *mut *mut LodeError) -> u32 {
        lodedb_engine_create_index_json(
            engine,
            string_view(r#"{"index_id":"default","vector_dim":8,"bit_width":4}"#),
            error,
        )
    }

    static NEXT_APPENDER_DIR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    fn appender_temp_dir() -> std::path::PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "lodedb-ffi-appender-{}-{}",
            std::process::id(),
            NEXT_APPENDER_DIR.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn wal_options_json(dir: &std::path::Path) -> String {
        serde_json::json!({
            "path": dir.to_str().expect("utf-8 temp path"),
            "read_only": false,
            "durability": "buffered",
            "commit_mode": "wal",
            "store_text": false,
            "index_text": false,
            "acquire_writer_lock": true,
        })
        .to_string()
    }

    /// A `LodeStringView` over a non-static string. The caller must keep `text`
    /// alive for as long as the view is used (the view borrows its bytes).
    fn borrowed_view(text: &str) -> LodeStringView {
        LodeStringView {
            size: std::mem::size_of::<LodeStringView>() as u32,
            version: ABI_VERSION,
            data: text.as_ptr().cast::<c_char>(),
            len: text.len(),
        }
    }

    unsafe fn owned_to_string(owned: *mut LodeOwnedString) -> String {
        let owned = &*owned;
        let bytes = slice::from_raw_parts(owned.data.cast::<u8>(), owned.len);
        String::from_utf8_lossy(bytes).into_owned()
    }

    #[test]
    fn appender_ffi_round_trips_through_a_durable_store() {
        let dir = appender_temp_dir();
        let options = wal_options_json(&dir);
        let options_view = borrowed_view(&options); // `options` outlives every use below

        // A writer creates the index and checkpoints an empty base, then closes so
        // the shared appender can take the lock.
        unsafe {
            let mut error: *mut LodeError = ptr::null_mut();
            let mut engine: *mut LodeEngine = ptr::null_mut();
            assert_eq!(
                lodedb_engine_open_json(options_view, &mut engine, &mut error),
                0
            );
            assert!(!engine.is_null());
            assert_eq!(create_default_index(engine, &mut error), 0);
            assert_eq!(lodedb_engine_persist(engine, &mut error), 0);
            lodedb_engine_free(engine);
        }

        // The appender logs a vector-in record, then a delete, each handing back an
        // LSN; the delete's LSN is above the append's.
        unsafe {
            let mut error: *mut LodeError = ptr::null_mut();
            let mut appender: *mut LodeAppender = ptr::null_mut();
            assert_eq!(
                lodedb_appender_open_json(options_view, &mut appender, &mut error),
                0
            );
            assert!(!appender.is_null());
            let documents = r#"[{"document_id":"doc-a","vector":[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"metadata":{"topic":"ops"},"text":null}]"#;
            let mut lsn: u64 = 0;
            assert_eq!(
                lodedb_appender_append_vectors_json(
                    appender,
                    string_view(documents),
                    &mut lsn,
                    &mut error
                ),
                0
            );
            assert!(lsn > 0, "expected a positive LSN, got {lsn}");
            let mut delete_lsn: u64 = 0;
            assert_eq!(
                lodedb_appender_append_deletes_json(
                    appender,
                    string_view(r#"["doc-a"]"#),
                    &mut delete_lsn,
                    &mut error
                ),
                0
            );
            assert!(
                delete_lsn > lsn,
                "delete LSN {delete_lsn} not above append {lsn}"
            );
            lodedb_appender_free(appender);
        }

        // A null appender is rejected as an invalid argument, not a crash.
        unsafe {
            let mut error: *mut LodeError = ptr::null_mut();
            let mut lsn: u64 = 0;
            let status = lodedb_appender_append_vectors_json(
                ptr::null(),
                string_view("[]"),
                &mut lsn,
                &mut error,
            );
            assert_eq!(status, CoreErrorCode::InvalidArgument.ffi_status_code());
            lodedb_error_free(error);
        }

        // The next writer folds the appended-then-deleted record in: doc-a is gone.
        unsafe {
            let mut error: *mut LodeError = ptr::null_mut();
            let mut engine: *mut LodeEngine = ptr::null_mut();
            assert_eq!(
                lodedb_engine_open_json(options_view, &mut engine, &mut error),
                0
            );
            let mut stats: *mut LodeOwnedString = ptr::null_mut();
            assert_eq!(
                lodedb_engine_stats_json(engine, string_view("default"), &mut stats, &mut error),
                0
            );
            let stats_json = owned_to_string(stats);
            assert!(
                stats_json.contains("\"document_count\":0"),
                "expected doc-a folded in then deleted: {stats_json}"
            );
            lodedb_owned_string_free(stats);
            lodedb_engine_free(engine);
        }

        std::fs::remove_dir_all(dir).unwrap();
    }
}
