//! C ABI for the native LodeDB core.

use lodedb_core::engine::CoreEngine;
use lodedb_core::types::CoreVectorDocument;
use lodedb_core::{CoreError, CoreErrorCode};
use std::collections::BTreeMap;
use std::ffi::{c_char, CString};
use std::os::raw::c_float;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;

const ABI_VERSION: u32 = 1;

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
#[derive(Clone, Copy)]
pub struct LodeStringView {
    size: u32,
    version: u32,
    data: *const c_char,
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

/// Creates a vector index on an engine.
///
/// # Safety
///
/// `engine` must be a valid engine pointer. `index_id` must reference valid
/// UTF-8 bytes for the duration of the call. `error` may be null or writable.
#[no_mangle]
pub unsafe extern "C" fn lodedb_engine_create_index(
    engine: *mut LodeEngine,
    index_id: LodeStringView,
    vector_dim: usize,
    bit_width: usize,
    error: *mut *mut LodeError,
) -> u32 {
    ffi_result(error, || {
        let engine = engine_mut(engine)?;
        engine.create_index(read_string(index_id)?, vector_dim, bit_width)
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

fn read_string(view: LodeStringView) -> Result<String, CoreError> {
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

fn read_f32_slice<'a>(data: *const c_float, len: usize) -> Result<&'a [f32], CoreError> {
    if data.is_null() && len > 0 {
        return invalid("f32 data pointer is null");
    }
    Ok(unsafe { slice::from_raw_parts(data, len) })
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
        .map(|pair| Ok((read_string(pair.key)?, read_string(pair.value)?)))
        .collect()
}

fn c_string(text: impl AsRef<str>) -> Result<CString, CoreError> {
    CString::new(text.as_ref())
        .map_err(|_| CoreError::new(CoreErrorCode::InvalidArgument, "string contains NUL byte"))
}

fn invalid<T>(message: impl Into<String>) -> Result<T, CoreError> {
    Err(CoreError::new(CoreErrorCode::InvalidArgument, message))
}

#[cfg(test)]
mod tests {
    use super::{LodeError, LodeSearchHit, LodeSearchRequest, LodeSearchResults};

    #[test]
    fn abi_struct_versions_start_each_public_struct() {
        assert_eq!(std::mem::offset_of!(LodeError, size), 0);
        assert_eq!(std::mem::offset_of!(LodeError, version), 4);
        assert_eq!(std::mem::offset_of!(LodeSearchRequest, size), 0);
        assert_eq!(std::mem::offset_of!(LodeSearchRequest, version), 4);
        assert_eq!(std::mem::offset_of!(LodeSearchHit, size), 0);
        assert_eq!(std::mem::offset_of!(LodeSearchResults, size), 0);
    }
}
