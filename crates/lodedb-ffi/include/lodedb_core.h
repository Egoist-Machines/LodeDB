#ifndef LODEDB_CORE_H
#define LODEDB_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LODEDB_ABI_VERSION 1u

typedef enum LodeStatus {
  LODE_OK = 0,
  LODE_INVALID_ARGUMENT = 1,
  LODE_NOT_FOUND = 2,
  LODE_CORRUPT_STORE = 3,
  LODE_PLAN_STALE = 4,
  LODE_UNSUPPORTED = 5,
  LODE_INTERNAL = 255
} LodeStatus;

typedef struct LodeError {
  uint32_t size;
  uint32_t version;
  uint32_t code;
  const char *message;
} LodeError;

typedef struct LodeEngine LodeEngine;

typedef struct LodeStringView {
  uint32_t size;
  uint32_t version;
  const char *data;
  uintptr_t len;
} LodeStringView;

typedef struct LodeOwnedString {
  uint32_t size;
  uint32_t version;
  char *data;
  uintptr_t len;
} LodeOwnedString;

typedef struct LodeMetadataPair {
  uint32_t size;
  uint32_t version;
  LodeStringView key;
  LodeStringView value;
} LodeMetadataPair;

typedef struct LodeVectorDocument {
  uint32_t size;
  uint32_t version;
  LodeStringView document_id;
  const float *vector;
  uintptr_t vector_len;
  const LodeMetadataPair *metadata;
  uintptr_t metadata_len;
  LodeStringView text;
  uint8_t has_text;
} LodeVectorDocument;

typedef struct LodeSearchRequest {
  uint32_t size;
  uint32_t version;
  LodeStringView index_id;
  const float *query;
  uintptr_t query_len;
  uintptr_t top_k;
} LodeSearchRequest;

typedef struct LodeSearchHit {
  uint32_t size;
  uint32_t version;
  const char *document_id;
  const char *chunk_id;
  float score;
} LodeSearchHit;

typedef struct LodeSearchResults {
  uint32_t size;
  uint32_t version;
  LodeSearchHit *hits;
  uintptr_t hits_len;
  uintptr_t total_considered;
} LodeSearchResults;

uint32_t lodedb_abi_version(void);
void lodedb_error_free(LodeError *error);
void lodedb_owned_string_free(LodeOwnedString *text);
void lodedb_search_results_free(LodeSearchResults *results);

uint32_t lodedb_engine_new_in_memory(LodeEngine **out, LodeError **error);
void lodedb_engine_free(LodeEngine *engine);
uint32_t lodedb_engine_create_index(
    LodeEngine *engine,
    LodeStringView index_id,
    uintptr_t vector_dim,
    uintptr_t bit_width,
    LodeError **error);
uint32_t lodedb_engine_upsert_vectors(
    LodeEngine *engine,
    LodeStringView index_id,
    const LodeVectorDocument *documents,
    uintptr_t documents_len,
    LodeError **error);
uint32_t lodedb_engine_query_vector(
    const LodeEngine *engine,
    const LodeSearchRequest *request,
    LodeSearchResults **out,
    LodeError **error);
uint32_t lodedb_engine_prepare_text_upsert_json(
    LodeEngine *engine,
    LodeStringView index_id,
    LodeStringView documents_json,
    uint8_t store_text,
    uint8_t index_text,
    uintptr_t chunk_character_limit,
    LodeOwnedString **out,
    LodeError **error);
uint32_t lodedb_engine_apply_text_upsert_json(
    LodeEngine *engine,
    LodeStringView plan_json,
    LodeStringView embeddings_json,
    double embedding_time_ms,
    LodeOwnedString **out,
    LodeError **error);

#ifdef __cplusplus
}
#endif

#endif
