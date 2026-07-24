#ifndef LODEDB_CORE_H
#define LODEDB_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LODEDB_ABI_VERSION 5u

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

typedef struct LodeAppender LodeAppender;

typedef struct LodeCheckpointer LodeCheckpointer;

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
/* Single create entry point. options_json is a minimal CoreIndexCreateRequest:
 * {"index_id": str, "vector_dim": int, "bit_width"?: int (default 4),
 *  "model"?: str, "ann"?: {...}}. The core supplies the identity defaults. */
uint32_t lodedb_engine_create_index_json(
    LodeEngine *engine,
    LodeStringView options_json,
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
uint32_t lodedb_engine_prepare_query_text_json(
    const LodeEngine *engine,
    LodeStringView query,
    LodeStringView mode,
    LodeOwnedString **out,
    LodeError **error);
uint32_t lodedb_engine_search_embedded_text_json(
    const LodeEngine *engine,
    LodeStringView index_id,
    LodeStringView query_plan_json,
    LodeStringView query_embedding_json,
    uint8_t has_query_embedding,
    uintptr_t top_k,
    LodeStringView filter_json,
    uint8_t has_filter,
    LodeOwnedString **out,
    LodeError **error);
uint32_t lodedb_engine_query_vector_json(
    const LodeEngine *engine,
    LodeStringView index_id,
    const float *query,
    uintptr_t query_len,
    uintptr_t top_k,
    LodeStringView filter_json,
    uint8_t has_filter,
    LodeOwnedString **out,
    LodeError **error);
uint32_t lodedb_engine_upsert_vectors_json(
    LodeEngine *engine,
    LodeStringView index_id,
    LodeStringView documents_json,
    LodeError **error);

uint32_t lodedb_engine_open_json(
    LodeStringView options_json,
    LodeEngine **out,
    LodeError **error);
uint32_t lodedb_engine_open_readonly_json(
    LodeStringView options_json,
    LodeEngine **out,
    LodeError **error);
uint32_t lodedb_engine_persist(LodeEngine *engine, LodeError **error);
uint32_t lodedb_engine_close(LodeEngine *engine, LodeError **error);
uint32_t lodedb_engine_refresh(LodeEngine *engine, LodeError **error);
uint32_t lodedb_engine_applied_lsn(
    const LodeEngine *engine,
    LodeStringView index_id,
    uint64_t *out_lsn,
    LodeError **error);
uint32_t lodedb_engine_delete_documents_json(
    LodeEngine *engine,
    LodeStringView index_id,
    LodeStringView document_ids_json,
    LodeOwnedString **out,
    LodeError **error);
uint32_t lodedb_engine_update_document_payload_json(
    LodeEngine *engine,
    LodeStringView index_id,
    LodeStringView document_id,
    LodeStringView metadata_json,
    uint8_t has_metadata,
    LodeStringView text_json,
    uint8_t has_text,
    LodeOwnedString **out,
    LodeError **error);
uint32_t lodedb_engine_stats_json(
    const LodeEngine *engine,
    LodeStringView index_id,
    LodeOwnedString **out,
    LodeError **error);
uint32_t lodedb_engine_get_document_json(
    const LodeEngine *engine,
    LodeStringView index_id,
    LodeStringView document_id,
    LodeOwnedString **out,
    LodeError **error);
uint32_t lodedb_engine_get_document_text_json(
    const LodeEngine *engine,
    LodeStringView index_id,
    LodeStringView document_id,
    LodeOwnedString **out,
    LodeError **error);
uint32_t lodedb_engine_get_document_texts_json(
    const LodeEngine *engine,
    LodeStringView index_id,
    LodeStringView document_ids_json,
    LodeOwnedString **out,
    LodeError **error);
uint32_t lodedb_engine_list_documents_json(
    const LodeEngine *engine,
    LodeStringView index_id,
    LodeStringView filter_json,
    uint8_t has_filter,
    LodeStringView after,
    uint8_t has_after,
    uintptr_t limit,
    uint8_t has_limit,
    LodeOwnedString **out,
    LodeError **error);
uint32_t lodedb_engine_index_ids_json(
    const LodeEngine *engine,
    LodeOwnedString **out,
    LodeError **error);
uint32_t lodedb_engine_query_vectors_batch_json(
    const LodeEngine *engine,
    LodeStringView index_id,
    LodeStringView queries_json,
    uintptr_t top_k,
    LodeStringView filter_json,
    uint8_t has_filter,
    LodeOwnedString **out,
    LodeError **error);
uint32_t lodedb_engine_search_embedded_text_batch_json(
    const LodeEngine *engine,
    LodeStringView index_id,
    LodeStringView query_plans_json,
    LodeStringView query_embeddings_json,
    uint8_t has_query_embeddings,
    uintptr_t top_k,
    LodeStringView filter_json,
    uint8_t has_filter,
    LodeOwnedString **out,
    LodeError **error);
uint32_t lodedb_engine_query_multivector_json(
    const LodeEngine *engine,
    LodeStringView index_id,
    const float *query,
    uintptr_t query_len,
    uintptr_t n_query,
    uintptr_t top_k,
    LodeStringView filter_json,
    uint8_t has_filter,
    LodeOwnedString **out,
    LodeError **error);
uint32_t lodedb_engine_upsert_multivector_json(
    LodeEngine *engine,
    LodeStringView index_id,
    const float *vectors,
    uintptr_t rows,
    uintptr_t dim,
    const uint8_t *patch_bytes,
    uintptr_t patch_bytes_len,
    LodeStringView sidecar_json,
    LodeOwnedString **out,
    LodeError **error);

uint32_t lodedb_appender_open_json(
    LodeStringView options_json,
    LodeAppender **out,
    LodeError **error);
void lodedb_appender_free(LodeAppender *appender);
uint32_t lodedb_appender_append_vectors_json(
    const LodeAppender *appender,
    LodeStringView documents_json,
    uint64_t *out_lsn,
    LodeError **error);
uint32_t lodedb_appender_append_deletes_json(
    const LodeAppender *appender,
    LodeStringView document_ids_json,
    uint64_t *out_lsn,
    LodeError **error);
uint32_t lodedb_appender_prepare_documents_json(
    const LodeAppender *appender,
    LodeStringView documents_json,
    LodeOwnedString **out,
    LodeError **error);
uint32_t lodedb_appender_append_embedded_documents_json(
    const LodeAppender *appender,
    LodeStringView plan_json,
    LodeStringView embeddings_json,
    uint64_t *out_lsn,
    LodeError **error);

uint32_t lodedb_checkpointer_open_json(
    LodeStringView options_json,
    LodeCheckpointer **out,
    LodeError **error);
void lodedb_checkpointer_free(LodeCheckpointer *checkpointer);
uint32_t lodedb_checkpointer_checkpoint(
    LodeCheckpointer *checkpointer,
    uint64_t *out_folded,
    LodeError **error);

/* --- Bi-temporal knowledge graph (lodedb-graph) -------------------------
   Opened without an embedder: the caller (e.g. Swift LodeGraph) embeds
   label/fact/query text on device and passes vectors to the vector-in verbs.
   Every verb below takes one JSON request LodeStringView and writes one JSON
   response LodeOwnedString, using the same status/error contract as the engine. */
typedef struct LodeGraph LodeGraph;

uint32_t lodedb_graph_open_json(
    LodeStringView request,
    LodeGraph **out,
    LodeError **error);
void lodedb_graph_free(LodeGraph *graph);

uint32_t lodedb_graph_add_episode_json(
    LodeGraph *graph, LodeStringView request, LodeOwnedString **out, LodeError **error);
uint32_t lodedb_graph_upsert_entity_vec_json(
    LodeGraph *graph, LodeStringView request, LodeOwnedString **out, LodeError **error);
uint32_t lodedb_graph_add_fact_vec_json(
    LodeGraph *graph, LodeStringView request, LodeOwnedString **out, LodeError **error);
uint32_t lodedb_graph_invalidate_fact_json(
    LodeGraph *graph, LodeStringView request, LodeOwnedString **out, LodeError **error);
uint32_t lodedb_graph_remove_entity_json(
    LodeGraph *graph, LodeStringView request, LodeOwnedString **out, LodeError **error);
uint32_t lodedb_graph_remove_fact_json(
    LodeGraph *graph, LodeStringView request, LodeOwnedString **out, LodeError **error);
uint32_t lodedb_graph_get_entity_json(
    LodeGraph *graph, LodeStringView request, LodeOwnedString **out, LodeError **error);
uint32_t lodedb_graph_get_fact_json(
    LodeGraph *graph, LodeStringView request, LodeOwnedString **out, LodeError **error);
uint32_t lodedb_graph_get_episode_json(
    LodeGraph *graph, LodeStringView request, LodeOwnedString **out, LodeError **error);
uint32_t lodedb_graph_entities_json(
    LodeGraph *graph, LodeStringView request, LodeOwnedString **out, LodeError **error);
uint32_t lodedb_graph_history_json(
    LodeGraph *graph, LodeStringView request, LodeOwnedString **out, LodeError **error);
uint32_t lodedb_graph_neighbors_json(
    LodeGraph *graph, LodeStringView request, LodeOwnedString **out, LodeError **error);
uint32_t lodedb_graph_k_hop_json(
    LodeGraph *graph, LodeStringView request, LodeOwnedString **out, LodeError **error);
uint32_t lodedb_graph_semantic_entities_json(
    LodeGraph *graph, LodeStringView request, LodeOwnedString **out, LodeError **error);
uint32_t lodedb_graph_semantic_facts_json(
    LodeGraph *graph, LodeStringView request, LodeOwnedString **out, LodeError **error);
uint32_t lodedb_graph_search_subgraph_json(
    LodeGraph *graph, LodeStringView request, LodeOwnedString **out, LodeError **error);
uint32_t lodedb_graph_reindex_json(
    LodeGraph *graph, LodeOwnedString **out, LodeError **error);
uint32_t lodedb_graph_stats_json(
    const LodeGraph *graph, LodeOwnedString **out, LodeError **error);
uint32_t lodedb_graph_persist(
    LodeGraph *graph, LodeError **error);

#ifdef __cplusplus
}
#endif

#endif
