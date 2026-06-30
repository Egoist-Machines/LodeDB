#include "lodedb_core.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static LodeStringView sv(const char *text) {
  LodeStringView view;
  view.size = sizeof(LodeStringView);
  view.version = LODEDB_ABI_VERSION;
  view.data = text;
  view.len = (uintptr_t)strlen(text);
  return view;
}

static LodeStringView owned_sv(const LodeOwnedString *text) {
  LodeStringView view;
  view.size = sizeof(LodeStringView);
  view.version = LODEDB_ABI_VERSION;
  view.data = text->data;
  view.len = text->len;
  return view;
}

int main(void) {
  assert(lodedb_abi_version() == LODEDB_ABI_VERSION);
  assert(offsetof(LodeSearchRequest, size) == 0);
  assert(offsetof(LodeSearchRequest, version) == 4);

  LodeError *error = 0;
  LodeEngine *engine = 0;
  assert(lodedb_engine_new_in_memory(&engine, &error) == LODE_OK);
  assert(engine != 0);

  // TurboVec requires a positive-multiple-of-8 dimension; use 8 here.
  assert(lodedb_engine_create_index(engine, sv("default"), 8, 4, &error) == LODE_OK);

  float vector[8] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  LodeMetadataPair metadata[1];
  metadata[0].size = sizeof(LodeMetadataPair);
  metadata[0].version = LODEDB_ABI_VERSION;
  metadata[0].key = sv("topic");
  metadata[0].value = sv("ops");

  LodeVectorDocument document;
  document.size = sizeof(LodeVectorDocument);
  document.version = LODEDB_ABI_VERSION;
  document.document_id = sv("doc-a");
  document.vector = vector;
  document.vector_len = 8;
  document.metadata = metadata;
  document.metadata_len = 1;
  document.text = sv("");
  document.has_text = 0;
  assert(lodedb_engine_upsert_vectors(engine, sv("default"), &document, 1, &error) == LODE_OK);

  LodeSearchRequest request;
  request.size = sizeof(LodeSearchRequest);
  request.version = LODEDB_ABI_VERSION;
  request.index_id = sv("default");
  request.query = vector;
  request.query_len = 8;
  request.top_k = 1;

  LodeSearchResults *results = 0;
  assert(lodedb_engine_query_vector(engine, &request, &results, &error) == LODE_OK);
  assert(results != 0);
  assert(results->hits_len == 1);
  assert(strcmp(results->hits[0].document_id, "doc-a") == 0);
  lodedb_search_results_free(results);

  // JSON vector query carries per-hit metadata and honors the metadata filter.
  LodeOwnedString *vector_json = 0;
  assert(lodedb_engine_query_vector_json(
             engine, sv("default"), vector, 8, 1, sv("{\"metadata\":{\"topic\":\"ops\"}}"), 1,
             &vector_json, &error) == LODE_OK);
  assert(vector_json != 0);
  assert(strstr(vector_json->data, "\"document_id\":\"doc-a\"") != 0);
  assert(strstr(vector_json->data, "\"topic\":\"ops\"") != 0);
  lodedb_owned_string_free(vector_json);

  // JSON vector upsert ingests the same shape the Swift binding emits.
  const char *vectors_json =
      "[{\"document_id\":\"doc-b\",\"vector\":[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],"
      "\"metadata\":{\"topic\":\"ml\"},\"text\":null}]";
  assert(lodedb_engine_upsert_vectors_json(engine, sv("default"), sv(vectors_json), &error) ==
         LODE_OK);
  float query_b[8] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  LodeOwnedString *vector_b_json = 0;
  assert(lodedb_engine_query_vector_json(engine, sv("default"), query_b, 8, 1, sv(""), 0,
                                         &vector_b_json, &error) == LODE_OK);
  assert(vector_b_json != 0);
  assert(strstr(vector_b_json->data, "\"document_id\":\"doc-b\"") != 0);
  lodedb_owned_string_free(vector_b_json);

  const char *documents_json =
      "[{\"document_id\":\"doc-text\",\"text\":\"Alpha launch notes mention error code E-1001.\","
      "\"metadata\":{\"topic\":\"ops\"}}]";
  LodeOwnedString *plan = 0;
  assert(lodedb_engine_prepare_text_upsert_json(
             engine, sv("default"), sv(documents_json), 1, 1, 900, &plan, &error) == LODE_OK);
  assert(plan != 0);
  assert(plan->len > 0);
  assert(strstr(plan->data, "doc-text:d9041255442c:0000") != 0);
  assert(strstr(plan->data, "\"chunks_to_embed\"") != 0);

  LodeOwnedString *applied = 0;
  assert(lodedb_engine_apply_text_upsert_json(
             engine, owned_sv(plan), sv("[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]"), 2.5, &applied,
             &error) == LODE_OK);
  assert(applied != 0);
  assert(strstr(applied->data, "\"embedded_chunks\":1") != 0);
  assert(strstr(applied->data, "\"embedding_time_ms\":2.5") != 0);
  lodedb_owned_string_free(applied);

  LodeOwnedString *query_plan = 0;
  assert(lodedb_engine_prepare_query_text_json(
             engine, sv("E-1001"), sv("lexical"), &query_plan, &error) == LODE_OK);
  assert(query_plan != 0);
  assert(strstr(query_plan->data, "\"requires_embedding\":false") != 0);

  LodeOwnedString *search = 0;
  assert(lodedb_engine_search_embedded_text_json(
             engine, sv("default"), owned_sv(query_plan), sv(""), 0, 1,
             sv("{\"metadata\":{\"topic\":\"ops\"}}"), 1, &search, &error) == LODE_OK);
  assert(search != 0);
  assert(strstr(search->data, "\"document_id\":\"doc-text\"") != 0);
  lodedb_owned_string_free(search);
  lodedb_owned_string_free(query_plan);

  LodeOwnedString *hybrid_plan = 0;
  assert(lodedb_engine_prepare_query_text_json(
             engine, sv("E-1001"), sv("hybrid"), &hybrid_plan, &error) == LODE_OK);
  assert(hybrid_plan != 0);
  assert(strstr(hybrid_plan->data, "\"requires_embedding\":true") != 0);

  LodeOwnedString *hybrid_search = 0;
  assert(lodedb_engine_search_embedded_text_json(
             engine, sv("default"), owned_sv(hybrid_plan), sv("[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]"),
             1, 2,
             sv("{\"metadata\":{\"topic\":\"ops\"}}"), 1, &hybrid_search, &error) == LODE_OK);
  assert(hybrid_search != 0);
  assert(strstr(hybrid_search->data, "\"document_id\":\"doc-text\"") != 0);
  lodedb_owned_string_free(hybrid_search);
  lodedb_owned_string_free(hybrid_plan);
  lodedb_owned_string_free(plan);

  // ---- Durable storage + CRUD (Phase 1) ----
  // stats: metrics-only, reflects the documents added above.
  LodeOwnedString *stats = 0;
  assert(lodedb_engine_stats_json(engine, sv("default"), &stats, &error) == LODE_OK);
  assert(stats != 0 && strstr(stats->data, "\"document_count\"") != 0);
  lodedb_owned_string_free(stats);

  // get_document: payload-free record for doc-a (added as a vector earlier).
  LodeOwnedString *doc = 0;
  assert(lodedb_engine_get_document_json(engine, sv("default"), sv("doc-a"), &doc, &error) == LODE_OK);
  assert(doc != 0 && strstr(doc->data, "\"document_id\":\"doc-a\"") != 0);
  assert(strstr(doc->data, "\"chunk_count\"") != 0);
  lodedb_owned_string_free(doc);

  // get_document_text / get_document_texts: ABI + valid JSON response.
  LodeOwnedString *doc_text = 0;
  assert(lodedb_engine_get_document_text_json(engine, sv("default"), sv("doc-text"), &doc_text,
                                              &error) == LODE_OK);
  assert(doc_text != 0);
  lodedb_owned_string_free(doc_text);
  LodeOwnedString *doc_texts = 0;
  assert(lodedb_engine_get_document_texts_json(engine, sv("default"), sv("[\"doc-text\"]"),
                                               &doc_texts, &error) == LODE_OK);
  assert(doc_texts != 0);
  lodedb_owned_string_free(doc_texts);

  // list_documents: unfiltered, no cursor, no limit.
  LodeOwnedString *list = 0;
  assert(lodedb_engine_list_documents_json(engine, sv("default"), sv(""), 0, sv(""), 0, 0, 0, &list,
                                           &error) == LODE_OK);
  assert(list != 0 && strstr(list->data, "\"document_id\":\"doc-a\"") != 0);
  lodedb_owned_string_free(list);

  // update_document_payload: replace doc-a metadata, leave text untouched.
  LodeOwnedString *updated = 0;
  assert(lodedb_engine_update_document_payload_json(engine, sv("default"), sv("doc-a"),
                                                    sv("{\"topic\":\"updated\"}"), 1, sv(""), 0,
                                                    &updated, &error) == LODE_OK);
  assert(updated != 0);
  lodedb_owned_string_free(updated);
  LodeOwnedString *doc_after = 0;
  assert(lodedb_engine_get_document_json(engine, sv("default"), sv("doc-a"), &doc_after, &error) ==
         LODE_OK);
  assert(strstr(doc_after->data, "\"topic\":\"updated\"") != 0);
  lodedb_owned_string_free(doc_after);

  // delete_documents: remove doc-b, then confirm it is gone (get returns JSON null).
  LodeOwnedString *deleted = 0;
  assert(lodedb_engine_delete_documents_json(engine, sv("default"), sv("[\"doc-b\"]"), &deleted,
                                             &error) == LODE_OK);
  assert(deleted != 0 && strstr(deleted->data, "\"documents_deleted\":1") != 0);
  lodedb_owned_string_free(deleted);
  LodeOwnedString *gone = 0;
  assert(lodedb_engine_get_document_json(engine, sv("default"), sv("doc-b"), &gone, &error) ==
         LODE_OK);
  assert(gone != 0 && strcmp(gone->data, "null") == 0);
  lodedb_owned_string_free(gone);

  // persist/close on an in-memory engine are no-ops that still return OK.
  assert(lodedb_engine_persist(engine, &error) == LODE_OK);
  assert(lodedb_engine_close(engine, &error) == LODE_OK);

  lodedb_engine_free(engine);
  lodedb_error_free(error);
  return 0;
}
