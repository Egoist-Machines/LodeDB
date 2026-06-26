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

  assert(lodedb_engine_create_index(engine, sv("default"), 2, 4, &error) == LODE_OK);

  float vector[2] = {1.0f, 0.0f};
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
  document.vector_len = 2;
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
  request.query_len = 2;
  request.top_k = 1;

  LodeSearchResults *results = 0;
  assert(lodedb_engine_query_vector(engine, &request, &results, &error) == LODE_OK);
  assert(results != 0);
  assert(results->hits_len == 1);
  assert(strcmp(results->hits[0].document_id, "doc-a") == 0);
  lodedb_search_results_free(results);

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
             engine, owned_sv(plan), sv("[[1.0,0.0]]"), 2.5, &applied, &error) == LODE_OK);
  assert(applied != 0);
  assert(strstr(applied->data, "\"embedded_chunks\":1") != 0);
  assert(strstr(applied->data, "\"embedding_time_ms\":2.5") != 0);
  lodedb_owned_string_free(applied);
  lodedb_owned_string_free(plan);

  lodedb_engine_free(engine);
  lodedb_error_free(error);
  return 0;
}
