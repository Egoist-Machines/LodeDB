#ifndef CLODEDB_CORE_BRIDGE_H
#define CLODEDB_CORE_BRIDGE_H

#include <stdint.h>

#define LODEDB_ABI_VERSION 1u

typedef struct LodeError {
  uint32_t size;
  uint32_t version;
  uint32_t code;
  const char *message;
} LodeError;

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

#endif
