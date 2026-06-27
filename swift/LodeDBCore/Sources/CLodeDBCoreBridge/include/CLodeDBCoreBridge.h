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

#endif
