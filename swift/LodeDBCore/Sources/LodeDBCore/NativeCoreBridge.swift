import Foundation
import LodeDBCoreFFI

/// The C ABI version this binding is built against. Checked at engine creation so a
/// mismatched XCFramework fails loudly instead of corrupting memory.
let lodeNativeExpectedABIVersion: UInt32 = 1

/// Owning wrapper around a native `LodeEngine *`, statically linked from the
/// `LodeDBCoreFFI` XCFramework (no `dlopen`). Not thread-safe; callers serialize
/// access (see `LodeDB`).
final class NativeEngine {
    private let handle: OpaquePointer
    /// Mutable so a read-only open can rebind to the index id discovered on disk.
    var indexID: String

    static func abiVersion() -> UInt32 { lodedb_abi_version() }

    private init(handle: OpaquePointer, indexID: String) {
        self.handle = handle
        self.indexID = indexID
    }

    deinit {
        lodedb_engine_free(handle)
    }

    // MARK: - Construction

    /// Creates an empty in-memory engine and a vector index on it (optionally bound
    /// to a model identity).
    static func inMemory(
        vectorDimension: Int,
        bitWidth: Int = 4,
        indexID: String = "default",
        model: String? = nil,
        ann: LodeAnnOptions? = nil
    ) throws -> NativeEngine {
        try requireABI()
        var engine: OpaquePointer?
        var error: UnsafeMutablePointer<LodeError>?
        try check(lodedb_engine_new_in_memory(&engine, &error), error: error)
        guard let engine else {
            throw LodeDBError.internalError("native core did not return an engine")
        }
        let native = NativeEngine(handle: engine, indexID: indexID)
        try native.createIndex(vectorDimension: vectorDimension, bitWidth: bitWidth, model: model, ann: ann)
        return native
    }

    /// Opens a writable persistent engine from a `CoreOpenOptions` JSON document.
    static func open(optionsJSON: String, indexID: String = "default") throws -> NativeEngine {
        try requireABI()
        var engine: OpaquePointer?
        var error: UnsafeMutablePointer<LodeError>?
        let status = withStringView(optionsJSON) { lodedb_engine_open_json($0, &engine, &error) }
        try check(status, error: error)
        guard let engine else {
            throw LodeDBError.internalError("native core did not return an engine")
        }
        return NativeEngine(handle: engine, indexID: indexID)
    }

    /// Opens a lock-free read-only generation snapshot from a `CoreOpenOptions` JSON.
    static func openReadOnly(optionsJSON: String, indexID: String = "default") throws -> NativeEngine {
        try requireABI()
        var engine: OpaquePointer?
        var error: UnsafeMutablePointer<LodeError>?
        let status = withStringView(optionsJSON) { lodedb_engine_open_readonly_json($0, &engine, &error) }
        try check(status, error: error)
        guard let engine else {
            throw LodeDBError.internalError("native core did not return an engine")
        }
        return NativeEngine(handle: engine, indexID: indexID)
    }

    /// Creates the index. When `model` is non-nil the index is bound to that model
    /// identity (for the reopen-time embedder guard); otherwise native defaults apply.
    /// When `ann` is set the index opts into approximate-nearest-neighbor scanning.
    func createIndex(
        vectorDimension: Int,
        bitWidth: Int = 4,
        model: String? = nil,
        ann: LodeAnnOptions? = nil
    ) throws {
        // ANN needs the full CoreIndexCreateOptions, which the positional create
        // functions do not carry, so route it through the JSON create ABI. The
        // non-ann fields mirror CoreIndexCreateOptions::native_default so an ANN
        // index shares the exact identity and route of a positionally-created one.
        if let ann {
            let options = CoreIndexCreateOptionsJSON(
                index_id: indexID,
                index_key: indexID,
                client_id_hash: indexID,
                name: "lodedb-local",
                model: model ?? "native-core",
                provider: "native",
                task: "native-core",
                route_profile: "native-core",
                storage_profile: "native-core",
                vector_dim: vectorDimension,
                bit_width: bitWidth,
                ann: CoreAnnOptionsJSON(
                    algorithm: ann.algorithm, clusters: ann.clusters, nprobe: ann.nprobe))
            let optionsJSON = try encodeJSON(options)
            var error: UnsafeMutablePointer<LodeError>?
            let status = withStringView(optionsJSON) {
                lodedb_engine_create_index_json(handle, $0, &error)
            }
            try Self.check(status, error: error)
            return
        }
        var error: UnsafeMutablePointer<LodeError>?
        let status: UInt32
        if let model {
            status = withStringView(indexID) { indexView in
                withStringView(model) { modelView in
                    lodedb_engine_create_index_with_model(
                        handle, indexView, UInt(vectorDimension), UInt(bitWidth), modelView, &error)
                }
            }
        } else {
            status = withStringView(indexID) { indexView in
                lodedb_engine_create_index(handle, indexView, UInt(vectorDimension), UInt(bitWidth), &error)
            }
        }
        try Self.check(status, error: error)
    }

    // MARK: - Ingest

    func upsertVectorsJSON(_ documentsJSON: String) throws {
        var error: UnsafeMutablePointer<LodeError>?
        let status = withStringView(indexID) { indexView in
            withStringView(documentsJSON) { documentsView in
                lodedb_engine_upsert_vectors_json(handle, indexView, documentsView, &error)
            }
        }
        try Self.check(status, error: error)
    }

    func prepareTextUpsertJSON(
        _ documentsJSON: String,
        storeText: Bool,
        indexText: Bool,
        chunkCharacterLimit: Int
    ) throws -> String {
        try ownedCall { out, error in
            withStringView(indexID) { indexView in
                withStringView(documentsJSON) { documentsView in
                    lodedb_engine_prepare_text_upsert_json(
                        handle,
                        indexView,
                        documentsView,
                        storeText ? 1 : 0,
                        indexText ? 1 : 0,
                        UInt(chunkCharacterLimit),
                        out,
                        error
                    )
                }
            }
        }
    }

    func applyTextUpsertJSON(planJSON: String, embeddingsJSON: String, embeddingTimeMS: Double) throws -> String {
        try ownedCall { out, error in
            withStringView(planJSON) { planView in
                withStringView(embeddingsJSON) { embeddingsView in
                    lodedb_engine_apply_text_upsert_json(handle, planView, embeddingsView, embeddingTimeMS, out, error)
                }
            }
        }
    }

    // MARK: - Query

    func queryVectorJSON(_ vector: [Float], k: Int, filterJSON: String?) throws -> String {
        try ownedCall { out, error in
            withStringView(indexID) { indexView in
                vector.withUnsafeBufferPointer { queryBuffer in
                    withStringView(filterJSON ?? "") { filterView in
                        lodedb_engine_query_vector_json(
                            handle,
                            indexView,
                            queryBuffer.baseAddress,
                            UInt(vector.count),
                            UInt(k),
                            filterView,
                            filterJSON == nil ? 0 : 1,
                            out,
                            error
                        )
                    }
                }
            }
        }
    }

    func prepareQueryTextJSON(_ query: String, mode: String) throws -> String {
        try ownedCall { out, error in
            withStringView(query) { queryView in
                withStringView(mode) { modeView in
                    lodedb_engine_prepare_query_text_json(handle, queryView, modeView, out, error)
                }
            }
        }
    }

    func searchEmbeddedTextJSON(
        queryPlanJSON: String,
        queryEmbeddingJSON: String?,
        k: Int,
        filterJSON: String?
    ) throws -> String {
        try ownedCall { out, error in
            withStringView(indexID) { indexView in
                withStringView(queryPlanJSON) { queryPlanView in
                    withStringView(queryEmbeddingJSON ?? "") { embeddingView in
                        withStringView(filterJSON ?? "") { filterView in
                            lodedb_engine_search_embedded_text_json(
                                handle,
                                indexView,
                                queryPlanView,
                                embeddingView,
                                queryEmbeddingJSON == nil ? 0 : 1,
                                UInt(k),
                                filterView,
                                filterJSON == nil ? 0 : 1,
                                out,
                                error
                            )
                        }
                    }
                }
            }
        }
    }

    func queryVectorsBatchJSON(queriesJSON: String, k: Int, filterJSON: String?) throws -> String {
        try ownedCall { out, error in
            withStringView(indexID) { indexView in
                withStringView(queriesJSON) { queriesView in
                    withStringView(filterJSON ?? "") { filterView in
                        lodedb_engine_query_vectors_batch_json(
                            handle, indexView, queriesView, UInt(k), filterView,
                            filterJSON == nil ? 0 : 1, out, error)
                    }
                }
            }
        }
    }

    func searchEmbeddedTextBatchJSON(
        queryPlansJSON: String,
        queryEmbeddingsJSON: String?,
        k: Int,
        filterJSON: String?
    ) throws -> String {
        try ownedCall { out, error in
            withStringView(indexID) { indexView in
                withStringView(queryPlansJSON) { plansView in
                    withStringView(queryEmbeddingsJSON ?? "") { embeddingsView in
                        withStringView(filterJSON ?? "") { filterView in
                            lodedb_engine_search_embedded_text_batch_json(
                                handle, indexView, plansView, embeddingsView,
                                queryEmbeddingsJSON == nil ? 0 : 1, UInt(k), filterView,
                                filterJSON == nil ? 0 : 1, out, error)
                        }
                    }
                }
            }
        }
    }

    func queryMultivectorJSON(query: [Float], nQuery: Int, k: Int, filterJSON: String?) throws -> String {
        try ownedCall { out, error in
            withStringView(indexID) { indexView in
                query.withUnsafeBufferPointer { queryBuffer in
                    withStringView(filterJSON ?? "") { filterView in
                        lodedb_engine_query_multivector_json(
                            handle, indexView, queryBuffer.baseAddress, UInt(query.count),
                            UInt(nQuery), UInt(k), filterView, filterJSON == nil ? 0 : 1, out, error)
                    }
                }
            }
        }
    }

    func upsertMultivectorJSON(
        vectors: [Float],
        rows: Int,
        dim: Int,
        patchBytes: [UInt8],
        sidecarJSON: String
    ) throws -> String {
        try ownedCall { out, error in
            withStringView(indexID) { indexView in
                vectors.withUnsafeBufferPointer { vectorBuffer in
                    patchBytes.withUnsafeBufferPointer { patchBuffer in
                        withStringView(sidecarJSON) { sidecarView in
                            lodedb_engine_upsert_multivector_json(
                                handle, indexView, vectorBuffer.baseAddress, UInt(rows), UInt(dim),
                                patchBuffer.baseAddress, UInt(patchBytes.count), sidecarView, out, error)
                        }
                    }
                }
            }
        }
    }

    // MARK: - CRUD / introspection

    func deleteDocumentsJSON(_ idsJSON: String) throws -> String {
        try ownedCall { out, error in
            withStringView(indexID) { indexView in
                withStringView(idsJSON) { idsView in
                    lodedb_engine_delete_documents_json(handle, indexView, idsView, out, error)
                }
            }
        }
    }

    /// `metadataJSON == nil` leaves metadata unchanged; `textJSON == nil` leaves text
    /// unchanged, otherwise `textJSON` is an `Option<String>` JSON (`null` clears it).
    func updateDocumentPayloadJSON(documentID: String, metadataJSON: String?, textJSON: String?) throws -> String {
        try ownedCall { out, error in
            withStringView(indexID) { indexView in
                withStringView(documentID) { idView in
                    withStringView(metadataJSON ?? "") { metadataView in
                        withStringView(textJSON ?? "") { textView in
                            lodedb_engine_update_document_payload_json(
                                handle,
                                indexView,
                                idView,
                                metadataView,
                                metadataJSON == nil ? 0 : 1,
                                textView,
                                textJSON == nil ? 0 : 1,
                                out,
                                error
                            )
                        }
                    }
                }
            }
        }
    }

    func statsJSON() throws -> String {
        try ownedCall { out, error in
            withStringView(indexID) { indexView in
                lodedb_engine_stats_json(handle, indexView, out, error)
            }
        }
    }

    func getDocumentJSON(documentID: String) throws -> String {
        try ownedCall { out, error in
            withStringView(indexID) { indexView in
                withStringView(documentID) { idView in
                    lodedb_engine_get_document_json(handle, indexView, idView, out, error)
                }
            }
        }
    }

    func getDocumentTextJSON(documentID: String) throws -> String {
        try ownedCall { out, error in
            withStringView(indexID) { indexView in
                withStringView(documentID) { idView in
                    lodedb_engine_get_document_text_json(handle, indexView, idView, out, error)
                }
            }
        }
    }

    func getDocumentTextsJSON(_ idsJSON: String) throws -> String {
        try ownedCall { out, error in
            withStringView(indexID) { indexView in
                withStringView(idsJSON) { idsView in
                    lodedb_engine_get_document_texts_json(handle, indexView, idsView, out, error)
                }
            }
        }
    }

    func listDocumentsJSON(filterJSON: String?, after: String?, limit: Int?) throws -> String {
        try ownedCall { out, error in
            withStringView(indexID) { indexView in
                withStringView(filterJSON ?? "") { filterView in
                    withStringView(after ?? "") { afterView in
                        lodedb_engine_list_documents_json(
                            handle,
                            indexView,
                            filterView,
                            filterJSON == nil ? 0 : 1,
                            afterView,
                            after == nil ? 0 : 1,
                            UInt(limit ?? 0),
                            limit == nil ? 0 : 1,
                            out,
                            error
                        )
                    }
                }
            }
        }
    }

    func indexIdsJSON() throws -> String {
        try ownedCall { out, error in
            lodedb_engine_index_ids_json(handle, out, error)
        }
    }

    func persist() throws {
        var error: UnsafeMutablePointer<LodeError>?
        try Self.check(lodedb_engine_persist(handle, &error), error: error)
    }

    func close() throws {
        var error: UnsafeMutablePointer<LodeError>?
        try Self.check(lodedb_engine_close(handle, &error), error: error)
    }

    // MARK: - Helpers

    fileprivate static func requireABI() throws {
        let abi = lodedb_abi_version()
        guard abi == lodeNativeExpectedABIVersion else {
            throw LodeDBError.corruptStore(
                "native core ABI \(abi) does not match expected \(lodeNativeExpectedABIVersion)")
        }
    }

    /// Runs an FFI call that returns an owned JSON string out-parameter, checks the
    /// status, and copies the result into a Swift `String` (freeing the native one).
    private func ownedCall(
        _ body: (UnsafeMutablePointer<UnsafeMutablePointer<LodeOwnedString>?>,
                 UnsafeMutablePointer<UnsafeMutablePointer<LodeError>?>) -> UInt32
    ) throws -> String {
        var out: UnsafeMutablePointer<LodeOwnedString>?
        var error: UnsafeMutablePointer<LodeError>?
        let status = body(&out, &error)
        try Self.check(status, error: error)
        return try Self.copyOwnedString(out)
    }

    fileprivate static func check(_ status: UInt32, error: UnsafeMutablePointer<LodeError>?) throws {
        guard status != 0 else { return }
        defer { lodedb_error_free(error) }
        let message = error?.pointee.message.map { String(cString: $0) } ?? "native core call failed"
        switch status {
        case 1: throw LodeDBError.invalidArgument(message)
        case 2: throw LodeDBError.notFound(message)
        case 3: throw LodeDBError.corruptStore(message)
        case 4: throw LodeDBError.planStale(message)
        case 5: throw LodeDBError.unsupported(message)
        default: throw LodeDBError.internalError(message)
        }
    }

    private static func copyOwnedString(_ out: UnsafeMutablePointer<LodeOwnedString>?) throws -> String {
        guard let out else {
            throw LodeDBError.internalError("native core did not return JSON")
        }
        defer { lodedb_owned_string_free(out) }
        let owned = out.pointee
        guard let data = owned.data else {
            if owned.len == 0 {
                return ""
            }
            throw LodeDBError.internalError("native core returned null string data")
        }
        let bytes = Data(bytes: data, count: Int(owned.len))
        guard let text = String(data: bytes, encoding: .utf8) else {
            throw LodeDBError.internalError("native core returned invalid UTF-8")
        }
        return text
    }
}

/// Owning wrapper around a native `LodeAppender *`, statically linked from the
/// `LodeDBCoreFFI` XCFramework (no `dlopen`). Not thread-safe; callers serialize
/// access (see `LodeAppender`).
final class NativeAppender {
    private let handle: OpaquePointer

    private init(handle: OpaquePointer) {
        self.handle = handle
    }

    deinit {
        lodedb_appender_free(handle)
    }

    /// Opens a shared-lock appender over the single index at the store described by
    /// a `CoreOpenOptions` JSON document (WAL commit mode).
    static func open(optionsJSON: String) throws -> NativeAppender {
        try NativeEngine.requireABI()
        var appender: OpaquePointer?
        var error: UnsafeMutablePointer<LodeError>?
        let status = withStringView(optionsJSON) { lodedb_appender_open_json($0, &appender, &error) }
        try NativeEngine.check(status, error: error)
        guard let appender else {
            throw LodeDBError.internalError("native core did not return an appender")
        }
        return NativeAppender(handle: appender)
    }

    /// Durably logs one `upsert_vectors` record and returns its assigned LSN.
    func appendVectorsJSON(_ documentsJSON: String) throws -> UInt64 {
        var lsn: UInt64 = 0
        var error: UnsafeMutablePointer<LodeError>?
        let status = withStringView(documentsJSON) {
            lodedb_appender_append_vectors_json(handle, $0, &lsn, &error)
        }
        try NativeEngine.check(status, error: error)
        return lsn
    }

    /// Durably logs one `delete_documents` record and returns its assigned LSN.
    func appendDeletesJSON(_ documentIDsJSON: String) throws -> UInt64 {
        var lsn: UInt64 = 0
        var error: UnsafeMutablePointer<LodeError>?
        let status = withStringView(documentIDsJSON) {
            lodedb_appender_append_deletes_json(handle, $0, &lsn, &error)
        }
        try NativeEngine.check(status, error: error)
        return lsn
    }
}

/// Serde-shaped payload for `lodedb_engine_create_index_json`. Field names match
/// `CoreIndexCreateOptions`; `nil` optionals (including `ann`) are omitted by the
/// encoder, so an exact index sends no `ann` key.
private struct CoreIndexCreateOptionsJSON: Encodable {
    let index_id: String
    let index_key: String
    let client_id_hash: String
    let name: String
    let model: String
    let provider: String
    let task: String
    let route_profile: String
    let storage_profile: String
    let vector_dim: Int
    let bit_width: Int
    let ann: CoreAnnOptionsJSON?
}

/// Serde-shaped payload for `CoreAnnOptions`; unset tuning is omitted so the core
/// applies its corpus-derived defaults.
private struct CoreAnnOptionsJSON: Encodable {
    let algorithm: String
    let clusters: Int?
    let nprobe: Int?
}

func withStringView<T>(_ string: String, _ body: (LodeStringView) throws -> T) rethrows -> T {
    try string.withCString { pointer in
        let view = LodeStringView(
            size: UInt32(MemoryLayout<LodeStringView>.size),
            version: lodeNativeExpectedABIVersion,
            data: pointer,
            len: UInt(string.utf8.count)
        )
        return try body(view)
    }
}
