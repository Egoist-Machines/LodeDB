import Foundation

public enum LodeDBError: Error, Equatable {
    case invalidArgument(String)
    case notFound(String)
    case corruptStore(String)
}
