import Foundation

struct Report: Identifiable, Hashable {
    let id = UUID()
    let imageData: Data
    let plant: String
    let disease: String
    let remedy: String
    let date: Date
}
