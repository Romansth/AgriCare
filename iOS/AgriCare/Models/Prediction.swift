import Foundation

struct Prediction: Decodable {
    let plant: String
    let disease: String
    let remedy: String
}

struct PredictionResponse {
    let request : [Prediction]
}
