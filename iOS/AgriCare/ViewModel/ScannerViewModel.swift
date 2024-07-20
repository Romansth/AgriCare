import SwiftUI
import Combine

class ScannerViewModel: ObservableObject {
    // Published properties to notify views of changes
    @Published var reports: [Report] = []
    @Published var result: Report?

    // Set to hold cancellables for Combine subscriptions
    private var cancellables = Set<AnyCancellable>()

    // Function to process the image by sending it to an endpoint and saving the report
    func processImage(_ image: UIImage) {
        sendImageToEndpoint(image: image) { [weak self] result, plant, disease, remedy in
            guard let self = self else { return }
            DispatchQueue.main.async {
                self.saveReport(image: image, plant: plant, disease: disease, remedy: remedy)
            }
        }
    }

    // Function to send the image to the endpoint
    private func sendImageToEndpoint(image: UIImage, completion: @escaping (String, String, String, String) -> Void) {
        // Convert image to JPEG data
        let resizedImage = resizeImage(image: image, targetSize: CGSize(width: 224, height: 224))
        guard let imageData = resizedImage.jpegData(compressionQuality: 1) else {
            print("Failed to convert image to JPEG data")
            DispatchQueue.main.async {
                completion("Error: Failed to convert image to JPEG data", "", "", "")
            }
            return
        }

        // Encode image data to Base64 string
        let base64String = imageData.base64EncodedString()
        print("Base64 image string length:", base64String.count)

        // Define URL for the endpoint
        let urlString = "API_ENDPOINT"
        guard let url = URL(string: urlString) else {
            print("Invalid URL:", urlString)
            DispatchQueue.main.async {
                completion("Error: Invalid URL", "", "", "")
            }
            return
        }
        print("Sending POST request to:", url.absoluteString)

        // Create URLRequest for the POST request
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        // Create JSON body with the Base64 image string
        let body: [String: String] = ["image": base64String]
        guard let jsonData = try? JSONSerialization.data(withJSONObject: body, options: []) else {
            print("Failed to serialize JSON body")
            DispatchQueue.main.async {
                completion("Error: Failed to serialize JSON body", "", "", "")
            }
            return
        }
        request.httpBody = jsonData

        // Perform the POST request
        URLSession.shared.dataTask(with: request) { (data, response, error) in
            if let error = error {
                print("Failed to send image:", error)
                DispatchQueue.main.async {
                    completion("Error: \(error.localizedDescription)", "", "", "")
                }
                return
            }

            guard let httpResponse = response as? HTTPURLResponse else {
                print("No HTTP response")
                DispatchQueue.main.async {
                    completion("Error: No HTTP response", "", "", "")
                }
                return
            }

//            print("HTTP Response:", httpResponse)

            guard (200...299).contains(httpResponse.statusCode) else {
                print("Server error: \(httpResponse.statusCode)")
                DispatchQueue.main.async {
                    completion("Error: Server error \(httpResponse.statusCode)", "", "", "")
                }
                return
            }

            if let data = data {
                do {
                    // Parse the JSON response
                    if let jsonResponse = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] {
                        if let disease = jsonResponse["disease"] as? String,
                           let plant = jsonResponse["plant"] as? String,
                           let remedy = jsonResponse["remedy"] as? String {
                            let result = """
                            Disease: \(disease)
                            Plant: \(plant)
                            Remedy: \(remedy)
                            """
                            DispatchQueue.main.async {
                                completion(result, plant, disease, remedy)
                            }
                        } else {
                            DispatchQueue.main.async {
                                completion("Incomplete data in JSON response", "", "", "")
                            }
                        }
                    } else {
                        DispatchQueue.main.async {
                            completion("Failed to parse JSON response", "", "", "")
                        }
                    }
                } catch {
                    print("Error parsing JSON response:", error)
                    DispatchQueue.main.async {
                        completion("Error: \(error.localizedDescription)", "", "", "")
                    }
                }
            } else {
                print("No data received")
                DispatchQueue.main.async {
                    completion("Error: No data received", "", "", "")
                }
            }
        }.resume()
    }

    // Function to save the report
    private func saveReport(image: UIImage, plant: String, disease: String, remedy: String) {
        let report = Report(
            imageData: image.jpegData(compressionQuality: 1.0) ?? Data(),
            plant: plant,
            disease: disease,
            remedy: remedy,
            date: Date()
        )
        DispatchQueue.main.async {
            self.reports.append(report)
            self.result = report
        }
    }
}

private func resizeImage(image: UIImage, targetSize: CGSize) -> UIImage {
    let size = image.size
    let widthRatio  = targetSize.width  / size.width
    let heightRatio = targetSize.height / size.height

    // Determine new size
    var newSize: CGSize
    if(widthRatio > heightRatio) {
        newSize = CGSize(width: size.width * heightRatio, height: size.height * heightRatio)
    } else {
        newSize = CGSize(width: size.width * widthRatio, height: size.height * widthRatio)
    }

    // Resize image
    let rect = CGRect(origin: .zero, size: newSize)
    UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
    image.draw(in: rect)
    let newImage = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()

    return newImage!
}
