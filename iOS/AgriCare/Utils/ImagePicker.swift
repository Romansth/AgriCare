import SwiftUI
import UIKit

struct ImagePicker: UIViewControllerRepresentable {
    // Binding to determine if the source is camera or photo library
    @Binding var isCamera: Bool
    
    // Closure to handle the picked image
    var onImagePicked: (UIImage) -> Void

    // Coordinator class to act as delegate for UIImagePickerController
    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        var parent: ImagePicker

        init(parent: ImagePicker) {
            self.parent = parent
        }

        // Delegate method called when an image is picked
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.onImagePicked(uiImage)
            }
            picker.dismiss(animated: true)
        }

        // Delegate method called when the picker is cancelled
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true)
        }
    }

    // Method to create the Coordinator instance
    func makeCoordinator() -> Coordinator {
        return Coordinator(parent: self)
    }

    // Method to create the UIImagePickerController
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = isCamera ? .camera : .photoLibrary
        return picker
    }

    // Method to update the UIImagePickerController (not used in this case)
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
}
