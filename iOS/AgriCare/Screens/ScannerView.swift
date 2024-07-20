import SwiftUI

struct ScannerView: View {
    // EnvironmentObject to access the view model
    @EnvironmentObject var viewModel: ScannerViewModel
    
    // State properties to manage various UI states
    @State private var isShowingImagePicker = false
    @State private var isCamera = true
    @State private var isResultPresented = false
    @State private var showMessage = false
    
    // Binding property to manage selected tab
    @Binding var selectedTab: Int

    var body: some View {
        NavigationView {
            
            VStack(alignment: .leading) {
                HStack {
                    Spacer()
                }
                
                
                Text("Diagnosis")
                    .font(.title)
                    .fontWeight(.bold)
                    .foregroundColor(.green)
                    .padding(.top, 10)
                    .padding(.horizontal, 40)
                
                
                // Instruction text
                Text("Scan a plant leaf to get started")
                    .foregroundColor(.gray)
                    .font(.headline)
                    .fontWeight(.bold)
                    .padding(.top, 5)
                    .padding(.horizontal, 40)
                
                VStack(spacing: 20) {
                    // Button to open camera
                    Button(action: {
                        isCamera = true
                        isShowingImagePicker = true
                    }) {
                        CareToolButton(imageName: "camera.circle.fill", title: "Camera", description: "", color: .green)
                    }

                    // Button to open photo gallery
                    Button(action: {
                        isCamera = false
                        isShowingImagePicker = true
                    }) {
                        CareToolButton(imageName: "photo.circle.fill", title: "Gallery", description: "", color: .orange)
                    }
                }
                .padding(.horizontal, 20)

                // Display message when report is saved successfully
                if showMessage {
                    VStack(spacing: 16) {
                        Text("Report saved successfully!")
                            .foregroundColor(.green)
                            .padding(.top)

                        Button(action: {
                            selectedTab = 2
                        }) {
                            HStack {
                                Text("View Reports")
                                    .fontWeight(.semibold)
                                    .foregroundColor(.white)
                                Image(systemName: "arrow.forward.circle.fill")
                                    .foregroundColor(.white)
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color(hex: "#27AE60"))
                            .cornerRadius(15)
                        }
                    }
                    .padding(.horizontal, 20)
                }

                Spacer()
            }
            .sheet(isPresented: $isShowingImagePicker) {
                ImagePicker(isCamera: $isCamera, onImagePicked: { image in
                    viewModel.processImage(image)
                    isResultPresented = true
                })
            }
            .sheet(isPresented: $isResultPresented, onDismiss: {
                showMessage = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 10) {
                    showMessage = false
                }
            }) {
                if let report = viewModel.result {
                    ReportDetailView(report: report)
                }
            }
        }
    }
}

struct ScannerView_Previews: PreviewProvider {
    static var previews: some View {
        ScannerView(selectedTab: .constant(1))
            .environmentObject(ScannerViewModel())
    }
}

// Extension to initialize Color from hex string
extension Color {
    init(hex: String) {
        let scanner = Scanner(string: hex)
        scanner.scanLocation = 1
        var rgbValue: UInt64 = 0
        scanner.scanHexInt64(&rgbValue)
        let red = Double((rgbValue & 0xff0000) >> 16) / 0xff
        let green = Double((rgbValue & 0x00ff00) >> 8) / 0xff
        let blue = Double(rgbValue & 0x0000ff) / 0xff
        self.init(red: red, green: green, blue: blue)
    }
}
