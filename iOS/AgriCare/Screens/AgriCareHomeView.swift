import SwiftUI

struct AgriCareHomeView: View {
    // Binding property to manage selected tab
    @Binding var selectedTab: Int

    var body: some View {
        NavigationView {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    Spacer()
                }
                
                VStack(alignment: .leading, spacing: 30) {
                    // Title text
                    Text("AgriCare: Crop Disease Diagnosis")
                        .font(.title)
                        .fontWeight(.bold)
                        .foregroundColor(.green)
                        .padding(.horizontal)
                    
                    VStack(spacing: 20) {
                        // Button to navigate to Diagnose tab
                        Button(action: {
                            selectedTab = 1
                        }) {
                            CareToolButton(imageName: "shield.lefthalf.fill", title: "Diagnose", description: "Check your plant's health", color: .green)
                        }

                        // Button to navigate to Reports tab
                        Button(action: {
                            selectedTab = 1
                        }) {
                            CareToolButton(imageName: "leaf.fill", title: "Reports", description: "View plant history", color: .orange)
                        }
                    }
                    
                    // Supported crops information
                    VStack(alignment: .leading, spacing: 10) {
                        Text("We currently support these crops: ")
                            .font(.headline)
                            .foregroundColor(.red)
                            .padding(.horizontal)
                        
                        Text("Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Squash, Strawberry, Tomato")
                            .font(.subheadline)
                            .foregroundColor(.green)
                            .padding(.horizontal)
                    }
                }
                .padding(.horizontal)
                
                Spacer()
            }
//            .navigationBarTitle("AgriCare: Crop Disease Diagnosis", displayMode: .inline)
        }
    }
}

struct BrikshaHomeView_Previews: PreviewProvider {
    static var previews: some View {
        AgriCareHomeView(selectedTab: .constant(0))
    }
}
