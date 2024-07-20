import SwiftUI

struct AgriCareTabView: View {
    // State property to track the selected tab
    @State private var selectedTab: Int = 0
    
    // StateObject to manage the ScannerViewModel
    @StateObject var viewModel = ScannerViewModel()
    
    var body: some View {
        TabView(selection: $selectedTab) {
            // Home tab
            AgriCareHomeView(selectedTab: $selectedTab)
                .tabItem {
                    Image(systemName: "house")
                    Text("Home")
                }
                .tag(0)
            
            // Scan tab
            ScannerView(selectedTab: $selectedTab)
                .tabItem {
                    Image(systemName: "camera")
                    Text("Scan")
                }
                .tag(1)
            
            // Reports tab
            ReportView(selectedTab: $selectedTab)
                .tabItem {
                    Image(systemName: "doc")
                    Text("Reports")
                }
                .tag(2)
        }
        .environmentObject(viewModel) // Pass the view model to the environment
        .accentColor(Color("brandPrimary")) // Set the accent color
    }
}

#Preview {
    AgriCareTabView()
}
