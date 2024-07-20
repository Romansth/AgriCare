import SwiftUI

struct ReportView: View {
    @EnvironmentObject var viewModel: ScannerViewModel
    @Binding var selectedTab: Int

    var body: some View {
        NavigationView {
            VStack {
                Spacer()
                VStack {
                    // Display message when there are no reports
                    if viewModel.reports.isEmpty {
                        VStack {
                            Image(systemName: "leaf.fill")
                                .resizable()
                                .frame(width: 100, height: 100)
                                .foregroundColor(.green)
                                .padding()

                            Text("You Have No Reports")
                                .font(.title)
                                .fontWeight(.bold)
                                .foregroundColor(.gray)
                                .padding(.top, 16)

                            Text("Scan your first plant and start caring for it.")
                                .font(.body)
                                .foregroundColor(.gray)
                                .padding(.top, 8)

                            Button(action: {
                                selectedTab = 1
                            }) {
                                Text("Scan a plant")
                                    .font(.headline)
                                    .foregroundColor(.white)
                                    .padding()
                                    .background(Color.green)
                                    .cornerRadius(8)
                                    .padding(.top, 16)
                            }
                        }
                        .padding()
                    } else {
                        // Display list of reports when there are reports
                        VStack(alignment: .leading, spacing: 20) {
                            Text("Reports")
                                .font(.title)
                                .fontWeight(.bold)
                                .foregroundColor(.green)
                                .padding(.horizontal)
                            
                            ScrollView {
                                VStack(spacing: 10) {
                                    ForEach(viewModel.reports, id: \.id) { report in
                                        ReportItemView(report: report)
                                    }
                                }
                            }
                        }
                    }
                }
                .padding()

                Spacer()
            }
            .background(Color(hex: "#F0F0F5"))
        }
    }
}
