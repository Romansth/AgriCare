import SwiftUI

struct ReportDetailView: View {
    let report: Report

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            if let uiImage = UIImage(data: report.imageData) {
                Image(uiImage: uiImage)
                    .resizable()
                    .scaledToFit()
                    .frame(height: 300)
                    .cornerRadius(15)
                    .shadow(radius: 10)
            }
            
            Text("Plant: \(report.plant)")
                .font(.title)
                .fontWeight(.bold)
                .foregroundColor(.green)
                .padding(.top, 16)

            Text("Disease: \(report.disease)")
                .font(.headline)
                .foregroundColor(.red)
                .padding(.top, 8)

            Text("Remedy: \(report.remedy)")
                .font(.body)
                .foregroundColor(.primary)
                .padding(.top, 8)

            Text("Date: \(report.date, style: .date)")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .padding(.top, 8)

            Spacer()
        }
        .background(Color.white)
        .cornerRadius(15)
        .padding()
    }
}
