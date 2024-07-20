import SwiftUI

struct ReportItemView: View {
    let report: Report

    var body: some View {
        NavigationLink(destination: ReportDetailView(report: report)) {
            HStack {
                VStack(alignment: .leading) {
                    Text(report.plant)
                        .font(.headline)
                        .foregroundColor(.primary)
                    Text(report.disease)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Text("\(report.date, style: .date)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Spacer()
            }
            .padding()
            .background(RoundedRectangle(cornerRadius: 15).fill(backgroundColor(for: report.disease)))
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    func backgroundColor(for disease: String) -> Color {
        switch disease.lowercased() {
        case "healthy":
            return Color.green.opacity(0.1)
        default:
            return Color.red.opacity(0.1)
        }
    }
}
