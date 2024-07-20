import SwiftUI

struct CareToolButton: View {
    let imageName: String
    let title: String
    let description: String
    let color: Color

    var body: some View {
        HStack {
            Image(systemName: imageName)
                .foregroundColor(color)
                .imageScale(.large)
                .font(.system(size: 40))
                .padding()

            VStack(alignment: .leading) {
                Text(title)
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(.primary)
                if !description.isEmpty {
                    Text(description)
                        .font(.title3)
                        .foregroundColor(.secondary)
                }
            }
            .padding(.leading)

            Spacer()
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 15).fill(color.opacity(0.1)))
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
