import "./globals.css";

export const metadata = {
  title: "Temperature Bench Explorer",
  description: "Explore how language model writing changes as sampling temperature rises.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
