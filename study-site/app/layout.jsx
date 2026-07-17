import "./styles.css";

export const metadata = {
  title: "Temperature Study Results",
  description: "Results from a 576-generation benchmark of sampling temperature across 16 local language models.",
};

export default function Layout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
