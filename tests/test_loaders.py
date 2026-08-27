import tempfile
import unittest
from pathlib import Path

from policy_rag.loaders import load_document


class LoaderTests(unittest.TestCase):
    def test_html_loader_extracts_policy_body(self) -> None:
        html = """
        <html>
        <body>
            <nav>Skip to navigation</nav>
            <h1>IMP 99: Test Policy</h1>
            <p>Information Classification - Public</p>
            <p>
                6.1 Use
                <a href="#">prior approval</a>
                from IDG.
            </p>
            <p>Page contact: Test Owner</p>
            <div>Cookie preferences</div>
        </body>
        </html>
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test_policy.html"
            path.write_text(html, encoding="utf-8")

            document = load_document(path)

        self.assertTrue(document.text.startswith("IMP 99: Test Policy"))
        normalised_text = " ".join(document.text.split())
        self.assertIn("prior approval from IDG", normalised_text)
        self.assertNotIn("Skip to navigation", document.text)
        self.assertNotIn("Cookie preferences", document.text)
        self.assertNotIn("Page contact", document.text)


if __name__ == "__main__":
    unittest.main()