"""Offline collection invariants: selection, originals, and HTTP transfer boundaries."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts.collect_hf_items import RangeReader, image_facts, select_rows, store_image
from scripts.analyze_candidates import inspect


class CollectionTests(unittest.TestCase):
    def test_selects_only_allowlisted_categories_with_reproducible_caps(self):
        rows = [dict(row_idx=i, category="sword" if i < 20 else "car") for i in range(40)]
        selected = select_rows(rows, {"sword": "weapon"}, {"weapon": 7}, 42)
        self.assertEqual(len(selected), 7)
        self.assertTrue(all(r["row_idx"] < 20 for r in selected))
        self.assertEqual(selected, select_rows(list(reversed(rows)), {"sword": "weapon"}, {"weapon": 7}, 42))
        self.assertTrue(all("item_class" not in r for r in rows))

    def test_native_original_and_alpha_hash(self):
        a = Image.new("RGBA", (16, 16), (80, 0, 100, 0))
        a.putpixel((4, 4), (255, 0, 0, 255))
        b = a.copy()
        b.putpixel((0, 0), (0, 0, 0, 0))
        bodies = []
        for image in (a, b):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            bodies.append(buffer.getvalue())
        facts = [image_facts(body, [16, 16]) for body in bodies]
        self.assertEqual(facts[0]["canonical_rgba_sha256"], facts[1]["canonical_rgba_sha256"])
        self.assertNotEqual(facts[0]["sha256"], facts[1]["sha256"])
        with self.assertRaises(ValueError):
            image_facts(bodies[0], [32, 32])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            path.write_bytes(bodies[0])
            result = store_image(dict(row_idx=2, item_class="weapon"), bodies[0], "hf_krea16", "revision",
                                 [16, 16], {2: {"path": path.as_posix()}})
            self.assertTrue(result["reused_sample"])
            self.assertEqual(path.read_bytes(), bodies[0])
            inspected = inspect(path, "hf_krea16", result, allow_synthetic_review=True)
            self.assertEqual(inspected["role"], "synthetic_item_candidate")
            self.assertEqual(inspected["category"], "label:weapon")

    def test_range_reader_rejects_full_file_fallback(self):
        response = io.BytesIO(b"entire file")
        response.status = 200
        response.headers = {}
        with patch("scripts.collect_hf_items.urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "exact byte range"):
                RangeReader("https://example.test/file")

    def test_range_reader_seek_eof_and_budget(self):
        payload = b"0123456789abcdefghij"
        def open_range(request, timeout):
            start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
            response = io.BytesIO(payload[start:end+1])
            response.status = 206
            response.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}
            return response
        with patch("scripts.collect_hf_items.urlopen", side_effect=open_range):
            with RangeReader("https://example.test/file", max_bytes=14) as reader:
                reader.seek(-4, 2)
                self.assertEqual(reader.read(50), b"ghij")
                self.assertEqual(reader.read(1), b"")
                reader.seek(0)
                with self.assertRaisesRegex(ValueError, "budget"):
                    reader.read(4)


if __name__ == "__main__":
    unittest.main()
