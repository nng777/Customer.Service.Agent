+191
-0

import contextlib
import io
import sys
import types
import unittest
from types import SimpleNamespace

try:
    import requests
except ModuleNotFoundError:
    stub = types.SimpleNamespace()

    def _stub_post(*args, **kwargs):
        raise RuntimeError("requests.post stub called unexpectedly")

    stub.post = _stub_post
    sys.modules["requests"] = stub

import app
from app import (
    AgenticPurchaseWorkflow,
    CartItem,
    CheckoutPlan,
    FAQ,
    KnowledgeBase,
    Product,
    RAGRetriever,
)


class ProductTests(unittest.TestCase):
    def test_context_includes_optional_fields(self) -> None:
        product = Product(
            name="Smartphone X",
            description="Layar 6 inci dengan baterai tahan lama.",
            id="SKU-001",
            brand="Acme",
            category="Smartphone",
            variant="128GB",
            price_idr="1.500.000",
            currency="IDR",
            marketplace="Tokopedia",
            store_name="Acme Official Store",
            store_type="Official",
            store_url="https://tokopedia.com/acme",
            product_url="https://tokopedia.com/acme/smartphone-x",
            stock_status="Tersedia",
            last_updated="2024-05-01",
        )

        context = product.context

        self.assertIn("Nama Produk: Smartphone X", context)
        self.assertIn("ID: SKU-001", context)
        self.assertIn("Brand: Acme", context)
        self.assertIn("Harga: IDR 1.500.000", context)
        self.assertIn("URL Produk: https://tokopedia.com/acme/smartphone-x", context)


class KnowledgeBaseTests(unittest.TestCase):
    def test_tokenize_removes_non_alphanumeric(self) -> None:
        tokens = KnowledgeBase._tokenize("Promo terbaru!!! untuk Laptop-123?")
        self.assertEqual(tokens, {"promo", "terbaru", "untuk", "laptop", "123"})

    def test_find_best_products_prioritises_relevant_match(self) -> None:
        laptop = Product(
            name="Laptop Pro",
            description="Laptop profesional untuk kerja harian.",
            brand="Acme",
            category="Laptop",
        )
        keyboard = Product(
            name="Keyboard Mekanik",
            description="Keyboard dengan switch biru.",
            brand="KeyMaster",
            category="Aksesoris",
        )
        kb = KnowledgeBase([laptop, keyboard], [])

        ranked = kb.find_best_products("Laptop Acme", limit=1)

        self.assertEqual(len(ranked), 1)
        self.assertIs(ranked[0][0], laptop)
        self.assertGreater(ranked[0][1], 0.5)

    def test_normalize_price_idr_strips_currency_text(self) -> None:
        normalized = KnowledgeBase._normalize_price_idr("IDR 1.250.000 ")
        self.assertEqual(normalized, "1.250.000")


class RetrieverTests(unittest.TestCase):
    def test_build_context_joins_snippets_with_separator(self) -> None:
        class DummyKnowledgeBase:
            def __init__(self) -> None:
                self.calls = []

            def search(self, query: str, *, limit: int) -> list[str]:
                self.calls.append((query, limit))
                return ["produk", "faq"]

        dummy_kb = DummyKnowledgeBase()
        retriever = RAGRetriever(dummy_kb, top_k=2)

        context = retriever.build_context("Ada promo?")

        self.assertEqual(context, "produk\n\n---\n\nfaq")
        self.assertEqual(dummy_kb.calls, [("Ada promo?", 2)])


class CheckoutPlanTests(unittest.TestCase):
    def test_format_summary_contains_all_sections(self) -> None:
        plan = CheckoutPlan(
            cart_items=[CartItem(name="Laptop Pro", quantity=2, price_hint="Rp 15 jt")],
            shipping_address="Jakarta",
            contact="email@contoh.id",
            payment_method="Transfer Bank",
            next_steps=["Hubungi pelanggan", "Kirim invoice"],
            upsell="Tambahkan paket garansi.",
            notes="Pastikan stok tersedia.",
        )

        summary = plan.format_summary()

        self.assertIn("Keranjang Belanja", summary)
        self.assertIn("- Laptop Pro (jumlah: 2) perkiraan harga: Rp 15 jt", summary)
        self.assertIn("Alamat pengiriman: Jakarta", summary)
        self.assertIn("1. Hubungi pelanggan", summary)
        self.assertIn("Saran tambahan: Tambahkan paket garansi.", summary)
        self.assertIn("Catatan tambahan: Pastikan stok tersedia.", summary)


class WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = AgenticPurchaseWorkflow(agent=SimpleNamespace())

    def test_extract_json_handles_fenced_block(self) -> None:
        payload = """Berikut rencana:\n```json\n{\n  \"cart\": []\n}\n```"""
        extracted = self.workflow._extract_json(payload)
        self.assertEqual(extracted, '{\n  "cart": []\n}')

    def test_parse_cart_converts_quantities_and_filters(self) -> None:
        cart_payload = [
            {
                "name": "Laptop",
                "quantity": "2",
                "price": "Rp 10 jt",
            },
            {
                "produk": "Mouse",
                "qty": "banyak",
                "alasan": "Pelengkap",
            },
            "bukan kamus",
        ]

        items = self.workflow._parse_cart(cart_payload)

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].name, "Laptop")
        self.assertEqual(items[0].quantity, 2)
        self.assertEqual(items[0].price_hint, "Rp 10 jt")
        self.assertEqual(items[1].name, "Mouse")
        self.assertEqual(items[1].quantity, 1)
        self.assertEqual(items[1].reason, "Pelengkap")


class CliTests(unittest.TestCase):
    def test_parse_args_accepts_valid_values(self) -> None:
        args = app.parse_args([
            "--top-k",
            "5",
            "--tone",
            "Santai",
            "--server-port",
            "9000",
            "--no-gradio",
        ])

        self.assertEqual(args.top_k, 5)
        self.assertEqual(args.tone, "Santai")
        self.assertEqual(args.server_port, 9000)
        self.assertFalse(args.gradio)

    def test_parse_args_rejects_invalid_top_k(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                app.parse_args(["--top-k", "0"])


if __name__ == "__main__":
    unittest.main()