import argparse
import json
import os
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import difflib
import logging
import requests



LOGGER = logging.getLogger(__name__)


def _read_text(path: Path) -> str:

    LOGGER.debug("Reading text file: %s", path)
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> Any:
    LOGGER.debug("Reading JSON file: %s", path)
    return json.loads(_read_text(path))


@dataclass
class Product:


    name: str
    description: str
    price: Optional[str] = None
    sku: Optional[str] = None
    url: Optional[str] = None

    @property
    def context(self) -> str:


        price = f"Harga: {self.price}" if self.price else ""
        sku = f"SKU: {self.sku}" if self.sku else ""
        url = f"URL: {self.url}" if self.url else ""
        lines = [f"Nama Produk: {self.name}", f"Deskripsi: {self.description}"]
        for metadata in (price, sku, url):
            if metadata:
                lines.append(metadata)
        return "\n".join(lines)


@dataclass
class FAQ:
    """Frequently asked question entry."""

    question: str
    answer: str

    @property
    def context(self) -> str:
        return f"Pertanyaan: {self.question}\nJawaban: {self.answer}"



@dataclass
class CartItem:


    name: str
    quantity: int = 1
    reason: Optional[str] = None
    price_hint: Optional[str] = None

    def format_line(self) -> str:
        parts = [f"- {self.name}"]
        if self.quantity != 1:
            parts.append(f"(jumlah: {self.quantity})")
        if self.price_hint:
            parts.append(f"perkiraan harga: {self.price_hint}")
        if self.reason:
            parts.append(f"alasan: {self.reason}")
        return " ".join(parts)


@dataclass
class CheckoutPlan:


    cart_items: List[CartItem]
    shipping_address: str
    contact: str
    payment_method: str
    next_steps: List[str]
    upsell: Optional[str] = None
    notes: Optional[str] = None

    def format_summary(self) -> str:
        lines: List[str] = [
            "RINGKASAN RENCANA PEMBELIAN",
            "===========================",
            "Keranjang Belanja:",
        ]

        if self.cart_items:
            lines.extend(item.format_line() for item in self.cart_items)
        else:
            lines.append("- Belum ada produk yang dipilih.")

        lines.extend(
            [
                "",
                "Informasi Checkout:",
                f"Alamat pengiriman: {self.shipping_address or 'Belum ditentukan'}",
                f"Kontak pelanggan: {self.contact or 'Belum ditentukan'}",
                f"Metode pembayaran: {self.payment_method or 'Belum dipilih'}",
            ]
        )

        if self.notes:
            lines.append(f"Catatan tambahan: {self.notes}")
        if self.upsell:
            lines.append(f"Saran tambahan: {self.upsell}")

        if self.next_steps:
            lines.extend(["", "Langkah berikutnya:"])
            for idx, step in enumerate(self.next_steps, start=1):
                lines.append(f"{idx}. {step}")

        return "\n".join(lines)

class KnowledgeBase:
    """In-memory storage for the product catalogue and FAQ documents."""

    def __init__(self, products: Sequence[Product], faqs: Sequence[FAQ]) -> None:
        self.products = list(products)
        self.faqs = list(faqs)
        LOGGER.debug(
            "Knowledge base initialized with %d products and %d FAQ entries",
            len(self.products),
            len(self.faqs),
        )

    @classmethod
    def from_files(
        cls,
        catalog_path: Optional[Path] = None,
        faq_path: Optional[Path] = None,
    ) -> "KnowledgeBase":

        products = cls._load_products(catalog_path) if catalog_path else []
        faqs = cls._load_faqs(faq_path) if faq_path else []

        if not products and not faqs:
            LOGGER.warning(
                "Knowledge base is empty. Loading fallback sample data for demo purposes."
            )
            products, faqs = cls._sample_data()

        return cls(products, faqs)

    @staticmethod
    def _load_products(path: Path) -> List[Product]:
        if path.suffix.lower() != ".json":
            raise ValueError(
                f"Unsupported product file format '{path.suffix}'. Only JSON is supported."
            )
        raw_payload = _read_json(path)
        products = KnowledgeBase._parse_products_payload(raw_payload)

        LOGGER.info("Loaded %d products from %s", len(products), path)
        return products

    @staticmethod
    def _load_faqs(path: Path) -> List[FAQ]:
        if path.suffix.lower() in {".md", ".markdown"}:
            content = _read_text(path)
            return KnowledgeBase._parse_markdown_faq(content)

        raw_payload = _read_json(path)
        faqs = KnowledgeBase._parse_faqs_payload(raw_payload)

        LOGGER.info("Loaded %d FAQ entries from %s", len(faqs), path)
        return faqs

    @staticmethod
    def _parse_markdown_faq(markdown_text: str) -> List[FAQ]:
        faqs: List[FAQ] = []
        question: Optional[str] = None
        answer_lines: List[str] = []

        for line in markdown_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                if question and answer_lines:
                    faqs.append(FAQ(question=question, answer="\n".join(answer_lines).strip()))
                question = stripped.lstrip("# ")
                answer_lines = []
            elif stripped:
                answer_lines.append(stripped)

        if question and answer_lines:
            faqs.append(FAQ(question=question, answer="\n".join(answer_lines).strip()))

        LOGGER.info("Parsed %d FAQ entries from Markdown", len(faqs))
        return faqs

    @staticmethod
    def _sample_data() -> tuple[list[Product], list[FAQ]]:
        sample_path = Path(__file__).with_name("data.json")
        try:
            payload = _read_json(sample_path)
        except FileNotFoundError:
            LOGGER.warning(
                "Sample data file %s not found; returning empty fallback.", sample_path
            )
            return [], []
        except json.JSONDecodeError as exc:
            LOGGER.error("Failed to parse sample data JSON %s: %s", sample_path, exc)
            return [], []

        products = KnowledgeBase._parse_products_payload(payload)
        faqs = KnowledgeBase._parse_faqs_payload(payload)

        LOGGER.info(
            "Loaded %d sample products and %d FAQ entries from %s",
            len(products),
            len(faqs),
            sample_path,
        )

        return products, faqs

    def search(self, query: str, *, limit: int = 3) -> List[str]:

        candidates: List[str] = []

        product_names = [product.name for product in self.products]
        faq_questions = [faq.question for faq in self.faqs]

        product_matches = difflib.get_close_matches(query, product_names, n=limit, cutoff=0.1)
        faq_matches = difflib.get_close_matches(query, faq_questions, n=limit, cutoff=0.1)

        LOGGER.debug("Product matches: %s", product_matches)
        LOGGER.debug("FAQ matches: %s", faq_matches)

        for product in self.products:
            if product.name in product_matches:
                candidates.append(product.context)

        for faq in self.faqs:
            if faq.question in faq_matches:
                candidates.append(faq.context)

        if not candidates:
            LOGGER.debug("No fuzzy matches found. Falling back to first items.")
            for product in self.products[:limit]:
                candidates.append(product.context)
            for faq in self.faqs[: limit - len(candidates)]:
                candidates.append(faq.context)

        return candidates[:limit]

    @staticmethod
    def _parse_products_payload(payload: Any) -> List[Product]:
        if isinstance(payload, list):
            raw_items = payload
        elif isinstance(payload, dict):
            raw_items = payload.get("products", [])
        else:
            LOGGER.error(
                "Unsupported product payload type: %s", type(payload).__name__
            )
            return []

        products: List[Product] = []
        for item in raw_items:
            product = KnowledgeBase._build_product(item)
            if product:
                products.append(product)
        return products

    @staticmethod
    def _parse_faqs_payload(payload: Any) -> List[FAQ]:
        if isinstance(payload, list):
            raw_items = payload
        elif isinstance(payload, dict):
            raw_items = payload.get("faqs", [])
        else:
            LOGGER.error(
                "Unsupported FAQ payload type: %s", type(payload).__name__
            )
            return []

        faqs: List[FAQ] = []
        for item in raw_items:
            faq = KnowledgeBase._build_faq(item)
            if faq:
                faqs.append(faq)
        return faqs

    @staticmethod
    def _build_product(item: Any) -> Optional[Product]:
        if not isinstance(item, dict):
            return None

        name = (
            item.get("name")
            or item.get("Nama")
            or item.get("product_name")
            or ""
        )
        if not name:
            return None

        description_parts: List[str] = []
        description_raw = (
            item.get("description")
            or item.get("Deskripsi")
            or item.get("product_description")
        )
        if description_raw:
            description_parts.append(str(description_raw).strip())

        metadata_fields = (
            ("Brand", "brand"),
            ("Kategori", "category"),
            ("Varian", "variant"),
            ("Marketplace", "marketplace"),
            ("Nama Toko", "store_name"),
            ("Tipe Toko", "store_type"),
            ("Status Stok", "stock_status"),
            ("Terakhir diperbarui", "last_updated"),
        )

        for label, key in metadata_fields:
            value = item.get(key)
            if value:
                description_parts.append(f"{label}: {value}")

        store_url = item.get("store_url")
        if store_url:
            description_parts.append(f"URL Toko: {store_url}")

        description = "\n".join(description_parts).strip() or "Tidak ada deskripsi."

        price = KnowledgeBase._format_price(item)
        sku = item.get("sku") or item.get("SKU") or item.get("id")
        url = item.get("product_url") or item.get("url") or item.get("URL")

        return Product(name=name, description=description, price=price, sku=sku, url=url)

    @staticmethod
    def _build_faq(item: Any) -> Optional[FAQ]:
        if not isinstance(item, dict):
            return None

        question = item.get("question") or item.get("pertanyaan")
        answer = item.get("answer") or item.get("jawaban")

        if not question and not answer:
            return None

        return FAQ(question=str(question or "").strip(), answer=str(answer or "").strip())

    @staticmethod
    def _format_price(item: Dict[str, Any]) -> Optional[str]:
        direct_price = (
            item.get("price") or item.get("Harga") or item.get("product_price")
        )
        if direct_price:
            return str(direct_price)

        price_idr = item.get("price_idr")
        if price_idr is not None:
            if isinstance(price_idr, (int, float)):
                formatted = f"{price_idr:,.0f}".replace(",", ".")
            else:
                formatted = str(price_idr)
            currency = item.get("currency") or item.get("currency_code")
            return f"{currency} {formatted}".strip() if currency else formatted

        return None


class RAGRetriever:

    def __init__(self, knowledge_base: KnowledgeBase, *, top_k: int = 3) -> None:
        self.knowledge_base = knowledge_base
        self.top_k = top_k

    def build_context(self, query: str) -> str:
        snippets = self.knowledge_base.search(query, limit=self.top_k)
        divider = "\n\n---\n\n"
        return divider.join(snippets)


class GeminiChatModel:

    DEFAULT_MODEL = "gemini-2.0-flash"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. Provide a valid Gemini API key."
            )

        self.model = model or os.environ.get("GEMINI_MODEL") or self.DEFAULT_MODEL
        self.api_url = self.BASE_URL.format(model=self.model)
        LOGGER.debug("Gemini model configured: %s", self.model)

    def generate(self, prompt: str) -> str:
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ]
                }
            ]
        }
        LOGGER.debug("Sending request to Gemini with payload length %d", len(prompt))
        response = requests.post(
            self.api_url,
            params={"key": self.api_key},
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Gemini API returned status {response.status_code}: {response.text[:200]}"
            )

        body = response.json()
        try:
            candidates = body["candidates"]
            first_candidate = candidates[0]
            parts = first_candidate["content"]["parts"]
            text_parts = [part["text"] for part in parts if "text" in part]
            answer = "\n".join(text_parts).strip()
        except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover - depends on API
            raise RuntimeError(
                "Unexpected response structure from Gemini API: "
                f"{json.dumps(body, ensure_ascii=False)[:200]}"
            ) from exc

        LOGGER.debug("Received response length %d", len(answer))
        return answer


class CustomerServiceAgent:

    def __init__(
        self,
        retriever: RAGRetriever,
        model: GeminiChatModel,
        *,
        tone: str = "Profesional dan ramah",
    ) -> None:
        self.retriever = retriever
        self.model = model
        self.tone = tone

    def build_prompt(self, query: str, context: str) -> str:
        instruction = textwrap.dedent(
            f"""
            Kamu adalah agen layanan pelanggan untuk toko e-commerce di Indonesia.
            Jawablah selalu dalam Bahasa Indonesia dengan nada {self.tone.lower()}.
            Gunakan informasi pada bagian KONTEKS untuk memberikan jawaban yang akurat.
            Jika informasinya tidak ada, cari ulang terlebih dahulu berdasarkan nama produk atau kategori, dan bila masih tidak dapat ditemukan, jelaskan secara jujur dan tawarkan bantuan lanjutan.
            """
        ).strip()

        prompt = textwrap.dedent(
            f"""
            {instruction}

            KONTEKS:
            {context}

            PERTANYAAN PELANGGAN:
            {query}

            JAWABAN:
            """
        ).strip()

        LOGGER.debug("Prompt built with %d characters", len(prompt))
        return prompt

    def answer(self, query: str) -> str:
        context = self.retriever.build_context(query)
        prompt = self.build_prompt(query, context)
        try:
            return self.model.generate(prompt)
        except Exception as exc:  # pragma: no cover - depends on external API
            LOGGER.error("Gemini request failed: %s", exc)
            fallback = textwrap.dedent(
                f"""
                Maaf, saat ini saya tidak dapat menghubungi model bahasa. Berikut informasi
                yang mungkin membantu:

                {context}

                Silakan coba lagi beberapa saat atau hubungi tim dukungan kami.
                """
            ).strip()
            return fallback


    def plan_purchase(self, request: str) -> "CheckoutPlan":
        workflow = AgenticPurchaseWorkflow(self)
        return workflow.run(request)


class AgenticPurchaseWorkflow:

    """Structured checkout workflow orchestrated by the agent."""

    JSON_KEYS = {
        "cart": ("cart", "items", "produk", "keranjang"),
        "shipping": ("shipping", "pengiriman", "delivery"),
        "payment": ("payment", "pembayaran"),
        "next_steps": ("next_steps", "steps", "langkah"),
        "upsell": ("upsell", "cross_sell", "saran"),
    }

    def __init__(self, agent: CustomerServiceAgent) -> None:
        self.agent = agent

    def run(self, request: str) -> CheckoutPlan:
        context = self.agent.retriever.build_context(request)
        prompt = self._build_prompt(request, context)

        try:
            response = self.agent.model.generate(prompt)
            plan = self._parse_plan(response)
            if plan:
                return plan
            LOGGER.warning("Model response could not be parsed into checkout plan.")
        except Exception as exc:  # pragma: no cover - depends on external API
            LOGGER.error("Workflow generation failed: %s", exc)

        return self._fallback_plan(request)

    def _build_prompt(self, request: str, context: str) -> str:
        instruction = textwrap.dedent(
            """
            Kamu adalah agen pembelian yang membantu pelanggan e-commerce di Indonesia
            untuk menyelesaikan proses checkout secara terstruktur. Susun rencana
            pembelian yang mencakup produk yang relevan, ringkasan keranjang, informasi
            pengiriman, pilihan pembayaran, dan langkah tindak lanjut.

            Berikan jawaban dalam JSON valid dengan format berikut:
            {
              "cart": [
                {
                  "product_name": "...",
                  "quantity": 1,
                  "reason": "...",
                  "price_hint": "..."
                }
              ],
              "shipping": {
                "address": "...",
                "contact": "...",
                "notes": "..."
              },
              "payment": {
                "method": "...",
                "instructions": "..."
              },
              "next_steps": ["Langkah 1", "Langkah 2"],
              "upsell": "Saran tambahan jika ada"
            }

            Jika informasi tertentu belum diketahui, isi dengan penjelasan singkat.
            Hindari menambahkan teks lain di luar JSON.
            """
        ).strip()

        prompt = textwrap.dedent(
            f"""
            {instruction}

            KONTEKS PRODUK DAN FAQ:
            {context}

            PERMINTAAN PELANGGAN:
            {request}
            """
        ).strip()

        LOGGER.debug("Purchase workflow prompt length %d", len(prompt))
        return prompt

    def _parse_plan(self, raw_response: str) -> Optional[CheckoutPlan]:
        cleaned = self._extract_json(raw_response)
        if not cleaned:
            return None

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            LOGGER.debug("Failed to decode JSON from workflow response: %s", raw_response)
            return None

        cart_entries = self._get_first_key(payload, "cart", default=[])
        cart_items = self._parse_cart(cart_entries)

        shipping_data = self._get_first_key(payload, "shipping", default={})
        payment_data = self._get_first_key(payload, "payment", default={})
        next_steps = self._get_first_key(payload, "next_steps", default=[])
        upsell = self._get_first_key(payload, "upsell")

        shipping_address = self._select_text(
            shipping_data, ("address", "alamat", "detail", "destination")
        )
        contact = self._select_text(shipping_data, ("contact", "phone", "kontak", "email"))
        notes = self._select_text(shipping_data, ("notes", "catatan"))

        payment_method = self._select_text(
            payment_data, ("method", "metode", "tipe", "channel")
        )
        payment_notes = self._select_text(
            payment_data, ("instructions", "catatan", "notes", "detail")
        )

        steps_list = [str(step).strip() for step in (next_steps or []) if str(step).strip()]

        return CheckoutPlan(
            cart_items=cart_items,
            shipping_address=shipping_address or "Perlu konfirmasi pelanggan.",
            contact=contact or "Perlu data kontak pelanggan.",
            payment_method=payment_method or "Perlu memilih metode pembayaran.",
            next_steps=steps_list if steps_list else [
                "Konfirmasi alamat dan kontak pelanggan.",
                "Kirim tautan pembayaran kepada pelanggan.",
            ],
            upsell=str(upsell).strip() if upsell else None,
            notes=notes or payment_notes,
        )

    def _extract_json(self, raw: str) -> Optional[str]:
        if not raw:
            return None

        text = raw.strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.startswith("```")]
            text = "\n".join(lines).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]

    def _get_first_key(self, payload: Dict[str, Any], key: str, default: Any = None) -> Any:
        candidates = self.JSON_KEYS.get(key, (key,))
        for candidate in candidates:
            if isinstance(payload, dict) and candidate in payload:
                return payload[candidate]
        return default

    def _parse_cart(self, entries: Any) -> List[CartItem]:
        items: List[CartItem] = []
        if not isinstance(entries, Iterable):
            return items

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = self._select_text(
                entry, ("product_name", "name", "nama", "title", "produk")
            )
            if not name:
                continue
            quantity_raw = entry.get("quantity") or entry.get("qty") or entry.get("jumlah")
            try:
                quantity = int(quantity_raw) if quantity_raw is not None else 1
            except (TypeError, ValueError):
                quantity = 1
            reason = self._select_text(entry, ("reason", "alasan", "why"))
            price_hint = self._select_text(entry, ("price_hint", "price", "harga"))
            items.append(
                CartItem(
                    name=name,
                    quantity=quantity if quantity > 0 else 1,
                    reason=reason,
                    price_hint=price_hint,
                )
            )

        return items

    def _select_text(self, payload: Any, keys: Tuple[str, ...]) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        for key in keys:
            if key in payload and payload[key]:
                return str(payload[key]).strip()
        return None

    def _fallback_plan(self, request: str) -> CheckoutPlan:
        product = self._match_product(request)
        cart_items = [CartItem(name=product.name, reason="Produk paling relevan.")]

        return CheckoutPlan(
            cart_items=cart_items,
            shipping_address="Perlu alamat pengiriman dari pelanggan.",
            contact="Minta nomor telepon atau email pelanggan.",
            payment_method="Pilih metode pembayaran yang disetujui pelanggan.",
            next_steps=[
                "Konfirmasi detail produk dan jumlah.",
                "Kumpulkan alamat lengkap dan metode pembayaran.",
                "Buat pesanan di sistem internal dan kirim konfirmasi.",
            ],
            upsell="Tawarkan aksesoris atau layanan terkait jika tersedia.",
        )

    def _match_product(self, request: str) -> Product:
        products = self.agent.retriever.knowledge_base.products
        if not products:
            return Product(name="Produk tidak ditemukan", description="")

        names = [product.name for product in products]
        matches = difflib.get_close_matches(request, names, n=1, cutoff=0.3)
        if matches:
            for product in products:
                if product.name == matches[0]:
                    return product
        return products[0]

def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def create_agent(args: argparse.Namespace) -> CustomerServiceAgent:
    knowledge_base = KnowledgeBase.from_files(args.catalog, args.faq)
    retriever = RAGRetriever(knowledge_base, top_k=args.top_k)
    model = GeminiChatModel(api_key=args.api_key, model=args.model)
    return CustomerServiceAgent(retriever, model, tone=args.tone)


def interactive_chat(agent: CustomerServiceAgent) -> None:
    print("Masuk ke mode percakapan. Ketik 'keluar' untuk mengakhiri sesi.")
    while True:
        try:
            question = input("Anda: ")
        except (EOFError, KeyboardInterrupt):
            print("\nSesi dihentikan.")
            break

        if question.strip().lower() in {"keluar", "exit", "quit"}:
            print("Terima kasih telah menggunakan layanan kami.")
            break

        if not question.strip():
            print("Silakan masukkan pertanyaan.")
            continue

        answer = agent.answer(question)
        print(f"Agen: {answer}\n")


def launch_gradio_app(
    agent: CustomerServiceAgent,
    *,
    server_name: str,
    server_port: int,
    share: bool,
) -> None:
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Gradio belum terpasang. Jalankan 'pip install gradio' untuk menggunakan --gradio."
        ) from exc

    LOGGER.info(
        "Meluncurkan antarmuka Gradio di %s:%s (share=%s)", server_name, server_port, share
    )

    def _respond(message, history):  # type: ignore[override]

        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = str(message) if message is not None else ""

        if not content.strip():
            return {
                "role": "assistant",
                "content": "Silakan masukkan pertanyaan yang ingin Anda tanyakan.",
            }

        answer = agent.answer(content)
        return {"role": "assistant", "content": answer}


    def _plan_checkout(purchase_request: str) -> str:
        if not purchase_request.strip():
            return "Silakan jelaskan kebutuhan pembelian Anda terlebih dahulu."

        plan = agent.plan_purchase(purchase_request)
        return plan.format_summary()


    description = textwrap.dedent(
        """
        Ajukan pertanyaan layanan pelanggan dalam Bahasa Indonesia.
        Sistem akan menggunakan basis pengetahuan internal untuk membantu menjawab.
        """
    ).strip()

    checkout_description = textwrap.dedent(
        """
        Masukkan permintaan pembelian Anda untuk mendapatkan ringkasan checkout
        terstruktur, termasuk produk yang dipilih, informasi pengiriman,
        metode pembayaran, dan langkah berikutnya.
        """
    ).strip()


    chat = gr.ChatInterface(
        fn=_respond,
        title="Agen Layanan Pelanggan (Bahasa Indonesia)",
        description=description,
        type="messages",
    )

    with gr.Blocks(title="Agen Layanan Pelanggan") as demo:
        with gr.Tab("Percakapan"):
            chat.render()

        with gr.Tab("Checkout"):
            gr.Markdown(f"### Rencana Checkout\n{checkout_description}")
            purchase_request = gr.Textbox(
                label="Kebutuhan Pembelian",
                placeholder="Contoh: Saya ingin membeli payung otomatis.",
                lines=4,
            )
            plan_output = gr.Textbox(
                label="Rencana Checkout", lines=15, elem_id="checkout-plan-output"
            )
            gr.HTML(
                textwrap.dedent(
                    """
                    <style>
                      #checkout-plan-printable {
                        display: none;
                        font-family: "Segoe UI", system-ui, sans-serif;
                        font-size: 14px;
                        line-height: 1.5;
                        margin: 40px;
                      }
                      #checkout-plan-printable h1 {
                        font-size: 20px;
                        text-align: center;
                        margin-bottom: 24px;
                      }
                      #checkout-plan-printable pre {
                        white-space: pre-wrap;
                        word-break: break-word;
                        margin: 0;
                      }
                      @media print {
                        body {
                          margin: 0;
                        }
                        body * {
                          visibility: hidden;
                        }
                        #checkout-plan-printable,
                        #checkout-plan-printable * {
                          visibility: visible;
                        }
                        #checkout-plan-printable {
                          position: fixed;
                          inset: 0;
                          display: block !important;
                          padding: 40px;
                        }
                      }
                    </style>
                    <div id="checkout-plan-printable" role="presentation">
                      <h1>Rencana Checkout</h1>
                      <pre></pre>
                    </div>
                    """
                )
            )
            with gr.Row():
                submit_btn = gr.Button("Buat Rencana", variant="primary")
                print_btn = gr.Button("Print")

            print_btn.click(
                None,
                [],
                [],
                js=textwrap.dedent(
                    """
                    () => {
                        const textarea = document.querySelector(
                            '#checkout-plan-output textarea'
                        );
                        const plan = textarea ? textarea.value : '';

                        if (!plan.trim()) {
                            window.alert('Rencana checkout masih kosong.');
                            return [];
                        }

                        const printable = document.getElementById('checkout-plan-printable');
                        const pre = printable ? printable.querySelector('pre') : null;

                        if (!printable || !pre) {
                            window.alert('Gagal menyiapkan tampilan cetak.');
                            return [];
                        }

                        const previousDisplay = printable.style.display;
                        pre.textContent = plan;
                        printable.style.display = 'block';

                        const cleanup = () => {
                            printable.style.display = previousDisplay || 'none';
                            window.removeEventListener('afterprint', cleanup);
                        };

                        window.addEventListener('afterprint', cleanup);
                        window.print();

                        // Safari fallback: ensure cleanup even if afterprint doesn't fire.
                        setTimeout(cleanup, 1000);
                        return [];
                    }
                    """
                ),
            )

            submit_btn.click(_plan_checkout, purchase_request, plan_output)
            purchase_request.submit(_plan_checkout, purchase_request, plan_output)

    demo.launch(server_name=server_name, server_port=server_port, share=share)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        help="Path ke katalog produk (JSON).",
    )
    parser.add_argument(
        "--faq",
        type=Path,
        help="Path ke FAQ (JSON atau Markdown).",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        help="Override GEMINI_API_KEY jika diperlukan.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Jumlah konteks yang diambil untuk setiap pertanyaan (default: 3).",
    )
    parser.add_argument(
        "--tone",
        default="Profesional dan ramah",
        help="Nada respons agen dalam Bahasa Indonesia.",
    )
    parser.add_argument(
        "--model",
        help=(
            "Nama model Gemini yang akan digunakan (default: gemini-2.0-flash atau variabel GEMINI_MODEL)."
        ),
    )
    parser.add_argument(
        "--question",
        help="Ajukan satu pertanyaan dan tampilkan jawabannya tanpa mode interaktif.",
    )
    parser.add_argument(
        "--purchase",
        help="Jalankan workflow pembelian terstruktur berdasarkan permintaan pelanggan.",
    )
    parser.add_argument(
        "--no-gradio",
        dest="gradio",
        action="store_false",
        help="Gunakan antarmuka CLI teks sebagai ganti Gradio.",
    )
    parser.add_argument(
        "--server-name",
        default="127.0.0.1",
        help="Nama host untuk server Gradio (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Port untuk server Gradio (default: 7860).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Aktifkan share URL publik dari Gradio (gunakan dengan hati-hati).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Aktifkan log level DEBUG.",
    )

    parser.set_defaults(gradio=True)

    args = parser.parse_args(argv)

    if args.top_k < 1:
        parser.error("--top-k harus bernilai positif")

    if args.server_port < 0:
        parser.error("--server-port harus bernilai non-negatif")

    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        agent = create_agent(args)
    except Exception as exc:
        LOGGER.error("Gagal membuat agen: %s", exc)
        return 1

    if args.question and args.gradio:
        LOGGER.error(
            "Tidak dapat menggunakan --question bersamaan dengan antarmuka Gradio. "
            "Tambahkan --no-gradio untuk menggunakan mode pertanyaan tunggal."
        )
        return 1

    if args.purchase and args.question:
        LOGGER.error("Gunakan hanya salah satu dari --question atau --purchase dalam satu waktu.")
        return 1


    if args.question:
        print(agent.answer(args.question))
        return 0

    if args.purchase:
        plan = agent.plan_purchase(args.purchase)
        print(plan.format_summary())
        return 0


    if args.gradio:
        try:
            launch_gradio_app(
                agent,
                server_name=args.server_name,
                server_port=args.server_port,
                share=args.share,
            )
        except Exception as exc:
            LOGGER.error("Gagal menjalankan Gradio: %s", exc)
            return 1
        return 0

    interactive_chat(agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())