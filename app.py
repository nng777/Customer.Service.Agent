import argparse
import csv
import json
import os
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import difflib
import logging

try:  # ``requests`` is a very common dependency, but we guard the import.
    import requests
except ModuleNotFoundError as exc:  # pragma: no cover - only executed w/out dependency
    raise RuntimeError(
        "The 'requests' package is required to run this script. Install it with "
        "'pip install requests'."
    ) from exc


LOGGER = logging.getLogger(__name__)


def _read_text(path: Path) -> str:
    """Read a text file using UTF-8 encoding.

    Parameters
    ----------
    path:
        Location of the file to read.
    """

    LOGGER.debug("Reading text file: %s", path)
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> Any:
    LOGGER.debug("Reading JSON file: %s", path)
    return json.loads(_read_text(path))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    LOGGER.debug("Reading CSV file: %s", path)
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


@dataclass
class Product:
    """Representation of a product in the mock catalogue."""

    name: str
    description: str
    price: Optional[str] = None
    sku: Optional[str] = None
    url: Optional[str] = None

    @property
    def context(self) -> str:
        """Formatted text that can be passed to the language model."""

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
        """Load knowledge base content from files.

        ``catalog_path`` can be a CSV or JSON file with product fields.  The JSON
        format is expected to be a list of objects.  ``faq_path`` may be either a
        JSON file containing ``{"question": ..., "answer": ...}`` dictionaries or a
        Markdown file using heading/question pairs.
        """

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
        if path.suffix.lower() == ".csv":
            raw_items = _read_csv(path)
        else:
            raw_items = _read_json(path)
        products = [
            Product(
                name=item.get("name") or item.get("Nama") or item.get("product_name", ""),
                description=item.get("description")
                or item.get("Deskripsi")
                or item.get("product_description", ""),
                price=item.get("price") or item.get("Harga") or item.get("product_price"),
                sku=item.get("sku") or item.get("SKU"),
                url=item.get("url") or item.get("URL"),
            )
            for item in raw_items
            if item
        ]
        LOGGER.info("Loaded %d products from %s", len(products), path)
        return products

    @staticmethod
    def _load_faqs(path: Path) -> List[FAQ]:
        if path.suffix.lower() in {".md", ".markdown"}:
            content = _read_text(path)
            return KnowledgeBase._parse_markdown_faq(content)

        raw_items = _read_json(path)
        faqs = [
            FAQ(question=item.get("question", ""), answer=item.get("answer", ""))
            for item in raw_items
            if item
        ]
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
        products = [
            Product(
                name="Sepatu Lari AeroX",
                description="Sepatu lari ringan dengan bantalan responsif untuk pelari jarak jauh.",
                price="Rp1.299.000",
                sku="SP-AEROX-001",
                url="https://contoh-toko.id/produk/sepatu-lari-aerox",
            ),
            Product(
                name="Tas Ransel Harian Urban",
                description="Tas ransel tahan air dengan kompartemen laptop 15 inci.",
                price="Rp499.000",
                sku="TS-URBAN-010",
                url="https://contoh-toko.id/produk/tas-ransel-urban",
            ),
        ]

        faqs = [
            FAQ(
                question="Bagaimana kebijakan pengembalian barang?",
                answer="Pengembalian dapat dilakukan dalam 14 hari dengan menyertakan bukti pembelian.",
            ),
            FAQ(
                question="Apakah tersedia layanan pengiriman ekspres?",
                answer="Ya, kami bekerja sama dengan beberapa kurir untuk pengiriman ekspres 1-2 hari kerja.",
            ),
        ]

        return products, faqs

    def search(self, query: str, *, limit: int = 3) -> List[str]:
        """Retrieve relevant snippets for the provided query."""

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


class RAGRetriever:
    """Simple context retriever that builds prompts for the agent."""

    def __init__(self, knowledge_base: KnowledgeBase, *, top_k: int = 3) -> None:
        self.knowledge_base = knowledge_base
        self.top_k = top_k

    def build_context(self, query: str) -> str:
        snippets = self.knowledge_base.search(query, limit=self.top_k)
        divider = "\n\n---\n\n"
        return divider.join(snippets)


class GeminiChatModel:
    """Minimal HTTP client for the Gemini API."""

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
    """High-level orchestration of the retrieval augmented chatbot."""

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
            Jika informasinya tidak ada, jelaskan secara jujur dan tawarkan bantuan lanjutan.
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


def launch_gradio_chat(
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

    def _respond(message: str, history: list[tuple[str, str]]):  # type: ignore[override]
        if not message.strip():
            return "Silakan masukkan pertanyaan yang ingin Anda tanyakan."
        return agent.answer(message)

    description = textwrap.dedent(
        """
        Ajukan pertanyaan layanan pelanggan dalam Bahasa Indonesia.
        Sistem akan menggunakan basis pengetahuan internal untuk membantu menjawab.
        """
    ).strip()

    chat = gr.ChatInterface(
        fn=_respond,
        title="Agen Layanan Pelanggan (Bahasa Indonesia)",
        description=description,
    )

    chat.launch(server_name=server_name, server_port=server_port, share=share)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        help="Path ke katalog produk (CSV atau JSON).",
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

    if args.question:
        print(agent.answer(args.question))
        return 0

    if args.gradio:
        try:
            launch_gradio_chat(
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


if __name__ == "__main__":  # pragma: no cover - manual execution
    raise SystemExit(main())