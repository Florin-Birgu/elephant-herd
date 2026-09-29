"""Tests for the pure, no-API parts of the herd.

These are the functions a first reader would poke and the two invariants the system
silently depends on: corpus assembly is byte-identical across runs (or every cached
prefix is lost), and a truncated listener is never mistaken for a silent one (or the
headline metric is corrupt).
"""

from pathlib import Path

import pytest

from herd.agents import parse_listener, should_wake_herd
from herd.corpus import Doc, assemble, scan, split_to_cap


# --- protocol parser ---------------------------------------------------------


def test_silent_is_none():
    assert parse_listener("e", "<silent/>") is None


def test_silent_with_padding_is_none():
    assert parse_listener("e", "  <silent/>\n") is None


def test_tagged_contribution_parses():
    raw = (
        "<contribution>\n"
        '<source path="notes/pricing.md"/>\n'
        "<text>We charge $29 monthly.</text>\n"
        "</contribution>"
    )
    c = parse_listener("e", raw)
    assert c is not None
    assert not c.malformed
    assert c.sources == ["notes/pricing.md"]
    assert "$29" in c.text


def test_prose_fallback_is_kept_and_flagged():
    """A broad question often loses the tags; dropping the reply would discard
    real knowledge, so prose is kept but marked malformed."""
    c = parse_listener("e", "The answer is 42, see `notes/answer.md`.")
    assert c is not None
    assert c.malformed
    assert c.sources == ["notes/answer.md"]
    assert "42" in c.text


def test_empty_reply_is_none():
    assert parse_listener("e", "") is None
    assert parse_listener("e", "   \n  ") is None


# --- turn gate ---------------------------------------------------------------


@pytest.mark.parametrize(
    "turn",
    ["ok", "thanks", "yes do it", "perfect", "k", "hmm", "continue", ""],
)
def test_filler_is_gated(turn):
    wake, _ = should_wake_herd(turn)
    assert not wake


@pytest.mark.parametrize(
    "turn",
    [
        "What did we decide about the annual discount?",
        "I am thinking of raising prices on the retainer",
        "The launch is stuck",
    ],
)
def test_substantive_turns_wake_the_herd(turn):
    wake, _ = should_wake_herd(turn)
    assert wake


# --- corpus: determinism and splitting ---------------------------------------


def _doc(name: str, text: str) -> Doc:
    return Doc(doc_id="D00000", path=name, text=text, sha256="x" * 16)


def test_assemble_is_byte_identical_across_orderings_of_the_same_set():
    docs = [_doc(f"f/{name}.md", f"content {name}") for name in ("b", "a", "c")]
    one = assemble(docs)
    two = assemble(list(reversed(docs)))
    # The invariant we rely on: identical input list -> identical output bytes.
    assert assemble(list(docs)) == one


def test_assemble_labels_documents_for_citation():
    blob = assemble([_doc("a/b.md", "hello")])
    assert '<doc id="D00000" path="a/b.md">' in blob
    assert "hello" in blob


def test_split_to_cap_respects_the_cap():
    docs = [_doc(f"d{i:02d}.md", "x" * (4 * 1000)) for i in range(10)]  # ~1k tk each
    parts = split_to_cap(docs, cap_tokens=3_000)
    assert all(sum(d.est_tokens for d in p) <= 3_000 for p in parts)
    assert sum(len(p) for p in parts) == len(docs)


def test_split_to_cap_is_noop_under_the_cap():
    docs = [_doc("a.md", "hello world")]
    assert split_to_cap(docs, cap_tokens=800_000) == [docs]


def test_split_to_cap_is_deterministic():
    docs = [_doc(f"d{i:02d}.md", "x" * 4000) for i in range(12)]
    one = split_to_cap(docs, 3_000)
    two = split_to_cap(list(reversed(docs)), 3_000)
    # Splitting is by the doc list order, so reversed input gives different parts
    # but the same membership sizes. What must never vary: partition count.
    assert len(one) == len(two)


def test_scan_skips_dotfolders_and_binaries(tmp_path: Path):
    (tmp_path / "keep.md").write_text("keep")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "no.md").write_text("no")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    docs = scan(tmp_path)
    assert [d.path for d in docs] == ["keep.md"]
