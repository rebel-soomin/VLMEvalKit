"""US-007 — GOT-OCR2.0 OCR-query routing.

GOT-OCR2.0 takes no free-form question: ``GotOcr2Processor`` synthesises
one of a fixed set of queries (``"OCR: "``, ``"OCR with format: "``,
``"[box] OCR: "``, ...). :class:`~vlmeval.vlm.rbln.RBLNGotOcr2` therefore
maps a dataset question onto a *query mode*, and these tests lock that
mapping against the real OCRBench_v2 question templates.

Nothing here loads a model, a processor or touches the NPU: the routing
functions are exercised on an unbound instance so no ``__init__`` runs.
"""

from __future__ import annotations

import pytest

from vlmeval.vlm.rbln import RBLNGotOcr2


class _Router:
    """Unbound stand-in exposing only the routing surface."""

    def __init__(self, ocr_mode: str = 'auto'):
        self.ocr_mode = ocr_mode

    _extract_box = staticmethod(RBLNGotOcr2._extract_box)
    _resolve_mode = RBLNGotOcr2._resolve_mode
    _processor_kwargs = RBLNGotOcr2._processor_kwargs

    def _debug_log(self, *a, **k):
        pass


# Verbatim OCRBench_v2 / OCRBench_v2_MINI question templates.
_NORMALIZED = ('The coordinates have been normalized ranging from 0 to 1000 '
               'by the image width and height.')

_PLAIN = [
    ('text recognition en', 'what is written in the image?'),
    ('full-page OCR en',
     'Read all the text in the image. Directly output the content and split '
     'the texts with space.'),
    ('full-page OCR cn', 'Read all the text in the image.'),
    ('cognition VQA en', 'What is the telephone number of the store?'),
    ('text counting en',
     "How many times does the character 'e' appear in the picture? Please "
     'output the exact number without any additional explanation.'),
    # Grounding / spotting mention normalized coordinates but only carry the
    # literal (x1, y1, x2, y2) placeholders — must NOT enter the box path.
    ('text grounding en',
     "Where is the region of the text 'SELINCOLN'? Output the normalized "
     'coordinates of the left-top and right-bottom corners of the bounding '
     'box. The coordinates should be normalized ranging from 0 to 1000 by the '
     'image width and height.Your answer should be in the following '
     'format:(x1, y1, x2, y2)'),
    ('text spotting en',
     'Spotting all the text in the image with word-level. Output the '
     'normalized coordinates ... Your answer should be in the following '
     'format:[(x1, y1, x2, y2, text content), (x1, y1, x2, y2, text content)...]'),
]

_FORMAT = [
    ('document parsing en', 'convert the privided document into markdown format.'),
    ('document parsing cn', 'Parse the document image in Markdown format'),
    ('table parsing en', 'Please represent this table with the HTML-format in text.'),
    ('formula recognition en',
     'What is the Latex tag for mathematical expression in images?'),
    ('formula recognition cn', '将图中的数学公式转换为LaTex表达式'),
]

_BOX_ONLY = [
    ('fine-grained text recognition en',
     f'Recognize the text within the [423, 617, 524, 700] of the image. {_NORMALIZED}',
     [423, 617, 524, 700]),
    ('text translation cn',
     'Please translate the text extracted from the area defined by '
     f'[513, 366, 887, 454] in this image to English. {_NORMALIZED}',
     [513, 366, 887, 454]),
]

_BOX_AND_FORMAT = [
    ('table parsing cn',
     'Parse the HTML-formatted table structure within the region '
     f'[136, 750, 863, 870] in the image {_NORMALIZED}',
     [136, 750, 863, 870]),
]


@pytest.mark.parametrize('category,question', _PLAIN, ids=[c[0] for c in _PLAIN])
def test_plain_queries(category, question):
    fmt, box = _Router()._resolve_mode(question)
    assert (fmt, box) == (False, None)


@pytest.mark.parametrize('category,question', _FORMAT, ids=[c[0] for c in _FORMAT])
def test_format_queries(category, question):
    fmt, box = _Router()._resolve_mode(question)
    assert fmt is True
    assert box is None


@pytest.mark.parametrize('category,question,expected', _BOX_ONLY,
                         ids=[c[0] for c in _BOX_ONLY])
def test_box_queries(category, question, expected):
    fmt, box = _Router()._resolve_mode(question)
    assert box == expected
    assert fmt is False


@pytest.mark.parametrize('category,question,expected', _BOX_AND_FORMAT,
                         ids=[c[0] for c in _BOX_AND_FORMAT])
def test_box_and_format_compose(category, question, expected):
    """A region-scoped parsing question must yield BOTH axes — the processor
    composes them into ``"[box] OCR with format: "``."""
    fmt, box = _Router()._resolve_mode(question)
    assert box == expected
    assert fmt is True


def test_ocr_mode_plain_overrides_routing():
    q = 'convert the privided document into markdown format.'
    assert _Router('plain')._resolve_mode(q) == (False, None)


def test_ocr_mode_format_overrides_routing():
    q = 'what is written in the image?'
    assert _Router('format')._resolve_mode(q) == (True, None)


def test_ocr_mode_validated():
    with pytest.raises(ValueError):
        RBLNGotOcr2(model_path='dummy', ocr_mode='nonsense')


def test_box_requires_coordinate_context():
    """Four bracketed integers alone are not a region — a KIE question that
    happens to contain them must stay on the plain path."""
    q = "Find out the value of 'ratios' [1, 2, 3, 4] stated in the image."
    assert _Router()._extract_box(q) is None


@pytest.mark.parametrize('box', [
    [500, 100, 400, 200],   # x1 >= x2
    [100, 500, 200, 400],   # y1 >= y2
    [10, 10, 2000, 500],    # out of the 0-1000 normalized range
])
def test_degenerate_boxes_rejected(box):
    q = (f'Recognize the text within the {box} of the image. {_NORMALIZED}')
    assert _Router()._extract_box(q) is None


class _Img:
    def __init__(self, w, h):
        self.size = (w, h)


def test_box_rescaled_to_pixels_for_processor():
    """GotOcr2Processor.preprocess_box_annotation divides by the image size,
    so it wants pixel coordinates. The wrapper converts the prompt's 0-1000
    values to pixels; the round trip must land back on the same 0-1000 box
    (modulo the processor's int() truncation).
    """
    q = f'Recognize the text within the [250, 500, 750, 1000] of the image. {_NORMALIZED}'
    kwargs = _Router()._processor_kwargs(q, _Img(1240, 1754))
    assert kwargs['format'] is False
    assert kwargs['box'] == pytest.approx([310.0, 877.0, 930.0, 1754.0])
    # Round trip the way the processor does it.
    round_tripped = [int(v / d * 1000) for v, d in
                     zip(kwargs['box'], (1240, 1754, 1240, 1754))]
    assert round_tripped == [250, 500, 750, 1000]


def test_plain_query_passes_no_box_kwarg():
    kwargs = _Router()._processor_kwargs('what is written in the image?', _Img(800, 600))
    assert kwargs == {'format': False}
