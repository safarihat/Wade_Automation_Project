# invoice_reconciliation/services/parser.py
import os
import logging
import re
from decimal import Decimal, InvalidOperation
from datetime import datetime, date
from typing import Optional, List, Tuple, Dict, Any, Pattern, TYPE_CHECKING

# External libraries (all optional at runtime)
# Install with: pip install pdfplumber pytesseract pillow pdf2image opencv-python-headless numpy

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None

try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None

try:
    from PIL import Image, ImageOps, ImageFilter  # type: ignore
except Exception:
    Image = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from pdf2image import convert_from_path  # type: ignore
except Exception:
    convert_from_path = None

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore


# Number pattern that tolerates thousand separators (., or space) and decimal separators (., or ,)
# Examples matched: 1,234.56 | 1.234,56 | 1 234,56 | 1234.56 | 1,234 | 1.234
AMOUNT_NUMBER_RE = re.compile(r"(?<!\w)(?:\d{1,3}(?:[.,\s]\d{3})+|\d+)(?:[.,]\d{2})?(?!\d)")
# Currency tokens either side of the number (not strictly required to match the amount)
CURRENCY_TOKENS = [
    "USD", "NZD", "AUD", "GBP", "EUR", "CAD", "ZAR",
    "R$", "$", "£", "€", "NZ$", "AU$", "US$", "C$"
]

DATE_PATTERNS = [
    (re.compile(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b"), ("%Y-%m-%d", "%Y/%m/%d")),
    (re.compile(r"\b(0?[1-9]|[12]\d|3[01])[-/](0?[1-9]|1[0-2])[-/](20\d{2})\b"), ("%d-%m-%Y", "%d/%m/%Y")),
    (re.compile(r"\b(0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b"), ("%m-%d-%Y", "%m/%d/%Y")),
    (re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(20\d{2})\b", re.IGNORECASE), ("%b %d, %Y",)),
    (re.compile(r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*,?\s+(20\d{2})\b", re.IGNORECASE), ("%d %b %Y",)),
]

# Strengthened keyword sets for totals
KEYWORDS_TOTAL_STRONG = re.compile(r"\b(amount due|balance due|total due|amount payable|amount to pay|pay now|total payable)\b", re.IGNORECASE)
KEYWORDS_TOTAL_WEAK = re.compile(r"\b(invoice total|grand total|total amount|total)(?!\s*excl)\b", re.IGNORECASE)
# Penalize lines likely to be components rather than totals
KEYWORDS_SUBTOTAL_TOKEN = re.compile(r"\b(subtotal|sub total)\b", re.IGNORECASE)
KEYWORDS_TAX_TOKEN = re.compile(r"\b(gst|vat|tax)\b", re.IGNORECASE)
KEYWORDS_SUBTOTAL = re.compile(r"\b(subtotal|tax|gst|vat|discount|shipping|delivery|freight|duty)\b", re.IGNORECASE)
# Handle inclusive/exclusive qualifiers to bias toward payable totals
KEYWORDS_INCLUSIVE = re.compile(r"\b(incl|including|incl\.?\s*gst|incl\.?\s*vat)\b", re.IGNORECASE)

KEYWORDS_DATE = re.compile(r"\b(invoice date|issue date|date)\b", re.IGNORECASE)

STOP_WORDS_VENDOR = {
    "invoice", "tax invoice", "invoice #", "invoice no", "invoice number",
    "bill to", "ship to", "sold to", "deliver to", "amount due", "total",
    "subtotal", "balance", "date", "due date", "terms", "qty", "quantity",
}


# ------------------ OCR helpers ------------------

def _pil_preprocess(img: "PILImage") -> "PILImage":
    if Image is None:
        return img
    # Convert to grayscale; try adaptive threshold when OpenCV is available
    img = ImageOps.grayscale(img)
    if cv2 is not None and np is not None:
        arr = np.array(img)
        arr_thr = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 35, 11)
        img = Image.fromarray(arr_thr)
    else:
        img = img.filter(ImageFilter.SHARPEN)
    return img


def _ocr_image(img: "PILImage") -> str:
    if pytesseract is None or Image is None:
        return ""
    try:
        pre = _pil_preprocess(img)
        text = pytesseract.image_to_string(pre)
        return text or ""
    except Exception:
        try:
            return pytesseract.image_to_string(img) or ""
        except Exception:
            return ""


# ------------------ PDF text extraction ------------------

def _extract_text_pdf(file_path: str) -> str:
    # First try embedded text via pdfplumber
    if pdfplumber is not None:
        try:
            pieces: List[str] = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    if t:
                        pieces.append(t)
            if pieces:
                return "\n".join(pieces)
        except Exception:
            pass
    # Fallback to OCR via pdf2image (if available)
    if convert_from_path is not None and Image is not None and pytesseract is not None:
        try:
            images = convert_from_path(file_path, dpi=200)
            texts = []
            for img in images:
                texts.append(_ocr_image(img))
            return "\n".join([t for t in texts if t])
        except Exception:
            return ""
    return ""


# ------------------ Amount parsing helpers ------------------

def _strip_currency_tokens(s: str) -> str:
    out = s
    for tok in CURRENCY_TOKENS:
        out = out.replace(tok, "")
    return out


def _normalize_amount_string(s: str) -> Optional[str]:
    """Normalize numbers with various thousand/decimal separators to standard 1234.56 string."""
    s = s.strip()
    s = _strip_currency_tokens(s)
    s = s.replace(" ", "")
    s = s.replace("\u00A0", "")  # non-breaking space
    # Handle parentheses negatives
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]

    # Use regex to find the last separator (comma or dot) followed by exactly two digits.
    # If found, that separator is the decimal point.
    m = re.search(r"([.,])(\d{2})$", s)
    if m:
        # A decimal separator was found. Replace it with a dot and remove all other separators.
        decimal_separator = m.group(1)
        s = s.replace(decimal_separator, '.')
        other_separator = ',' if decimal_separator == '.' else '.'
        s = s.replace(other_separator, '')
    else:
        # No clear decimal separator found, so remove all of them as they are thousand separators.
        s = s.replace(",", "").replace(".", "")

    s = _strip_currency_tokens(s)
    if negative and not s.startswith("-"):
        s = "-" + s
    # Basic sanity check
    if not re.match(r"^-?\d+(?:\.\d{1,2})?$", s):
        return None
    return s


def _find_amounts_with_positions(line: str) -> List[Tuple[Decimal, int]]:
    """Return list of (amount, position_index) for amounts in a line, right-most preferred later."""
    results: List[Tuple[Decimal, int]] = []
    for m in AMOUNT_NUMBER_RE.finditer(line):
        raw = m.group(0)
        norm = _normalize_amount_string(raw)
        if norm is None:
            continue
        try:
            amt = Decimal(norm)
            results.append((amt, m.start()))
        except (InvalidOperation, ValueError):
            continue
    return results


def _score_line_for_total(line: str) -> int:
    score = 0
    if KEYWORDS_TOTAL_STRONG.search(line):
        score += 8
    if KEYWORDS_TOTAL_WEAK.search(line):
        score += 4
    if KEYWORDS_INCLUSIVE.search(line):
        score += 2
    if KEYWORDS_SUBTOTAL.search(line):
        score -= 6
    return score


def _pick_total_amount(text: str, debug: Optional[Dict[str, Any]] = None) -> Optional[Decimal]:
    lines = text.splitlines()
    best_amt: Optional[Decimal] = None
    best_score: float = -1e9
    if debug is not None:
        debug['lines'] = []

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        score = _score_line_for_total(line)
        if debug is not None:
            entry = {
                'i': i,
                'line': line[:200],
                'score': score,
                'candidates': []
            }
        if score <= 0:
            if debug is not None:
                debug['lines'].append(entry)
            continue
        # Candidate amounts on same line; if none, try next line; then previous line
        candidates = _find_amounts_with_positions(line)
        source = 'same'
        if not candidates and i + 1 < len(lines):
            next_line = lines[i+1].strip()
            candidates = _find_amounts_with_positions(next_line)
            if candidates:
                score -= 1  # slight penalty for next-line amounts
                source = 'next'
        if not candidates and i - 1 >= 0:
            prev_line = lines[i-1].strip()
            candidates = _find_amounts_with_positions(prev_line)
            if candidates:
                score -= 2  # additional penalty for previous-line amounts
                source = 'prev'
        if not candidates:
            if debug is not None:
                debug['lines'].append(entry)
            continue
        # Prefer the right-most amount on the chosen line
        candidates.sort(key=lambda t: (t[1], t[0]))
        amt = candidates[-1][0]
        if debug is not None:
            entry['candidates'] = [{'amount': str(a), 'pos': p} for a, p in candidates]
            entry['chosen'] = {'amount': str(amt), 'source': source}
            debug['lines'].append(entry)
        # Composite score: keyword score + small preference for larger amount
        composite = score + float(amt) * 1e-6
        if composite > best_score:
            best_score = composite
            best_amt = amt
            if debug is not None:
                debug['selection'] = {'line_index': i, 'amount': str(amt), 'score': composite, 'source': source}

    if best_amt is not None:
        return best_amt

    # Fallback: choose the largest amount on lines that are not obviously subtotal/tax
    fallback_best: Optional[Decimal] = None
    for raw_line in lines:
        line = raw_line.strip()
        if KEYWORDS_SUBTOTAL.search(line):
            continue
        for amt, _pos in _find_amounts_with_positions(line):
            if fallback_best is None or amt > fallback_best:
                fallback_best = amt
    if debug is not None and fallback_best is not None:
        debug['selection'] = {'fallback': True, 'amount': str(fallback_best)}
    return fallback_best


# ------------------ Date parsing helpers ------------------

def _first_date_in_text(text: str) -> Optional[date]:
    for rx, fmts in DATE_PATTERNS:
        for m in rx.finditer(text):
            s = m.group(0)
            for fmt_try in fmts:
                try:
                    return datetime.strptime(s, fmt_try).date()
                except Exception:
                    continue
    return None


def _parse_date(text: str) -> Optional[date]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if KEYWORDS_DATE.search(line):
            segment = "\n".join(lines[i:i+3])
            d = _first_date_in_text(segment)
            if d:
                return d
    return _first_date_in_text(text)


# ------------------ Vendor parsing helpers ------------------

def _normalize_vendor(name: Optional[str]) -> Optional[str]:
    if not name:
        return name
    n = re.sub(r"\s+", " ", name).strip()
    if not n:
        return n
    parts = []
    for token in n.split(" "):
        if token.isupper() and len(token) <= 4:
            parts.append(token)
        else:
            parts.append(token.title())
    return " ".join(parts)[:255]


def _parse_vendor(text: str) -> Optional[str]:
    lines = [ln.strip() for ln in text.splitlines()[:25]]
    lines = [ln for ln in lines if ln]
    def is_bad(ln: str) -> bool:
        low = ln.lower()
        if any(sw in low for sw in STOP_WORDS_VENDOR):
            return True
        digits = sum(ch.isdigit() for ch in ln)
        if len(ln) > 0 and digits / len(ln) > 0.5:
            return True
        return False
    candidates = [ln for ln in lines if not is_bad(ln)]
    if candidates:
        return _normalize_vendor(candidates[0])
    return None


# ------------------ Subtotal/Tax and Line Items ------------------

def _pick_best_amount(
    lines: List[str],
    score_func: Any,
    avoid_keywords_re: Optional[Pattern] = None,
    debug_key: Optional[str] = None,
    debug: Optional[Dict[str, Any]] = None,
) -> Optional[Decimal]:
    """Generic helper to find the best candidate amount based on a scoring function."""
    best_amt: Optional[Decimal] = None
    best_score: float = -1e9
    if debug is not None and debug_key:
        debug[debug_key] = []

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        if avoid_keywords_re and avoid_keywords_re.search(line):
            continue

        score = score_func(line)
        if debug is not None and debug_key:
            entry: Dict[str, Any] = {
                'i': i,
                'line': line[:200],
                'score': score,
                'candidates': []
            }

        if score <= 0:
            if debug is not None and debug_key:
                debug[debug_key].append(entry)
            continue

        candidates = _find_amounts_with_positions(line)
        if not candidates:
            if debug is not None and debug_key:
                debug[debug_key].append(entry)
            continue

        candidates.sort(key=lambda t: (t[1], t[0]))
        amt = candidates[-1][0]
        if debug is not None and debug_key:
            entry['candidates'] = [{'amount': str(a), 'pos': p} for a, p in candidates]
            entry['chosen'] = {'amount': str(amt)}
            debug[debug_key].append(entry)

        composite = score + float(amt) * 1e-6
        if composite > best_score:
            best_score = composite
            best_amt = amt
            if debug is not None and debug_key and 'selection' not in debug:
                debug['selection'] = {}
            if debug is not None and debug_key:
                debug['selection'][debug_key] = {'line_index': i, 'amount': str(amt), 'score': composite}

    return best_amt

# Refined keyword sets for subtotal and tax
KEYWORDS_SUBTOTAL_TOKEN = re.compile(r'\b(subtotal|sub total|net amount|net total|total before tax)\b', re.IGNORECASE)
KEYWORDS_TAX_TOKEN = re.compile(r'\b(gst|vat|tax|sales tax|consumption tax)\b', re.IGNORECASE)
KEYWORDS_TOTAL_TOKEN = re.compile(r'\b(total|amount due|balance due)\b', re.IGNORECASE)

def _score_line_for_subtotal(line: str) -> int:
    score = 0
    if KEYWORDS_SUBTOTAL_TOKEN.search(line):
        score += 8
    # Penalize if it looks like a tax or total line
    if KEYWORDS_TAX_TOKEN.search(line):
        score -= 4
    if KEYWORDS_TOTAL_TOKEN.search(line):
        score -= 2
    return score

def _score_line_for_tax(line: str) -> int:
    score = 0
    if KEYWORDS_TAX_TOKEN.search(line):
        score += 8
    # Penalize if it looks like a subtotal or total line
    if KEYWORDS_SUBTOTAL_TOKEN.search(line):
        score -= 4
    if KEYWORDS_TOTAL_TOKEN.search(line):
        score -= 2
    return score

def _pick_subtotal_and_tax(text: str, debug: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    lines = text.splitlines()
    
    # Find subtotal first, avoiding lines that are clearly the final total
    subtotal = _pick_best_amount(lines, _score_line_for_subtotal, avoid_keywords_re=KEYWORDS_TOTAL_STRONG, debug_key='subtotal_debug', debug=debug)
    
    # Find tax total
    tax_total = _pick_best_amount(lines, _score_line_for_tax, debug_key='tax_debug', debug=debug)

    return subtotal, tax_total


def _looks_like_line_items_header(line: str) -> bool:
    low = line.lower()
    score = 0
    if any(k in low for k in ["description", "item", "product", "service"]):
        score += 1
    if any(k in low for k in ["qty", "quantity", "hours", "units"]):
        score += 1
    if any(k in low for k in ["unit", "rate", "price"]):
        score += 1
    if any(k in low for k in ["amount", "total"]):
        score += 1
    return score >= 3


def _is_line_items_terminator(line: str) -> bool:
    low = line.lower()
    return (
        "subtotal" in low or "total" in low or "amount due" in low or
        "balance" in low or "grand total" in low
    )


def _extract_line_items(text: str) -> List[Dict[str, Optional[str]]]:
    lines = [ln.strip() for ln in text.splitlines()]
    start_idx = None
    for i, ln in enumerate(lines[:80]):  # search header near top
        if _looks_like_line_items_header(ln):
            start_idx = i + 1
            break
    if start_idx is None:
        return []
    items: List[Dict[str, Optional[str]]] = []
    for ln in lines[start_idx:]:
        if not ln:
            continue
        if _is_line_items_terminator(ln):
            break
        # Heuristic: parse numbers in line
        nums = list(AMOUNT_NUMBER_RE.finditer(ln))
        text_only = ln
        for m in reversed(nums):
            text_only = text_only[:m.start()] + text_only[m.end():]
        description = re.sub(r"\s+", " ", text_only).strip() or None
        qty: Optional[str] = None
        unit_price: Optional[str] = None
        line_total: Optional[str] = None
        if nums:
            # Normalize numbers and map to qty/unit/total when possible
            values: List[Decimal] = []
            for m in nums:
                norm = _normalize_amount_string(m.group(0))
                if norm is None:
                    continue
                try:
                    values.append(Decimal(norm))
                except Exception:
                    continue
            if values:
                # If 3 or more values, check if qty * unit_price is close to total
                if len(values) >= 3:
                    # Check the last 3 values, as they are most likely to be qty, price, total
                    v1, v2, v3 = values[-3], values[-2], values[-1]
                    # Check if v1 * v2 is close to v3 (e.g., qty * price = total)
                    if abs(v1 * v2 - v3) < Decimal('0.05'):
                        qty = str(v1)
                        unit_price = str(v2)
                        line_total = str(v3)
                    # Check if v2 * v1 is close to v3 (e.g., price * qty = total)
                    elif abs(v2 * v1 - v3) < Decimal('0.05'):
                        qty = str(v2)
                        unit_price = str(v1)
                        line_total = str(v3)
                    else: # Fallback to original assumption
                        qty = str(values[0])
                        unit_price = str(values[-2])
                        line_total = str(values[-1])
                elif len(values) == 2:
                    # Assume the last value is the line total.
                    # The other could be quantity or unit price.
                    qty = str(values[0]) # Assume first is quantity-like
                    line_total = str(values[1])
                else:
                    line_total = str(values[0])
        # Ignore lines that have no description and no amounts
        if not description and not (qty or unit_price or line_total):
            continue
        items.append({
            'description': description,
            'quantity': qty,
            'unit_price': unit_price,
            'line_total': line_total,
        })
    return items


# ------------------ Public API ------------------

def parse_invoice_file(file_path: str):
    """
    Extract vendor, amount, and date from an invoice file (.pdf, .png, .jpg, .jpeg).
    Returns dict: {'vendor': str, 'amount': Decimal, 'date': date}
    Uses embedded text for PDFs when available; falls back to OCR when necessary.
    """
    vendor: Optional[str] = None
    amount: Optional[Decimal] = None
    inv_date: Optional[date] = None

    ext = os.path.splitext(file_path)[1].lower()

    text = ""
    if ext == ".pdf":
        text = _extract_text_pdf(file_path)
    elif ext in {".png", ".jpg", ".jpeg"} and Image is not None:
        try:
            img = Image.open(file_path)
            text = _ocr_image(img)
        except Exception:
            text = ""

    if text:
        try:
            vendor = _parse_vendor(text)
        except Exception:
            vendor = None
        try:
            amount = _pick_total_amount(text)
        except Exception:
            amount = None
        try:
            inv_date = _parse_date(text)
        except Exception:
            inv_date = None

    if vendor is None or vendor.strip() == "":
        vendor = "Unknown Vendor"
    if amount is None:
        amount = Decimal("0.00")
    if inv_date is None:
        inv_date = date.today()

    return {
        'vendor': vendor,
        'amount': amount,
        'date': inv_date,
    }


def parse_invoice_details(file_path: str) -> Dict[str, Any]:
    """Return rich details: vendor, amount, date, subtotal, tax_total, line_items."""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    if ext == ".pdf":
        text = _extract_text_pdf(file_path)
    elif ext in {".png", ".jpg", ".jpeg"} and Image is not None:
        try:
            img = Image.open(file_path)
            text = _ocr_image(img)
        except Exception:
            text = ""

    vendor = None
    amount = None
    inv_date = None
    subtotal = None
    tax_total = None
    line_items: List[Dict[str, Optional[str]]] = []

    if text:
        try:
            vendor = _parse_vendor(text)
        except Exception:
            logging.error("Error during invoice vendor parsing.", exc_info=True)
            pass
        try:
            amount = _pick_total_amount(text)
        except Exception:
            logging.error("Error during invoice amount parsing.", exc_info=True)
            pass
        try:
            inv_date = _parse_date(text)
        except Exception:
            logging.error("Error during invoice date parsing.", exc_info=True)
            pass
        try:
            subtotal, tax_total = _pick_subtotal_and_tax(text)
        except Exception:
            logging.error("Error during invoice subtotal/tax parsing.", exc_info=True)
            pass
        try:
            line_items = _extract_line_items(text)
        except Exception:
            logging.error("Error during invoice line item parsing.", exc_info=True)
            pass

    if vendor is None or vendor.strip() == "":
        vendor = "Unknown Vendor"
    if amount is None:
        amount = Decimal("0.00")
    if inv_date is None:
        inv_date = date.today()

    return {
        'vendor': vendor,
        'amount': amount,
        'date': inv_date,
        'subtotal': subtotal,
        'tax_total': tax_total,
        'line_items': line_items,
    }


def parse_invoice_file_debug(file_path: str) -> Dict[str, Any]:
    """
    Debug variant: returns parsed fields plus diagnostics on which lines/amounts were considered.
    """
    result: Dict[str, Any] = {
        'file': file_path,
        'vendor': None,
        'amount': None,
        'date': None,
        'diagnostics': {}
    }
    ext = os.path.splitext(file_path)[1].lower()

    text = ""
    if ext == ".pdf":
        text = _extract_text_pdf(file_path)
    elif ext in {".png", ".jpg", ".jpeg"} and Image is not None:
        try:
            img = Image.open(file_path)
            text = _ocr_image(img)
        except Exception:
            text = ""

    debug: Dict[str, Any] = {}
    if text:
        try:
            result['vendor'] = _parse_vendor(text)
        except Exception:
            pass
        try:
            amt = _pick_total_amount(text, debug=debug)
            result['amount'] = str(amt) if amt is not None else None
        except Exception:
            pass
        try:
            d = _parse_date(text)
            result['date'] = d.isoformat() if d is not None else None
        except Exception:
            pass
    try:
        subtotal, tax_total = _pick_subtotal_and_tax(text, debug=debug)
        result['subtotal'] = str(subtotal) if subtotal is not None else None
        result['tax_total'] = str(tax_total) if tax_total is not None else None
    except Exception:
        pass
    result['diagnostics'] = debug
    return result
