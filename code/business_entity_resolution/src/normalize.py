"""Normalization and blocking-key builders for business entity resolution.

Designed from the measured noise in the 2026 dataset (see CONTEXT.md / plan):
- legal suffixes differ across vendors and across countries (LLC/Inc vs Pvt/Ltd vs SARL/SAS/EURL)
- domains appear as names ("dadaorlive.com", "#mumbaieducation")
- accents and inserted-letter typos ("Gulf"/"Gúlf", "Brand"/"Bránd")
- house numbers carry prefixes/zeros ("#4292", "015741", "Unit #314", "HN 250")
- state/region written long or short ("OR"/"Oregon", "MH"/"Maharashtra", "Kerala"/"Keralam")
- files contain U+FFFD replacement chars from broken encoding
"""
import re
import unicodedata

# Legal suffixes seen in the data, including the French set that appears ONLY in test.
SUFFIXES = {
    "inc", "incorporated", "llc", "ltd", "limited", "pvt", "private", "llp",
    "corp", "corporation", "co", "company", "plc", "lp", "pc", "pllc",
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "ei", "gmbh", "bv",
}
# Words that carry no identity (aliases, boilerplate seen in real match groups).
NOISE_WORDS = {
    "the", "and", "of", "dba", "ta", "formerly", "known", "as", "aka",
    "services", "service", "center", "centre", "partners", "partner",
    "group", "holdings", "holding", "smt", "m/s", "ms", "mr", "mrs",
}
# Street/region words that are not distinctive inside an address.
ADDR_STOP = {
    "street", "st", "road", "rd", "avenue", "ave", "av", "boulevard", "blvd",
    "drive", "dr", "lane", "ln", "court", "ct", "place", "pl", "way",
    "highway", "hwy", "parkway", "pkwy", "circle", "cir", "trail", "trl",
    "terrace", "ter", "square", "sq", "alley", "aly", "floor", "fl", "unit",
    "apt", "apartment", "suite", "ste", "building", "bldg", "no", "number",
    "near", "nr", "opp", "opposite", "behind", "beside", "po", "box", "pmb",
    "hn", "hno", "plot", "flat", "gala", "ward", "block", "sector", "phase",
    "main", "cross", "nagar", "colony", "road", "marg",
    "rue", "allee", "quai", "impasse", "imp", "bis", "chemin", "route",
    "place", "cours", "passage", "residence",
}
# Region full-name -> abbreviation so "OR" and "Oregon" collide.
REGION = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "hampshire": "nh", "jersey": "nj", "mexico": "nm",
    "york": "ny", "carolina": "nc", "dakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode": "ri",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "wisconsin": "wi", "wyoming": "wy",
    "maharashtra": "mh", "karnataka": "ka", "tamilnadu": "tn", "kerala": "kl",
    "keralam": "kl", "delhi": "dl", "gujarat": "gj", "rajasthan": "rj",
    "telangana": "ts", "andhrapradesh": "ap", "uttarpradesh": "up",
    "westbengal": "wb", "madhyapradesh": "mp", "punjab": "pb", "bihar": "br",
    "haryana": "hr", "odisha": "or", "jharkhand": "jh", "assam": "as",
    "uttarakhand": "uk", "chhattisgarh": "cg", "goa": "ga",
    "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai",
    "bangalore": "bengaluru", "poona": "pune",
}

_COMBINING = dict.fromkeys(map(chr, range(0x0300, 0x0370)), None)


def clean_noise(s: str) -> str:
    """Strip emails and social handles — safe to remove, never business names."""
    s = re.sub(r'\S+@\S+\.\S+', ' ', s)   # emails
    s = re.sub(r'@\w+', ' ', s)            # @handles
    s = re.sub(r'#\w+', ' ', s)            # #hashtags
    return s


def fold(s: str) -> str:
    """Lowercase, strip accents, drop replacement chars, '&' -> 'and'."""
    s = s.replace("\ufffd", " ").replace("&", " and ")
    s = unicodedata.normalize("NFKD", s).translate(_COMBINING)
    return s.lower()


def tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", fold(s))


def name_tokens(s: str) -> list[str]:
    s = clean_noise(s)
    s = re.sub(r"(https?://|www\.)", " ", s)
    s = re.sub(r"\.(com|in|fr|net|org|co|io)\b", " ", s, flags=re.I)
    out = [t for t in tokens(s) if t not in SUFFIXES and t not in NOISE_WORDS and not t.isdigit()]
    return out


def name_squash(s: str) -> str:
    return "".join(name_tokens(s))


def house_numbers(s: str) -> set[str]:
    """All integer tokens, leading zeros removed. '015741' and '15741' collide."""
    return {m.lstrip("0") or "0" for m in re.findall(r"\d+", s)}


def addr_tokens(s: str) -> list[str]:
    out = []
    for t in tokens(s):
        t = REGION.get(t, t)
        if t in ADDR_STOP or t.isdigit() or len(t) < 2:
            continue
        out.append(t)
    return out


def city_token(s: str) -> str:
    """Last distinctive address token is usually the city/locality."""
    ts = addr_tokens(s)
    return ts[-1] if ts else ""


def street_token(s: str) -> str:
    """First long alpha token that is not the city: the street name."""
    ts = addr_tokens(s)
    for t in ts:
        if len(t) >= 4:
            return t
    return ts[0] if ts else ""
