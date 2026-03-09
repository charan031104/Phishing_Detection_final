# utils/feature_extraction.py
import re
import tldextract
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests
import math
from datetime import datetime, timezone
import os

try:
    import whois  # python-whois
except Exception:
    whois = None

SUSPICIOUS_TOKENS = [
    "login", "signin", "bank", "secure", "account", "update", "confirm", "verify",
    "ebayisapi", "webscr", "paypal", "appleid", "security", "alert", "suspended",
    "verification", "validate", "activate", "restore", "recover", "unlock",
    "amazon", "google", "facebook", "instagram", "twitter", "linkedin",
    "microsoft", "apple", "ebay", "netflix", "spotify", "dropbox"
]

def normalize_url(u: str) -> str:
    u = u.strip()
    if not re.match(r"^https?://", u):
        u = "http://" + u
    return u

def domain_and_subdomain(url):
    ext = tldextract.extract(url)
    domain = ext.domain + '.' + ext.suffix if ext.suffix else ext.domain
    subdomain = ext.subdomain
    return domain or "", subdomain or ""

def count_digits(s):
    return sum(c.isdigit() for c in s)

def count_special_chars(s):
    return sum(1 for c in s if not c.isalnum())

def _parse_whois_dates(record):
    created_ts = None
    expiry_ts = None
    now = datetime.now(timezone.utc)
    try:
        created = record.creation_date
        expires = record.expiration_date
        if isinstance(created, list):
            created = created[0]
        if isinstance(expires, list):
            expires = expires[0]
        if isinstance(created, datetime):
            created_ts = created if created.tzinfo else created.replace(tzinfo=timezone.utc)
        if isinstance(expires, datetime):
            expiry_ts = expires if expires.tzinfo else expires.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    domain_age_days = (now - created_ts).days if created_ts else -1
    days_until_expiry = (expiry_ts - now).days if expiry_ts else -1
    return domain_age_days, days_until_expiry

def _has_ip_in_domain(host: str) -> int:
    return 1 if re.match(r"^\d{1,3}(?:\.\d{1,3}){3}$", host) else 0

def _uses_shortener(host: str, path: str) -> int:
    shorteners = {
        "bit.ly","goo.gl","t.co","ow.ly","is.gd","buff.ly","tinyurl.com",
        "rebrand.ly","cutt.ly","rb.gy","s.id","adf.ly","bit.do"
    }
    return 1 if host in shorteners else 0

def get_basic_url_features(urls):
    """Return numeric feature matrix (list of feature lists) for given URL list.
    Includes length, structure, tokens, entropy, and WHOIS-based age features.
    WHOIS lookups gracefully degrade to -1 when unavailable.
    """
    feats = []
    for u in urls:
        u_norm = normalize_url(u)
        parsed = urlparse(u_norm)
        netloc = parsed.netloc.lower()
        path = parsed.path or ""
        query = parsed.query or ""

        domain, subdomain = domain_and_subdomain(u_norm)
        domain_parts = domain.split('.') if domain else []
        subdomain_parts = subdomain.split('.') if subdomain else []

        f_len = len(u_norm)
        f_len_path = len(path)
        f_has_https = 1 if parsed.scheme == "https" else 0
        f_count_dots = netloc.count('.')
        f_count_hyphens = netloc.count('-') + path.count('-')
        f_count_at = 1 if '@' in u_norm else 0
        f_digits = count_digits(u_norm)
        f_special = count_special_chars(u_norm)
        f_subdomain_parts = len([p for p in subdomain_parts if p])
        f_domain_parts = len([p for p in domain_parts if p])
        f_suspicious_tokens = sum(1 for t in SUSPICIOUS_TOKENS if t in u_norm.lower())
        f_has_non_ascii = 1 if any(ord(c) > 127 for c in u_norm) else 0
        f_ip_in_domain = _has_ip_in_domain(netloc.split(':')[0])
        f_uses_shortener = _uses_shortener(netloc.split(':')[0], path)

        # entropy-like measure for domain
        char_set = set(netloc)
        f_entropy = 0.0
        if len(netloc) > 0:
            probs = [netloc.count(ch)/len(netloc) for ch in char_set]
            f_entropy = -sum(p*math.log(p+1e-9) for p in probs)

        # WHOIS features
        f_domain_age_days = -1
        f_days_until_expiry = -1
        skip_whois = os.getenv("SKIP_WHOIS", "0").strip() in {"1","true","yes"}
        if (whois is not None) and domain and not skip_whois:
            try:
                w = whois.whois(domain)
                f_domain_age_days, f_days_until_expiry = _parse_whois_dates(w)
            except Exception:
                pass

        feats.append([
            f_len, f_len_path, f_has_https, f_count_dots, f_count_hyphens,
            f_count_at, f_digits, f_special, f_subdomain_parts, f_domain_parts,
            f_suspicious_tokens, f_entropy, f_has_non_ascii, f_ip_in_domain,
            f_uses_shortener, f_domain_age_days, f_days_until_expiry
        ])
    return feats

def fetch_page_text(url, timeout=3):
    """Optional: fetch page and return visible text. Used sparingly (may be unreliable)."""
    try:
        r = requests.get(normalize_url(url), timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.content, "lxml")
        # remove scripts/styles
        for s in soup(["script", "style", "noscript"]):
            s.decompose()
        text = " ".join(soup.stripped_strings)
        return text[:2000]  # cap length
    except Exception:
        return ""

def url_tokenize_for_vector(url):
    """Return a text-like token string from URL to feed into TF-IDF."""
    u = normalize_url(url).lower()
    # break by non-alphanumeric and dots/slashes/hyphens
    tokens = re.split(r"[^a-z0-9]", u)
    tokens = [t for t in tokens if t and len(t) > 1]
    return " ".join(tokens)


def quick_url_risk(url: str):
    """Compute a lightweight risk score (0..1) from the URL only, no network calls.
    Returns (score, reasons_list).
    """
    reasons = []
    u = normalize_url(url)
    parsed = urlparse(u)
    host = parsed.netloc.lower().split(':')[0]
    path = parsed.path or ""
    q = parsed.query or ""
    full = (host + path + q).lower()

    score = 0.0

    # Heuristics
    dot_count = host.count('.')
    hyphen_count = host.count('-') + path.count('-')
    at_present = ('@' in u)
    has_https = (parsed.scheme == 'https')
    long_url = len(u) > 80
    many_digits = count_digits(u) >= 8
    non_ascii = any(ord(c) > 127 for c in u)
    ip_in_domain = _has_ip_in_domain(host)
    shortener = _uses_shortener(host, path)
    suspicious_kw = sum(1 for t in SUSPICIOUS_TOKENS if t in full)

    if ip_in_domain:
        score += 0.35; reasons.append("IP address used as domain")
    if shortener:
        score += 0.25; reasons.append("URL shortener domain")
    if hyphen_count >= 3:
        score += 0.25; reasons.append("Many hyphens in host/path")  # Increased
    if dot_count >= 4:
        score += 0.2; reasons.append("Deeply nested subdomains")  # Increased
    if at_present:
        score += 0.2; reasons.append("Contains '@' symbol")  # Increased
    if not has_https:
        score += 0.15; reasons.append("Not using HTTPS")  # Increased
    if long_url:
        score += 0.15; reasons.append("Unusually long URL")  # Increased
    if many_digits:
        score += 0.15; reasons.append("Many digits in URL")  # Increased
    if non_ascii:
        score += 0.25; reasons.append("Contains non-ASCII (possible IDN spoof)")  # Increased
    if suspicious_kw >= 1:
        score += min(0.2 * suspicious_kw, 0.4); reasons.append("Suspicious keywords in URL")  # Increased
        # Additional penalty for multiple suspicious keywords
        if suspicious_kw >= 2:
            score += 0.1; reasons.append("Multiple suspicious keywords")
    # Check for suspicious TLD patterns
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.biz', '.info']
    if any(tld in host.lower() for tld in suspicious_tlds):
        score += 0.15; reasons.append("Suspicious top-level domain")
    # Check for brand name + suspicious word combinations
    brand_suspicious_patterns = ['paypal.*security', 'amazon.*suspended', 'apple.*verification', 
                                'google.*security', 'bank.*security', 'account.*suspended']
    if any(re.search(pattern, full, re.IGNORECASE) for pattern in brand_suspicious_patterns):
        score += 0.2; reasons.append("Brand name with suspicious keywords")
    
    # Check for typosquatting patterns (character substitutions and extra characters)
    brand_names = ['google', 'facebook', 'amazon', 'paypal', 'apple', 'microsoft', 'netflix', 
                   'twitter', 'instagram', 'linkedin', 'youtube', 'ebay', 'spotify', 'dropbox',
                   'anurag', 'github', 'stackoverflow', 'reddit', 'discord', 'zoom', 'slack']
    
    def is_typosquatting(domain, brand):
        """Check if domain is a typosquatting variant of brand name"""
        domain_clean = re.sub(r'[^a-z0-9]', '', domain.lower())
        brand_clean = brand.lower()
        
        # Skip if the exact brand name is in the domain (legitimate domain)
        if brand_clean in domain_clean and len(domain_clean) <= len(brand_clean) + 3:
            return False, ""
        
        # Direct character substitutions (e.g., goog1e, g00gle, fac3book)
        substitutions = {
            'o': '0', 'l': '1', 'e': '3', 'a': '@', 's': '5', 
            't': '7', 'i': '1', 'b': '6', 'g': '9'
        }
        
        # Check if domain matches any substitution variant
        for char, sub in substitutions.items():
            if char in brand_clean:
                variant = brand_clean.replace(char, sub)
                if variant in domain_clean and variant != brand_clean:
                    return True, f"{variant} instead of {brand}"
        
        # Check for numbers added to brand name (only if brand is close)
        if re.search(rf'{brand_clean}[0-9]', domain_clean) or re.search(rf'[0-9]{brand_clean}', domain_clean):
            return True, f"Brand name with numbers: {brand}"
        
        return False, ""
    
    def detect_generic_typosquatting(domain):
        """Detect typosquatting patterns for any domain, not just known brands"""
        domain_clean = re.sub(r'[^a-z0-9]', '', domain.lower())
        
        # Remove common prefixes that might cause false positives
        domain_without_www = domain_clean.replace('www', '')
        
        # Check for repeated characters (e.g., anuraggg, microsoftt) - but not in www
        repeated_chars = re.findall(r'(.)\1{2,}', domain_without_www)
        if repeated_chars:
            # Only flag if it's not just 'w' from www
            if repeated_chars[0] != 'w' or len(domain_without_www) > 0:
                return True, f"Repeated characters detected: {repeated_chars[0]}"
        
        # Check for suspiciously long domains with repeated patterns
        if len(domain_without_www) > 15:
            # Look for patterns like "wordwordword" or "wordwordwordword"
            for i in range(3, len(domain_without_www) // 2):
                pattern = domain_without_www[:i]
                if domain_without_www.startswith(pattern * 2) or domain_without_www.startswith(pattern * 3):
                    return True, f"Suspicious repeated pattern: {pattern}"
        
        # Check for extra characters in common patterns
        # Look for domains that are too long compared to common lengths
        if len(domain_without_www) > 20:
            return True, f"Unusually long domain: {len(domain_without_www)} characters"
        
        return False, ""
    
    # Check known brand typosquatting
    for brand in brand_names:
        is_typo, reason = is_typosquatting(host, brand)
        if is_typo:
            score += 0.4; reasons.append(f"Typosquatting detected: {reason}")
            break  # Only flag once per URL
    
    # Check generic typosquatting patterns
    is_generic_typo, generic_reason = detect_generic_typosquatting(host)
    if is_generic_typo:
        score += 0.5; reasons.append(f"Suspicious domain pattern: {generic_reason}")

    # Cap to [0,1]
    score = max(0.0, min(1.0, score))
    return score, reasons
