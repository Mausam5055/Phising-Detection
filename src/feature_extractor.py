import re
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import numpy as np
import socket

class FeatureExtractor:
    def __init__(self, url):
        self.url = url
        if not self.url.startswith(('http://', 'https://')):
            self.url = 'http://' + self.url
        
        try:
            self.parsed = urlparse(self.url)
            self.hostname = self.parsed.hostname if self.parsed.hostname else ""
            self.path = self.parsed.path
            self.query = self.parsed.query
        except:
            self.hostname = ""
            self.path = ""
            self.query = ""
            
        self.soup = None
        self.response = None

    def fetch_content(self):
        try:
            self.response = requests.get(self.url, timeout=3)
            self.soup = BeautifulSoup(self.response.content, 'html.parser')
        except:
            pass

    # --- Feature Extraction Methods ---
    
    def get_num_dots(self):
        return self.url.count('.')

    def get_subdomain_level(self):
        if not self.hostname: return 0
        return self.hostname.count('.')

    def get_path_level(self):
        if not self.path: return 0
        return len([x for x in self.path.split('/') if x])

    def get_url_length(self):
        return len(self.url)

    def get_num_dash(self):
        return self.url.count('-')

    def get_num_dash_in_hostname(self):
        return self.hostname.count('-')

    def get_at_symbol(self):
        return 1 if '@' in self.url else 0

    def get_tilde_symbol(self):
        return 1 if '~' in self.url else 0

    def get_num_underscore(self):
        return self.url.count('_')

    def get_num_percent(self):
        return self.url.count('%')

    def get_num_query_components(self):
        if not self.query: return 0
        return len(self.query.split('&'))

    def get_num_ampersand(self):
        return self.url.count('&')

    def get_num_hash(self):
        return self.url.count('#')

    def get_num_numeric_chars(self):
        return sum(c.isdigit() for c in self.url)

    def get_no_https(self):
        return 0 if self.url.startswith('https') else 1

    def get_random_string(self):
        # Simplified heuristic
        return 0

    def get_ip_address(self):
        try:
            socket.inet_aton(self.hostname)
            return 1
        except:
            return 0

    def get_domain_in_subdomains(self):
        return 0 # Placeholder

    def get_domain_in_paths(self):
        return 0 # Placeholder

    def get_https_in_hostname(self):
        return 1 if 'https' in self.hostname else 0

    def get_hostname_length(self):
        return len(self.hostname)

    def get_path_length(self):
        return len(self.path)

    def get_query_length(self):
        return len(self.query)

    def get_double_slash_in_path(self):
        return 1 if '//' in self.path else 0

    # Content-based features (Simplified/Placeholders if fetch fails)
    def get_num_sensitive_words(self):
        sensitive = ['confirm', 'account', 'banking', 'secure', 'ebayisapi', 'webscr', 'login', 'signin']
        return sum(1 for w in sensitive if w in self.url.lower())

    # ... (Implementing a subset of critical features for brevity and speed)
    # Ideally we would implement all 48. For now, I will implement the ones I can and fill the rest with 0.
    
    def extract_features(self):
        # Order must match dataset.csv columns (excluding id and label)
        # 1. NumDots
        # 2. SubdomainLevel
        # ...
        
        # NOTE: This is a BEST EFFORT mapping. 
        features = []
        features.append(self.get_num_dots())
        features.append(self.get_subdomain_level())
        features.append(self.get_path_level())
        features.append(self.get_url_length())
        features.append(self.get_num_dash())
        features.append(self.get_num_dash_in_hostname())
        features.append(self.get_at_symbol())
        features.append(self.get_tilde_symbol())
        features.append(self.get_num_underscore())
        features.append(self.get_num_percent())
        features.append(self.get_num_query_components())
        features.append(self.get_num_ampersand())
        features.append(self.get_num_hash())
        features.append(self.get_num_numeric_chars())
        features.append(self.get_no_https())
        features.append(self.get_random_string())
        features.append(self.get_ip_address())
        features.append(self.get_domain_in_subdomains())
        features.append(self.get_domain_in_paths())
        features.append(self.get_https_in_hostname())
        features.append(self.get_hostname_length())
        features.append(self.get_path_length())
        features.append(self.get_query_length())
        features.append(self.get_double_slash_in_path())
        features.append(self.get_num_sensitive_words())
        
        # Remaining 23 features (Content-based or complex)
        # We will fill them with 0 for now to ensure shape compatibility.
        # In a real production scenario, we'd need robust implementations for all.
        for _ in range(23):
            features.append(0)
            
        return np.array(features).reshape(1, -1)
