from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_caching import Cache
import requests
import json
import logging
import os
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Configure caching
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300
cache = Cache(app)

# Network environment configuration
NETWORK_ENV = "external"  # Change to "external" for external networks

class KeywordGenerator:
    def __init__(self):
        self.termdat_base_url = "https://register.ld.admin.ch/termdat/"
        self.gemet_base_url = "http://www.eionet.europa.eu/gemet/"
        self.wikidata_base_url = "https://www.wikidata.org/w/api.php"
        # Set SSL verification based on network environment
        self.verify_ssl = NETWORK_ENV != "internal"

        # Create a persistent session for outgoing HTTP requests to improve connection reuse
        # and set a descriptive User-Agent so Wikidata operators can contact us if needed.
        self.session = requests.Session()
        contact = os.environ.get('ADMIN_CONTACT', 'devteam@example.com')
        ua = f"keyword-generator/1.0 (mailto:{contact})"
        # Set headers used for Wikidata requests (also helpful for other endpoints)
        self.session.headers.update({'User-Agent': ua, 'From': contact})

        # Configure retries with exponential backoff for transient errors and 429 throttling
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 502, 503, 504], allowed_methods=["GET", "POST"])
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
        
        # OpenAI API key for keyword extraction (optional)
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')

    def _extract_keywords_from_text(self, text, max_keywords=5):
        """Extract relevant keywords from a longer text description.
        Uses OpenAI if available, otherwise falls back to simple NLP techniques.
        
        Args:
            text: The input text (could be single word or multiple sentences)
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            list: List of extracted keywords
        """
        # If text is short (likely a single keyword), return it as-is
        if len(text.split()) <= 3:
            return [text.strip()]
        
        logging.info(f"Extracting keywords from text: '{text[:100]}...'")
        
        # Try OpenAI first if API key is available
        if self.openai_api_key:
            try:
                keywords = self._extract_keywords_with_openai(text, max_keywords)
                if keywords:
                    logging.info(f"Extracted keywords via OpenAI: {keywords}")
                    return keywords
            except Exception as e:
                logging.warning(f"OpenAI keyword extraction failed: {e}, falling back to NLP")
        
        # Fallback to simple NLP-based extraction
        keywords = self._extract_keywords_with_nlp(text, max_keywords)
        logging.info(f"Extracted keywords via NLP: {keywords}")
        return keywords
    
    def _extract_keywords_with_openai(self, text, max_keywords=5):
        """Extract keywords using OpenAI GPT model"""
        try:
            import openai
            openai.api_key = self.openai_api_key
            
            prompt = f"""Extract the {max_keywords} most important keywords or key phrases from the following text. 
The keywords should be relevant for searching in controlled vocabularies like TERMDAT, GEMET, or Wikidata.
Return ONLY the keywords, one per line, without numbering or explanation.
Focus on domain-specific terms, concepts, and important nouns.

Text: {text}

Keywords:"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a keyword extraction expert for Swiss government data catalogs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            keywords = [kw.strip() for kw in content.split('\n') if kw.strip()]
            
            # Filter out very short keywords and limit to max_keywords
            keywords = [kw for kw in keywords if len(kw) > 2][:max_keywords]
            
            return keywords if keywords else None
            
        except ImportError:
            logging.info("OpenAI library not installed")
            return None
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            return None
    
    def _extract_keywords_with_nlp(self, text, max_keywords=5):
        """Extract keywords using simple NLP techniques (frequency analysis, stopwords, etc.)"""
        # German stopwords (common words to ignore)
        german_stopwords = {
            'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einer', 'eines', 'einem', 'einen',
            'und', 'oder', 'aber', 'als', 'bei', 'bis', 'durch', 'für', 'gegen', 'in', 'mit', 'nach',
            'über', 'um', 'von', 'vor', 'zu', 'zum', 'zur', 'im', 'am', 'an', 'auf', 'aus', 'bei',
            'ist', 'sind', 'war', 'waren', 'wird', 'werden', 'wurde', 'wurden', 'hat', 'haben',
            'nicht', 'nur', 'noch', 'auch', 'wenn', 'wie', 'was', 'wo', 'wer', 'alle', 'diese',
            'dieser', 'dieses', 'jede', 'jeder', 'jedes', 'sein', 'seine', 'ihr', 'ihre',
            'sich', 'sie', 'er', 'es', 'wir', 'uns', 'euch', 'ihnen', 'mein', 'dein',
            'kann', 'könnte', 'muss', 'sollte', 'möchte', 'soll', 'sehr', 'mehr', 'viel'
        }
        
        # Clean and tokenize text
        text_lower = text.lower()
        # Remove punctuation and split into words
        words = re.findall(r'\b[a-zäöüß]+\b', text_lower)
        
        # Filter out stopwords and short words
        meaningful_words = [
            word for word in words 
            if word not in german_stopwords and len(word) > 3
        ]
        
        # Count word frequencies
        from collections import Counter
        word_freq = Counter(meaningful_words)
        
        # Get most common words
        top_words = [word for word, count in word_freq.most_common(max_keywords * 2)]
        
        # Also look for multi-word phrases (bigrams, trigrams)
        phrases = self._extract_phrases(text, german_stopwords)
        
        # Combine single words and phrases, prioritize phrases
        keywords = phrases[:max_keywords // 2] + top_words[:max_keywords]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                unique_keywords.append(kw)
        
        return unique_keywords[:max_keywords]
    
    def _extract_phrases(self, text, stopwords):
        """Extract meaningful multi-word phrases from text"""
        phrases = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            # Look for capitalized phrases (likely proper nouns or important concepts)
            capitalized = re.findall(r'\b[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)+\b', sentence)
            phrases.extend(capitalized)
            
            # Look for quoted phrases
            quoted = re.findall(r'"([^"]+)"', sentence)
            phrases.extend(quoted)
        
        # Filter out very long phrases and those with stopwords
        filtered_phrases = []
        for phrase in phrases:
            words = phrase.lower().split()
            if 2 <= len(words) <= 4:  # 2-4 word phrases
                # Check if not mostly stopwords
                meaningful = [w for w in words if w not in stopwords and len(w) > 2]
                if len(meaningful) >= len(words) / 2:
                    filtered_phrases.append(phrase)
        
        return filtered_phrases[:5]  # Return top 5 phrases

    def _discover_synonyms_from_wikidata(self, query, limit=3):
        """Discover synonyms and related terms using Wikidata's linked data relationships"""
        synonyms = set()

        cache_key = f"wd_syn:{query.lower()}"
        try:
            cached = cache.get(cache_key)
            if cached:
                logging.debug(f"Returning cached Wikidata synonyms for '{query}'")
                return cached[:limit]
        except Exception:
            pass

        try:
            # First, find the main entity for the query
            search_params = {
                'action': 'wbsearchentities',
                'search': query,
                'language': 'de',  # Start with German
                'format': 'json',
                'limit': 1  # Just get the best match
            }

            response = self.session.get(self.wikidata_base_url, params=search_params, timeout=10, verify=self.verify_ssl)
            if response.status_code == 200:
                data = response.json()
                if 'search' in data and data['search']:
                    entity_id = data['search'][0].get('id')
                    if entity_id:
                        # Get related entities using Wikidata properties
                        related_terms = self._get_wikidata_related_terms(entity_id)
                        synonyms.update(related_terms)

        except Exception as e:
            logging.debug(f"Error discovering Wikidata synonyms for '{query}': {e}")

        result = list(synonyms)[:limit]
        try:
            cache.set(cache_key, result, timeout=3600)
        except Exception:
            pass
        return result

    def _get_wikidata_related_terms(self, entity_id):
        """Get related terms from Wikidata using semantic properties"""
        related_terms = set()

        try:
            # Query for the entity and its claims (relationships)
            query_params = {
                'action': 'wbgetentities',
                'ids': entity_id,
                'props': 'claims|labels|aliases',
                'languages': 'de|en',
                'format': 'json'
            }

            response = self.session.get(self.wikidata_base_url, params=query_params, timeout=10, verify=self.verify_ssl)
            if response.status_code == 200:
                data = response.json()
                if 'entities' in data and entity_id in data['entities']:
                    entity = data['entities'][entity_id]

                    # Extract aliases (alternative names)
                    if 'aliases' in entity:
                        for lang in ['de', 'en']:
                            if lang in entity['aliases']:
                                for alias in entity['aliases'][lang]:
                                    if 'value' in alias:
                                        related_terms.add(alias['value'])

                    # Look for specific semantic relationships in claims
                    if 'claims' in entity:
                        claims = entity['claims']

                        # Common properties that indicate related concepts:
                        # P31: instance of, P279: subclass of, P361: part of, P527: has part
                        # P1889: different from, P460: said to be the same as
                        semantic_properties = ['P31', 'P279', 'P361', 'P527', 'P1889', 'P460']

                        for prop in semantic_properties:
                            if prop in claims:
                                for claim in claims[prop][:2]:  # Limit to 2 per property
                                    if 'mainsnak' in claim and 'datavalue' in claim['mainsnak']:
                                        if 'value' in claim['mainsnak']['datavalue']:
                                            target_id = claim['mainsnak']['datavalue']['value'].get('id')
                                            if target_id:
                                                # Get label for the related entity
                                                related_label = self._get_wikidata_entity_label(target_id)
                                                if related_label:
                                                    related_terms.add(related_label)

        except Exception as e:
            logging.debug(f"Error getting Wikidata related terms for {entity_id}: {e}")

        return related_terms

    def _get_wikidata_entity_label(self, entity_id):
        """Get the German label for a Wikidata entity"""
        # Check cache first
        cache_key = f"wd_label:{entity_id}"
        try:
            cached = cache.get(cache_key)
            if cached:
                return cached
        except Exception:
            pass

        try:
            params = {
                'action': 'wbgetentities',
                'ids': entity_id,
                'props': 'labels',
                'languages': 'de|en',
                'format': 'json'
            }

            response = self.session.get(self.wikidata_base_url, params=params, timeout=5, verify=self.verify_ssl)
            if response.status_code == 200:
                data = response.json()
                if 'entities' in data and entity_id in data['entities']:
                    entity = data['entities'][entity_id]
                    if 'labels' in entity:
                        # Prefer German, fallback to English
                        for lang in ['de', 'en']:
                            if lang in entity['labels'] and 'value' in entity['labels'][lang]:
                                value = entity['labels'][lang]['value']
                                try:
                                    cache.set(cache_key, value, timeout=24*3600)
                                except Exception:
                                    pass
                                return value
        except Exception as e:
            logging.debug(f"Error getting label for entity {entity_id}: {e}")

        return None

    def _discover_synonyms_from_termdat(self, query, limit=2):
        """Discover related terms by analyzing TERMDAT search results for semantic similarity"""
        related_terms = set()

        try:
            # Do a broader search in TERMDAT to find related concepts
            url = "https://www.termdat.bk.admin.ch/api/Search/Search"
            params = {
                'pageindex': 1,
                'pagesize': 20,  # Get more results for analysis
                'phrase': query,
                'offices': 1,
                'officesPriority': 'true',
                'status': 1,
                'statusPriority': 'true',
                'fields.term': 'true',
                'fields.name': 'true',
                'fields.abbreviation': 'true',
                'fields.phraseology': 'true',
                'fields.definition': 'true',  # Include definitions for semantic analysis
                'fields.note': 'false',
                'fields.context': 'false',
                'fields.source': 'false',
                'fields.metadata': 'true',
                'fields.country': 'false',
                'fields.comment': 'false'
            }

            response = requests.get(url, params=params, timeout=10, verify=self.verify_ssl)
            if response.status_code == 200:
                data = response.json()
                if 'searchEntries' in data and isinstance(data['searchEntries'], list):
                    query_lower = query.lower()

                    for entry in data['searchEntries']:
                        if 'terms' in entry and isinstance(entry['terms'], list):
                            for term_obj in entry['terms']:
                                term_text = term_obj.get('terminus', '').strip()
                                if term_text and term_text.lower() != query_lower:
                                    # Check if this term is semantically related
                                    if self._is_semantically_related(query, term_text, entry):
                                        related_terms.add(term_text)

        except Exception as e:
            logging.debug(f"Error discovering TERMDAT synonyms for '{query}': {e}")

        return list(related_terms)[:limit]

    def _is_semantically_related(self, original_query, candidate_term, termdat_entry):
        """Determine if a candidate term is semantically related to the original query"""
        # Simple heuristics for semantic relatedness:

        # 1. Check if they appear in the same collection (domain similarity)
        collection_name = ''
        if 'collection' in termdat_entry and 'name' in termdat_entry['collection']:
            collection_name = termdat_entry['collection']['name'].lower()

        # 2. Check for shared word stems (basic linguistic similarity)
        original_words = set(original_query.lower().split())
        candidate_words = set(candidate_term.lower().split())

        # If they share at least one significant word (>2 chars), they might be related
        shared_words = original_words & candidate_words
        significant_shared = any(len(word) > 2 for word in shared_words)

        # 3. Check for common prefixes/suffixes (compound words in German)
        has_common_affix = False
        for orig_word in original_words:
            for cand_word in candidate_words:
                if len(orig_word) > 3 and len(cand_word) > 3:
                    # Check for common prefix (first 4+ chars)
                    if orig_word[:4] == cand_word[:4]:
                        has_common_affix = True
                    # Check for common suffix (last 4+ chars)
                    elif orig_word[-4:] == cand_word[-4:]:
                        has_common_affix = True

        # 4. Length similarity (avoid very short or very long terms)
        length_ratio = min(len(original_query), len(candidate_term)) / max(len(original_query), len(candidate_term))
        reasonable_length = length_ratio > 0.3

        # Term is related if it meets multiple criteria
        return (significant_shared or has_common_affix) and reasonable_length

    def _termdat_candidate_exists(self, candidate):
        """Lightweight TERMDAT check: return True if TERMDAT contains the candidate term (any language).
        This avoids invoking the heavier search_termdat which may loop back to synonyms.
        """
        try:
            url = "https://www.termdat.bk.admin.ch/api/Search/Search"
            params = {
                'pageindex': 1,
                'pagesize': 5,
                'phrase': candidate,
                'offices': 1,
                'status': 1,
                'fields.term': 'true',
                'fields.definition': 'false'
            }
            resp = requests.get(url, params=params, timeout=8, verify=self.verify_ssl)
            if resp.status_code == 200:
                data = resp.json()
                if 'searchEntries' in data and isinstance(data['searchEntries'], list):
                    for entry in data['searchEntries']:
                        if 'terms' in entry and isinstance(entry['terms'], list):
                            for term_obj in entry['terms']:
                                term_text = term_obj.get('terminus', '')
                                if term_text and term_text.strip().lower() == candidate.strip().lower():
                                    return True
        except Exception as e:
            logging.debug(f"TERMDAT existence check failed for '{candidate}': {e}")
        return False

    def _get_synonym_queries(self, query):
        """Get synonym queries using an LLM (OpenAI) when available, then validate candidates
        against Wikidata and TERMDAT. Falls back to linked-data discovery if needed.
        Implements a simple cache to avoid repeated identical network/LLM calls.
        """
        query = query.strip()

        # Simple cache: avoid repeated identical synonym lookups for the same query
        cache_key = f"synonyms:{query.lower()}"
        try:
            cached = cache.get(cache_key)
            if cached:
                logging.debug(f"Returning cached synonyms for '{query}'")
                return cached
        except Exception as e:
            logging.debug(f"Synonym cache read failed for '{query}': {e}")

        candidates = []

        # Try OpenAI if API key is available
        openai_key = os.environ.get('OPENAI_API_KEY')
        if openai_key:
            try:
                prompt = (
                    f"Provide up to 5 German synonyms or closely related short phrases for the term: '{query}'. "
                    "Return the result as a JSON array of strings only. Prefer single-word synonyms when possible."
                )

                headers = {
                    'Authorization': f'Bearer {openai_key}',
                    'Content-Type': 'application/json'
                }
                payload = {
                    'model': 'gpt-3.5-turbo',
                    'messages': [
                        {'role': 'system', 'content': 'You are a helpful assistant that outputs a JSON array of strings.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    'max_tokens': 200,
                    'temperature': 0.2,
                    'n': 1
                }

                resp = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload, timeout=15)
                if resp.ok:
                    body = resp.json()
                    text = ''
                    try:
                        text = body['choices'][0]['message']['content']
                    except Exception:
                        text = body.get('choices', [{}])[0].get('text', '')

                    # Try parsing JSON from assistant
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, list):
                            candidates = [s for s in parsed if isinstance(s, str)]
                    except Exception:
                        # Fallback: split by common separators and clean
                        parts = re.split('[,;\n\r]', text)
                        candidates = [p.strip().strip('"\'') for p in parts if p.strip()]

            except Exception as e:
                logging.debug(f"OpenAI synonym generation failed for '{query}': {e}")

        # Validate candidates using Wikidata and TERMDAT; keep those that are found in at least one
        validated = []
        for cand in candidates[:5]:  # only check up to 5 candidates from LLM to save calls
            try:
                found_in_wikidata = False
                found_in_termdat = False

                # Check Wikidata via existing search_wikidata (safe to call)
                try:
                    wd = self.search_wikidata(cand)
                    if wd and len(wd) > 0:
                        found_in_wikidata = True
                except Exception:
                    found_in_wikidata = False

                # Check TERMDAT via lightweight existence check
                try:
                    if self._termdat_candidate_exists(cand):
                        found_in_termdat = True
                except Exception:
                    found_in_termdat = False

                if found_in_wikidata or found_in_termdat:
                    validated.append(cand)
            except Exception as e:
                logging.debug(f"Error validating candidate '{cand}': {e}")

        # If no validated candidates from OpenAI, fall back to linked data discovery
        if not validated:
            logging.debug(f"No validated OpenAI candidates for '{query}', falling back to linked-data discovery")
            wikidata_synonyms = self._discover_synonyms_from_wikidata(query, limit=3)
            termdat_synonyms = self._discover_synonyms_from_termdat(query, limit=3)
            combined = list(dict.fromkeys(wikidata_synonyms + termdat_synonyms))
            result = combined[:5]

            # Cache the fallback result for a short period to avoid repeated discovery calls
            try:
                cache.set(cache_key, result, timeout=3600)  # 1 hour
            except Exception as e:
                logging.debug(f"Synonym cache write failed for '{query}': {e}")

            return result

        # Return up to 5 validated synonyms and cache them
        logging.debug(f"OpenAI validated synonyms for '{query}': {validated}")
        result = validated[:5]
        try:
            cache.set(cache_key, result, timeout=3600)  # 1 hour cache
        except Exception as e:
            logging.debug(f"Synonym cache write failed for '{query}': {e}")
        return result

    def search_termdat(self, query, query_lang='de', include_synonyms=True, max_results=10):
        """Search TERMDAT for keywords using the official API with multilingual support and synonym expansion
        This version limits pagesize and stops early when enough results have been gathered to reduce API calls.
        """
        results = []
        processed_entry_ids = set()  # Track processed entries to avoid duplicates

        # Lowercase the query language
        query_lang = query_lang.lower()

        # Prepare search queries: original + synonyms (limit synonyms to 2 to reduce calls)
        search_queries = [query]
        if include_synonyms:
            synonyms = self._get_synonym_queries(query)
            # Keep at most 2 synonyms to reduce number of API calls
            search_queries.extend(synonyms[:2])
            logging.debug(f"TERMDAT searching for: {query} + synonyms: {synonyms[:2]}")

        try:
            # Map language IDs to codes
            # 2=DE, 3=EN, 6=FR, 7=IT, 8=RM
            # Note: Excluding Spanish (4=ES) as I14Y platform cannot handle it
            language_map = {2: 'de', 6: 'fr', 7: 'it', 3: 'en', 8: 'rm'}

            # Search for each query (original + synonyms)
            for search_query in search_queries:
                # TERMDAT API search endpoint - use frontend-style parameters to get all collections
                url = "https://www.termdat.bk.admin.ch/api/Search/Search"

                # Build parameters matching the frontend approach for comprehensive results
                # Include language IDs to match frontend behavior and access geopolitical collections
                base_params = {
                    'pageindex': 1,
                    'pagesize': 25,  # Use frontend pagesize to get more comprehensive results
                    'phrase': search_query,
                    'offices': 1,
                    'officesPriority': 'true',
                    'status': 1,
                    'statusPriority': 'true',
                    'fields.term': 'true',
                    'fields.name': 'true',
                    'fields.abbreviation': 'true',
                    'fields.phraseology': 'true',
                    'fields.definition': 'false',
                    'fields.note': 'false',
                    'fields.context': 'false',
                    'fields.source': 'false',
                    'fields.metadata': 'true',
                    'fields.country': 'false',
                    'fields.comment': 'false'
                }
                
                # Build URL with multiple language ID parameters (like frontend does)
                # Language IDs: 2=DE, 3=EN, 6=FR, 7=IT, 8=RM
                language_ids = [2, 6, 7, 8, 3]  # DE, FR, IT, RM, EN (exclude Spanish=4)
                params_list = []
                for key, value in base_params.items():
                    params_list.append(f"{key}={value}")
                
                # Add language parameters as the frontend does (multiple params with same name)
                for lang_id in language_ids:
                    params_list.append(f"sourceLanguageIds={lang_id}")
                    params_list.append(f"targetLanguageIds={lang_id}")
                
                full_url = url + "?" + "&".join(params_list)

                logging.debug(f"TERMDAT Search Request for '{search_query}': {full_url}")

                response = requests.get(full_url, timeout=8, verify=self.verify_ssl)

                if response.status_code == 200:
                    try:
                        data = response.json()

                        # Process search entries from the response
                        if 'searchEntries' in data and isinstance(data['searchEntries'], list):
                            for entry in data['searchEntries']:
                                # Stop early if we've gathered enough results
                                if len(results) >= max_results:
                                    logging.debug("Reached max_results in TERMDAT search, stopping further processing")
                                    break

                                entry_id = entry.get('id')
                                if not entry_id or entry_id in processed_entry_ids:
                                    continue

                                # Only include validated entries (status == 1)
                                entry_status = entry.get('status')
                                if entry_status != 1 and entry_status != "1":
                                    logging.debug(f"Skipping TERMDAT entry {entry_id} - not validated (status: {entry_status})")
                                    continue

                                processed_entry_ids.add(entry_id)

                                # Extract multilingual terms from this entry
                                multilingual_terms = {}

                                if 'terms' in entry and isinstance(entry['terms'], list):
                                    for term_obj in entry['terms']:
                                        lang_id = term_obj.get('languageId')
                                        if lang_id in language_map:
                                            lang_code = language_map[lang_id]

                                            # Get term from terminus field, fallback to name or abbreviation
                                            term_text = term_obj.get('terminus', '').strip()
                                            if not term_text:
                                                term_text = term_obj.get('name', '').strip()
                                            if not term_text:
                                                term_text = term_obj.get('abbreviation', '').strip()

                                            if term_text:
                                                # For duplicate language entries, keep the first non-empty one
                                                if lang_code not in multilingual_terms:
                                                    multilingual_terms[lang_code] = term_text

                                # Only include entries that have meaningful multilingual coverage
                                # Require at least German and one other language from our core set (fr, it, en)
                                core_languages = {'de', 'fr', 'it', 'en'}
                                available_core_langs = set(multilingual_terms.keys()) & core_languages

                                if 'de' in available_core_langs and len(available_core_langs) >= 2:
                                    # Get collection info
                                    collection_name = ''
                                    if 'collection' in entry and 'name' in entry['collection']:
                                        collection_name = entry['collection']['name']

                                    # Build the TERMDAT URI
                                    termdat_uri = f"https://register.ld.admin.ch/termdat/{entry_id}"

                                    # Get a suitable description
                                    description = f"TERMDAT entry for '{query}'"
                                    if collection_name:
                                        description = f"TERMDAT: {collection_name}"

                                    # Track which languages are available
                                    available_languages_list = list(multilingual_terms.keys())

                                    # Mark if this was found via synonym search
                                    is_synonym_result = search_query != query

                                    logging.debug(f"Adding TERMDAT result for entry {entry_id} with languages {available_languages_list}: {multilingual_terms}")

                                    results.append({
                                        'source': 'TERMDAT',
                                        'multilingual_label': multilingual_terms,
                                        'uri': termdat_uri,
                                        'description': description,
                                        'entry_id': entry_id,
                                        'query_lang': query_lang,
                                        'available_languages': available_languages_list,
                                        'is_synonym_result': is_synonym_result,
                                        'found_via_query': search_query
                                    })
                                else:
                                    logging.debug(f"Skipping TERMDAT entry {entry_id} - insufficient core language coverage. Available: {available_core_langs}")

                    except Exception as e:
                        logging.error(f"Error parsing TERMDAT response for '{search_query}': {e}")
                else:
                    logging.error(f"TERMDAT API returned non-200 status for '{search_query}': {response.status_code}")

        except Exception as e:
            logging.error(f"Error searching TERMDAT: {e}")

        # Sort results: exact matches first, then synonym matches, then by number of available languages
        def sort_key(result):
            # Check if any term in any language is an exact match (case-insensitive)
            query_lower = query.lower()
            multilingual_terms = result.get('multilingual_label', {})

            # Check for exact match in any language
            has_exact_match = any(
                term.lower() == query_lower 
                for term in multilingual_terms.values()
            )

            # Check if this was found via synonym
            is_synonym_result = result.get('is_synonym_result', False)

            # Return tuple: (exact_match_priority, synonym_priority, language_count)
            # exact_match_priority: 0 for exact matches (highest priority), 1 for others
            # synonym_priority: 0 for original query results, 1 for synonym results
            # language_count: negative so more languages = higher priority
            return (
                0 if has_exact_match else 1,
                1 if is_synonym_result else 0,
                -len(result['available_languages'])
            )

        results.sort(key=sort_key)

        logging.debug(f"TERMDAT total results: {len(results)} (original + synonyms)")
        return results

    def search_gemet(self, query):
        """Search GEMET for keywords with multilingual support, optimized for geo-related contexts
        Uses the official GEMET REST API to find concepts and retrieve multilingual labels
        """
        results = []
        
        # Cache key based on the query to avoid repeated identical searches
        cache_key = f"gemet_results:{query.lower()}"
        try:
            cached = cache.get(cache_key)
            if cached:
                logging.debug(f"Returning cached GEMET results for '{query}'")
                return cached
        except Exception as e:
            logging.debug(f"GEMET cache read failed for '{query}': {e}")
        
        found_concepts = []  # List to store all found concepts
        
        try:
            logging.info(f"Starting GEMET search for '{query}'")
            
            # First, search for concepts matching the query using keyword search
            search_url = "https://www.eionet.europa.eu/gemet/getConceptsMatchingKeyword"
            
            # Try different search modes and languages for better coverage
            search_modes = [3, 2, 4]  # Contains, Prefix, Auto search
            
            # Try a few search strategies to maximize results
            search_strategies = [
                # Strategy 1: Original query in English with contains search
                {'keyword': query.lower(), 'language': 'en', 'search_mode': 3},
                # Strategy 2: Original query in German with contains search
                {'keyword': query.lower(), 'language': 'de', 'search_mode': 3},
                # Strategy 3: Try auto search mode with original query
                {'keyword': query.lower(), 'language': 'en', 'search_mode': 4},
                # Strategy 4: Try auto search mode with original query in German
                {'keyword': query.lower(), 'language': 'de', 'search_mode': 4},
            ]
            
            # Check if query is geographical in nature
            geo_terms = {
                # Mountains and alpine terms
                'alp': ['mountain', 'berg', 'gebirge', 'alps', 'alpine'],
                'berg': ['mountain', 'hill', 'gebirge', 'massif'],
                'gebirge': ['mountain range', 'mountains', 'mountain', 'alps'],
                'mountain': ['berg', 'gebirge', 'massif', 'alps'],
                # Water bodies
                'fluss': ['river', 'stream', 'waterway', 'watercourse'],
                'river': ['fluss', 'stream', 'waterway', 'watercourse'],
                'see': ['lake', 'water body', 'reservoir'],
                'lake': ['see', 'water body', 'reservoir'],
                # Regions and places
                'aargau': ['canton', 'region', 'switzerland', 'schweiz'],
                'region': ['area', 'zone', 'territory', 'gebiet'],
                'wald': ['forest', 'woodland', 'woods'],
                'forest': ['wald', 'woodland', 'woods'],
            }
            
            # Add related geo terms to search strategies if applicable
            query_lower = query.lower()
            for term, replacements in geo_terms.items():
                if term in query_lower:
                    for replacement in replacements:
                        for lang in ['en', 'de']:
                            search_strategies.append({
                                'keyword': replacement.lower(),
                                'language': lang,
                                'search_mode': 3
                            })
            
            # Special handling for certain geographical terms 
            if any(region in query_lower for region in ['aargau', 'zürich', 'bern', 'geneva']):
                # For Swiss cantons/regions, add search for geographical/administrative terms
                for term in ['administrative region', 'geographical region', 'canton', 'administrative unit']:
                    for lang in ['en', 'de']:
                        search_strategies.append({
                            'keyword': term.lower(),
                            'language': lang,
                            'search_mode': 3
                        })
            
            # Try all search strategies and collect found concepts
            for strategy in search_strategies:
                search_params = {
                    'keyword': strategy['keyword'],
                    'search_mode': strategy['search_mode'],
                    'thesaurus_uri': 'http://www.eionet.europa.eu/gemet/concept/',
                    'language': strategy['language']
                }
                
                logging.info(f"GEMET Search Strategy: {strategy}")
                try:
                    search_response = self.session.get(search_url, params=search_params, timeout=10, verify=self.verify_ssl)
                    logging.info(f"GEMET Search Response status: {search_response.status_code}")
                    
                    if search_response.status_code == 200:
                        search_concepts = search_response.json()
                        if search_concepts and len(search_concepts) > 0:
                            logging.info(f"GEMET found {len(search_concepts)} concepts with strategy: {strategy}")
                            # Add concepts to our collection
                            found_concepts.extend(search_concepts)
                            
                            # Mark this as a fallback result if we're using a related term
                            if strategy['keyword'] != query.lower():
                                logging.info(f"Used fallback term '{strategy['keyword']}' for original query '{query}'")
                except Exception as e:
                    logging.error(f"Error in GEMET search strategy: {e}")
            
            # If we have found concepts, process them
            if found_concepts:
                # Deduplicate concepts by URI
                unique_uris = set()
                unique_concepts = []
                
                for concept in found_concepts:
                    concept_uri = concept.get('uri', '')
                    if concept_uri and concept_uri not in unique_uris:
                        unique_uris.add(concept_uri)
                        unique_concepts.append(concept)
                
                # Process up to 5 unique concepts
                for concept in unique_concepts[:5]:
                    concept_uri = concept.get('uri', '')
                    
                    # Skip concepts without a proper URI
                    if not concept_uri:
                        logging.info(f"Skipping GEMET concept with missing URI")
                        continue
                    
                    # Get concept ID from URI (e.g., "5401" from "http://www.eionet.europa.eu/gemet/concept/5401")
                    concept_id = concept_uri.split('/')[-1]
                    logging.info(f"Processing GEMET concept ID: {concept_id}")
                    
                    # Get multilingual labels for this concept
                    multilingual_terms = self._get_gemet_multilingual_labels(concept_uri)
                    
                    # If we have multilingual terms, add this as a result
                    if multilingual_terms:
                        logging.info(f"Got multilingual terms for concept {concept_id}: {multilingual_terms}")
                        # Get definition for better description (English preferred, fallback to German)
                        definition = ""
                        if 'definition' in concept and 'string' in concept['definition']:
                            definition = concept['definition']['string']
                        elif not definition:
                            definition = self._get_gemet_definition(concept_uri)
                        
                        # If no definition, create a standard description
                        if not definition:
                            definition = f"Environmental term from GEMET thesaurus"
                        else:
                            # Truncate long definitions
                            if len(definition) > 100:
                                definition = definition[:97] + "..."
                        
                        # Create standardized concept URI format as specified
                        standard_uri = f"http://www.eionet.europa.eu/gemet/concept/{concept_id}"
                        
                        # Check if this term is likely to be relevant to the original query
                        is_relevant = True
                        
                        # For certain geographical queries where we had to use fallback terms,
                        # add a relevance note in the description
                        original_query_lower = query.lower()
                        if ('alp' in original_query_lower and not any(alp_term in str(multilingual_terms).lower() 
                                                                for alp_term in ['alp', 'mountain', 'berg'])) or \
                           ('fluss' in original_query_lower and not any(river_term in str(multilingual_terms).lower() 
                                                                 for river_term in ['fluss', 'river', 'stream'])) or \
                           ('aargau' in original_query_lower and not 'aargau' in str(multilingual_terms).lower()):
                            definition = f"Environmental term related to '{original_query_lower}': {definition}"
                        
                        results.append({
                            'source': 'GEMET',
                            'multilingual_label': multilingual_terms,
                            'uri': standard_uri,
                            'description': definition,
                            'entry_id': concept_id,  # Store concept ID for consistency with other sources
                            'is_synonym_result': original_query_lower not in str(multilingual_terms).lower(),
                            'found_via_query': original_query_lower
                        })
                        logging.info(f"Added GEMET result: {concept_id}")
                    else:
                        logging.info(f"No multilingual terms found for concept {concept_id}")
            
            # If we still have no results for specific geographical terms, try one more approach
            # by looking for more general environmental concepts
            if not results and any(geo_term in query.lower() for geo_term in ['alp', 'fluss', 'aargau']):
                logging.info(f"No GEMET results found for '{query}'. Trying general environmental concepts as fallback.")
                
                # Mapping of problem terms to known working GEMET concepts
                term_mapping = {
                    'alp': ['mountain', 'berg', 'alpine', 'gebirge'],
                    'fluss': ['river', 'watercourse', 'gewässer', 'water body'],
                    'aargau': ['administrative region', 'geographical region', 'canton', 'region']
                }
                
                # Find matching terms for our query
                matching_terms = []
                for term, alternatives in term_mapping.items():
                    if term in query.lower():
                        matching_terms.extend(alternatives)
                
                if not matching_terms:
                    # Default general environmental concepts as last resort
                    matching_terms = ['environmental protection', 'nature conservation', 'ecosystem']
                
                for term in matching_terms[:3]:  # Try up to 3 related terms
                    if len(results) >= 3:  # Limit to 3 fallback results
                        break
                        
                    for lang in ['en', 'de']:
                        try:
                            search_params = {
                                'keyword': term.lower(),
                                'search_mode': 3,  # Contains search for better matches
                                'thesaurus_uri': 'http://www.eionet.europa.eu/gemet/concept/',
                                'language': lang
                            }
                            
                            fallback_response = self.session.get(search_url, params=search_params, timeout=10, verify=self.verify_ssl)
                            if fallback_response.status_code == 200:
                                fallback_concepts = fallback_response.json()
                                if fallback_concepts:
                                    for concept in fallback_concepts[:1]:  # Take only the first match
                                        concept_uri = concept.get('uri', '')
                                        if not concept_uri:
                                            continue
                                            
                                        concept_id = concept_uri.split('/')[-1]
                                        multilingual_terms = self._get_gemet_multilingual_labels(concept_uri)
                                        
                                        if multilingual_terms:
                                            standard_uri = f"http://www.eionet.europa.eu/gemet/concept/{concept_id}"
                                            
                                            # Get definition for context
                                            definition = ""
                                            if 'definition' in concept and 'string' in concept['definition']:
                                                definition = concept['definition']['string']
                                            else:
                                                definition = self._get_gemet_definition(concept_uri)
                                                
                                            # Add fallback note
                                            description = f"Environmental term related to '{query}': "
                                            if definition:
                                                description += definition[:80] + "..." if len(definition) > 80 else definition
                                            else:
                                                description = f"Environmental term related to '{query}' from GEMET thesaurus"
                                            
                                            results.append({
                                                'source': 'GEMET',
                                                'multilingual_label': multilingual_terms,
                                                'uri': standard_uri,
                                                'description': description,
                                                'entry_id': concept_id,
                                                'is_synonym_result': True,
                                                'found_via_query': term
                                            })
                                            break
                        except Exception as e:
                            logging.error(f"Error in specific fallback search: {e}")
            
        except Exception as e:
            logging.error(f"Error searching GEMET: {e}", exc_info=True)
        
        # Cache results to avoid repeated API calls for the same query
        try:
            cache.set(cache_key, results, timeout=3600)  # 1 hour cache
            logging.info(f"Cached {len(results)} GEMET results for '{query}'")
        except Exception as e:
            logging.debug(f"GEMET cache write failed for '{query}': {e}")
        
        logging.info(f"GEMET search returned {len(results)} results")
        return results
    
    def _get_gemet_definition(self, concept_uri):
        """Get definition for a GEMET concept in English or German"""
        for language in ['en', 'de']:
            try:
                # Get concept details including definition
                url = "https://www.eionet.europa.eu/gemet/getConcept"
                params = {
                    'concept_uri': concept_uri,
                    'language': language
                }
                
                response = self.session.get(url, params=params, timeout=5, verify=self.verify_ssl)
                if response.status_code == 200:
                    data = response.json()
                    if 'definition' in data and 'string' in data['definition']:
                        return data['definition']['string']
            except Exception:
                pass
        return ""
    
    def _get_gemet_multilingual_labels(self, concept_uri):
        """Get multilingual labels for a GEMET concept using the official API
        Returns a dictionary mapping language codes to preferred labels
        """
        multilingual_terms = {}
        
        # Check cache first
        cache_key = f"gemet_labels:{concept_uri.split('/')[-1]}"
        try:
            cached = cache.get(cache_key)
            if cached:
                logging.info(f"Using cached GEMET labels for {concept_uri}")
                return cached
        except Exception:
            pass
        
        try:
            logging.info(f"Getting multilingual labels for GEMET concept: {concept_uri}")
            # Get all translations for this concept's preferred label
            url = "https://www.eionet.europa.eu/gemet/getAllTranslationsForConcept"
            params = {
                'concept_uri': concept_uri,
                'property_uri': 'http://www.w3.org/2004/02/skos/core#prefLabel'
            }
            
            response = self.session.get(url, params=params, timeout=8, verify=self.verify_ssl)
            logging.info(f"GEMET translations response status: {response.status_code}")
            
            if response.status_code == 200:
                translations = response.json()
                if isinstance(translations, list):
                    logging.info(f"Found {len(translations)} translations for concept")
                    # GEMET supports many languages - extract only the ones we need
                    for translation in translations:
                        lang_code = translation.get('language', '')
                        # Map GEMET language codes to our standardized ones
                        if lang_code in ['de', 'fr', 'it', 'en']:
                            label = translation.get('string', '')
                            if label:
                                multilingual_terms[lang_code] = label
                                logging.info(f"Found {lang_code} label: {label}")
            
            # Only cache if we found at least some translations
            if multilingual_terms:
                try:
                    cache.set(cache_key, multilingual_terms, timeout=24*3600)  # 24 hour cache
                    logging.info(f"Cached {len(multilingual_terms)} translations for concept")
                except Exception as e:
                    logging.debug(f"Error caching GEMET labels: {e}")
            else:
                logging.info("No translations found for concept")
                
        except Exception as e:
            logging.error(f"Error getting GEMET multilingual labels for {concept_uri}: {e}", exc_info=True)
        
        return multilingual_terms if multilingual_terms else None
    
    def search_wikidata(self, query):
        """Search Wikidata for keywords with multilingual support"""
        results = []
        try:
            # Try cached final results first to avoid repeated Wikidata calls
            cache_key = f"wd_results:{query.lower()}"
            cached = cache.get(cache_key)
            if cached:
                logging.debug(f"Returning cached Wikidata results for '{query}'")
                return cached

            # First search in English to get entity IDs
            search_params = {
                'action': 'wbsearchentities',
                'search': query,
                'language': 'en',
                'format': 'json',
                'limit': 3
            }

            search_response = self.session.get(self.wikidata_base_url, params=search_params, timeout=10, verify=self.verify_ssl)
            if search_response.status_code == 200:
                search_data = search_response.json()
                for item in search_data.get('search', []):
                    entity_id = item.get('id', '')
                    if entity_id:
                        # Get multilingual labels for this entity
                        multilingual_terms = self._get_wikidata_multilingual_labels(entity_id)
                        if multilingual_terms:
                            results.append({
                                'source': 'Wikidata',
                                'multilingual_label': multilingual_terms,
                                'uri': f"http://www.wikidata.org/entity/{entity_id}",
                                'description': item.get('description', f'Wikidata entity for {query}')
                            })

            try:
                cache.set(cache_key, results, timeout=3600)
            except Exception:
                pass

        except Exception as e:
            logging.error(f"Error searching Wikidata: {e}")

        return results

    def _get_wikidata_multilingual_labels(self, entity_id):
        """Get multilingual labels for a Wikidata entity"""
        multilingual_terms = {}
        try:
            cache_key = f"wd_labels:{entity_id}"
            cached = cache.get(cache_key)
            if cached:
                return cached

            # Use Wikidata API to get entity data with labels in multiple languages
            entity_params = {
                'action': 'wbgetentities',
                'ids': entity_id,
                'props': 'labels',
                'languages': 'de|fr|it|en',  # Request all 4 languages
                'format': 'json'
            }

            entity_response = self.session.get(self.wikidata_base_url, params=entity_params, timeout=10, verify=self.verify_ssl)
            if entity_response.status_code == 200:
                entity_data = entity_response.json()
                if 'entities' in entity_data and entity_id in entity_data['entities']:
                    entity = entity_data['entities'][entity_id]
                    labels = entity.get('labels', {})

                    # Extract labels for each language
                    for lang_code in ['de', 'fr', 'it', 'en']:
                        if lang_code in labels and 'value' in labels[lang_code]:
                            multilingual_terms[lang_code] = labels[lang_code]['value']

            if multilingual_terms:
                try:
                    cache.set(cache_key, multilingual_terms, timeout=24*3600)
                except Exception:
                    pass

        except Exception as e:
            logging.debug(f"Error getting Wikidata multilingual labels for {entity_id}: {e}")

        return multilingual_terms if multilingual_terms else None
    
    def _convert_to_i14y_format(self, keywords):
        """Convert multilingual keywords to I14Y format (new schema with label + uri).
        Returns a list like:
        [{"label": {"de": "..", "en": ".."}, "uri": "..."}, ...]
        Falls back to legacy language keys if labels are missing.
        """
        i14y_keywords = []
        
        for keyword in keywords:
            labels = {}
            uri = keyword.get('uri') or keyword.get('_uri', '')
            
            if 'multilingual_label' in keyword:
                multilingual_labels = keyword['multilingual_label']
                for lang_code in ['de', 'fr', 'it', 'en', 'rm']:
                    if multilingual_labels.get(lang_code):
                        labels[lang_code] = multilingual_labels[lang_code]
            else:
                # Fallback for old format (single label)
                label = keyword.get('label', '')
                if label:
                    labels['de'] = label
            
            if labels:
                i14y_keywords.append({
                    'label': labels,
                    'uri': uri
                })
        
        return i14y_keywords

    def generate_keywords(self, query, query_lang='de', include_synonyms=True):
        """Generate keywords following DCAT-AP CH priority cascade with exact match prioritization and synonym support.
        Supports both single keyword queries and multi-sentence descriptions.
        
        Args:
            query: Single keyword or longer description text
            query_lang: Language code (de, fr, it, en)
            include_synonyms: Whether to include synonym expansion
            
        Returns:
            list: List of keyword dictionaries with metadata
        """
        all_keywords = []
        
        logging.info(f"Starting keyword generation for: '{query[:100]}...' in language: {query_lang}")
        
        # Step 1: Extract keywords from the input text
        # If it's a short query (1-3 words), treat as single keyword
        # If it's longer, extract multiple keywords
        extracted_keywords = self._extract_keywords_from_text(query, max_keywords=5)
        logging.info(f"Extracted keywords: {extracted_keywords}")
        
        # Step 2: Search for each extracted keyword in vocabularies
        for keyword in extracted_keywords:
            logging.info(f"Searching vocabularies for keyword: '{keyword}'")
            
            # Add synonym support if enabled
            if include_synonyms:
                synonyms = self._get_synonym_queries(keyword)
                logging.info(f"Including synonyms for '{keyword}': {synonyms}")
            
            # Priority 1: TERMDAT (multilingual) - with synonym support
            termdat_results = self.search_termdat(keyword, query_lang, include_synonyms)
            for result in termdat_results:
                result['extracted_from'] = query if len(query.split()) > 3 else None
                result['search_keyword'] = keyword
            all_keywords.extend(termdat_results)
            
            # Priority 2: GEMET (multilingual)
            gemet_results = self.search_gemet(keyword)
            for result in gemet_results:
                result['extracted_from'] = query if len(query.split()) > 3 else None
                result['search_keyword'] = keyword
            all_keywords.extend(gemet_results)
            
            # Priority 3: Wikidata (multilingual)
            wikidata_results = self.search_wikidata(keyword)
            for result in wikidata_results:
                result['extracted_from'] = query if len(query.split()) > 3 else None
                result['search_keyword'] = keyword
            all_keywords.extend(wikidata_results)
        
        # Remove duplicates (same URI)
        seen_uris = set()
        unique_keywords = []
        for kw in all_keywords:
            uri = kw.get('uri', '')
            if uri and uri not in seen_uris:
                seen_uris.add(uri)
                unique_keywords.append(kw)
            elif not uri:  # Keep keywords without URIs (literals)
                unique_keywords.append(kw)
        
        # Sort all results: exact matches first, then by synonym priority, then by source priority
        def sort_key(result):
            # Check against all extracted keywords for exact match
            multilingual_terms = result.get('multilingual_label', {})
            search_keyword = result.get('search_keyword', query).lower()
            
            # Check for exact match in any language
            has_exact_match = any(
                term.lower() == search_keyword
                for term in multilingual_terms.values()
            )
            
            # Check if this was found via synonym search
            is_synonym_result = result.get('is_synonym_result', False)
            
            # Source priority mapping
            source_priority = {
                'TERMDAT': 0,
                'GEMET': 1, 
                'Wikidata': 2
            }
            
            source = result.get('source', 'Unknown')
            source_rank = source_priority.get(source, 999)
            
            # Return tuple: (exact_match_priority, synonym_priority, source_priority, negative_language_count)
            return (
                0 if has_exact_match else 1,
                1 if is_synonym_result else 0,
                source_rank,
                -len(result.get('available_languages', []))
            )
        
        unique_keywords.sort(key=sort_key)
        
        # Limit final results to 15 (increased from 10 to accommodate multiple extracted keywords)
        return unique_keywords[:15]
    
    def generate_keywords_progressive(self, i14y_item, query_lang='de'):
        """Generate keywords using improved progressive algorithm for workflow 1.
        
        Algorithm:
        1. If keywords exist in I14Y, query TERMDAT→GEMET→Wikidata for exact matches
        2. If < 8 matches, extract keywords with NLP and query vocabularies
        3. If still < 8 matches, use OpenAI to extract keywords and query vocabularies
        4. Stop when we have 8 keywords AND at least 3 different ones
        
        Args:
            i14y_item: Full I14Y item object with metadata
            query_lang: Language code (de, fr, it, en)
            
        Returns:
            list: List of keyword dictionaries with metadata
        """
        all_keywords = []
        seen_uris = set()
        seen_labels = set()
        existing_lang_keys = ['de', 'fr', 'it', 'en', 'rm']
        
        def add_unique_keywords(new_keywords):
            """Add keywords to collection, avoiding duplicates"""
            for kw in new_keywords:
                uri = kw.get('uri', '')
                # Get a normalized label for deduplication
                labels = kw.get('multilingual_label', {})
                label_key = tuple(sorted(labels.values())) if labels else str(kw)
                
                if uri and uri not in seen_uris:
                    seen_uris.add(uri)
                    seen_labels.add(label_key)
                    all_keywords.append(kw)
                elif not uri and label_key not in seen_labels:
                    seen_labels.add(label_key)
                    all_keywords.append(kw)
        
        def count_unique_labels():
            """Count truly unique keywords (different labels)"""
            unique = set()
            for kw in all_keywords:
                labels = kw.get('multilingual_label', {})
                if labels:
                    # Use the German label primarily, or first available
                    primary_label = labels.get('de') or labels.get('en') or labels.get('fr') or next(iter(labels.values()), '')
                    if primary_label:
                        unique.add(primary_label.lower())
            return len(unique)
        
        def should_stop():
            """Check if we should stop generating keywords"""
            return len(all_keywords) >= 15 and count_unique_labels() >= 5

        def extract_label_candidates(keyword_obj):
            """Return list of (label, lang) tuples for an I14Y keyword entry"""
            candidates = []

            def add_candidate(text, lang=None):
                if not text:
                    return
                normalized = str(text).strip()
                if normalized:
                    candidates.append((normalized, (lang or '').lower() or None))

            if isinstance(keyword_obj, dict):
                label_field = keyword_obj.get('label')
                if isinstance(label_field, str):
                    add_candidate(label_field)
                elif isinstance(label_field, dict):
                    nested = label_field.get('multilingual_label') if isinstance(label_field.get('multilingual_label'), dict) else None
                    if nested:
                        for lang_key in existing_lang_keys:
                            if nested.get(lang_key):
                                add_candidate(nested[lang_key], lang_key)
                    else:
                        for lang_key in existing_lang_keys:
                            if label_field.get(lang_key):
                                add_candidate(label_field[lang_key], lang_key)

                multilingual = keyword_obj.get('multilingual_label')
                if isinstance(multilingual, dict):
                    for lang_key in existing_lang_keys:
                        if multilingual.get(lang_key):
                            add_candidate(multilingual[lang_key], lang_key)

                for lang_key in existing_lang_keys:
                    if keyword_obj.get(lang_key):
                        add_candidate(keyword_obj[lang_key], lang_key)

            elif isinstance(keyword_obj, str):
                add_candidate(keyword_obj)

            return candidates
        
        logging.info("=== Starting Progressive Keyword Generation ===")
        
        # STAGE 1: Try exact matches for existing I14Y keywords
        existing_keywords = []
        if i14y_item.get('keywords'):
            existing_keywords = i14y_item['keywords']
        elif i14y_item.get('keyword'):
            existing_keywords = i14y_item['keyword']
        elif i14y_item.get('theme'):
            existing_keywords = i14y_item['theme']
        
        if not isinstance(existing_keywords, list):
            existing_keywords = []
        
        if existing_keywords:
            logging.info(f"Stage 1: Searching for exact matches of {len(existing_keywords)} existing keywords")
            searched_terms = set()
            for kw in existing_keywords:
                if should_stop():
                    break
                label_candidates = extract_label_candidates(kw)
                if not label_candidates:
                    continue

                for label_value, lang_hint in label_candidates:
                    if should_stop():
                        break

                    normalized_key = (label_value.lower(), lang_hint or 'auto')
                    if normalized_key in searched_terms:
                        continue
                    searched_terms.add(normalized_key)

                    search_lang = lang_hint or query_lang

                    termdat_matches = self.search_termdat(label_value, search_lang, include_synonyms=False)
                    add_unique_keywords(termdat_matches)
                    
                    if not should_stop():
                        gemet_matches = self.search_gemet(label_value)
                        add_unique_keywords(gemet_matches)
                    
                    if not should_stop():
                        wikidata_matches = self.search_wikidata(label_value)
                        add_unique_keywords(wikidata_matches)
            
            logging.info(f"After Stage 1: {len(all_keywords)} keywords, {count_unique_labels()} unique")
        
        # STAGE 2: If < 10 matches, use NLP extraction
        if not should_stop():
            logging.info("Stage 2: Extracting keywords with NLP")
            
            # Gather text for analysis
            text_parts = []
            if i14y_item.get('title'):
                title = i14y_item['title']
                if isinstance(title, dict):
                    text_parts.append(title.get('de') or title.get('en') or title.get('fr') or '')
                else:
                    text_parts.append(str(title))
            
            if i14y_item.get('description'):
                desc = i14y_item['description']
                if isinstance(desc, dict):
                    text_parts.append(desc.get('de') or desc.get('en') or desc.get('fr') or '')
                else:
                    text_parts.append(str(desc))
            
            analysis_text = ' '.join(text_parts)
            
            if analysis_text:
                extracted = self._extract_keywords_with_nlp(analysis_text, max_keywords=8)
                logging.info(f"NLP extracted keywords: {extracted}")
                
                for keyword in extracted:
                    if should_stop():
                        break
                    
                    termdat_matches = self.search_termdat(keyword, query_lang, include_synonyms=True)
                    add_unique_keywords(termdat_matches)
                    
                    if not should_stop():
                        gemet_matches = self.search_gemet(keyword)
                        add_unique_keywords(gemet_matches)
                    
                    if not should_stop():
                        wikidata_matches = self.search_wikidata(keyword)
                        add_unique_keywords(wikidata_matches)
                
                logging.info(f"After Stage 2: {len(all_keywords)} keywords, {count_unique_labels()} unique")
        
        # STAGE 3: If still < 10 matches, use OpenAI
        if not should_stop() and self.openai_api_key:
            logging.info("Stage 3: Extracting keywords with OpenAI")
            
            # Gather all available text including publisher and themes
            text_parts = []
            if i14y_item.get('title'):
                title = i14y_item['title']
                if isinstance(title, dict):
                    text_parts.append("Title: " + (title.get('de') or title.get('en') or title.get('fr') or ''))
                else:
                    text_parts.append("Title: " + str(title))
            
            if i14y_item.get('description'):
                desc = i14y_item['description']
                if isinstance(desc, dict):
                    text_parts.append("Description: " + (desc.get('de') or desc.get('en') or desc.get('fr') or ''))
                else:
                    text_parts.append("Description: " + str(desc))
            
            if i14y_item.get('publisher'):
                pub = i14y_item['publisher']
                if isinstance(pub, dict):
                    pub_name = pub.get('name', pub)
                    if isinstance(pub_name, dict):
                        text_parts.append("Publisher: " + (pub_name.get('de') or pub_name.get('en') or ''))
                    else:
                        text_parts.append("Publisher: " + str(pub_name))
                else:
                    text_parts.append("Publisher: " + str(pub))
            
            analysis_text = '\n'.join(text_parts)
            
            if analysis_text:
                try:
                    extracted = self._extract_keywords_with_openai(analysis_text, max_keywords=10)
                    if extracted:
                        logging.info(f"OpenAI extracted keywords: {extracted}")
                        
                        for keyword in extracted:
                            if should_stop():
                                break
                            
                            termdat_matches = self.search_termdat(keyword, query_lang, include_synonyms=True)
                            add_unique_keywords(termdat_matches)
                            
                            if not should_stop():
                                gemet_matches = self.search_gemet(keyword)
                                add_unique_keywords(gemet_matches)
                            
                            if not should_stop():
                                wikidata_matches = self.search_wikidata(keyword)
                                add_unique_keywords(wikidata_matches)
                        
                        logging.info(f"After Stage 3: {len(all_keywords)} keywords, {count_unique_labels()} unique")
                except Exception as e:
                    logging.error(f"OpenAI extraction failed: {e}")
        
        logging.info(f"=== Final: {len(all_keywords)} keywords, {count_unique_labels()} unique ===")
        return all_keywords

keyword_generator = KeywordGenerator()

@app.route('/')
def welcome():
    """Landing page for workflow selection"""
    return render_template('welcome.html')

@app.route('/generate')
def index():
    """Generate keywords from description"""
    return render_template('index.html')

@app.route('/refine')
def refine():
    """Refine existing I14Y item keywords - redirect to step 1"""
    return redirect(url_for('refine_step1'))

@app.route('/refine/step1')
def refine_step1():
    """Step 1: Search for I14Y item"""
    return render_template('refine_step1.html')

@app.route('/refine/step2')
def refine_step2():
    """Step 2: Review current item details"""
    return render_template('refine_step2.html')

@app.route('/refine/step3')
def refine_step3():
    """Step 3: Review, arrange, and select keywords"""
    return render_template('refine_step3.html')

@app.route('/refine/step4')
def refine_step4():
    """Step 4: Update keywords on I14Y"""
    return render_template('refine_step4.html')

@app.route('/search', methods=['POST'])
def search_keywords():
    query = request.json.get('query', '').strip()
    query_lang = request.json.get('lang', 'de').strip().lower()
    include_synonyms = request.json.get('include_synonyms', True)  # Default to True
    source_filter = request.json.get('source', 'all').lower()  # termdat, gemet, wikidata, or all
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Check cache first (include synonyms and source in cache key)
    cache_key = f"{query}:{query_lang}:{include_synonyms}:{source_filter}"
    cached_result = cache.get(cache_key)
    if cached_result:
        return jsonify(cached_result)

    try:
        # Extract keywords to show user what was searched
        extracted_keywords = keyword_generator._extract_keywords_from_text(query, max_keywords=5)
        
        # Generate keywords based on source filter
        if source_filter == 'all':
            keywords = keyword_generator.generate_keywords(query, query_lang, include_synonyms)
        else:
            keywords = []
            for keyword in extracted_keywords:
                if source_filter == 'termdat':
                    keywords.extend(keyword_generator.search_termdat(keyword, query_lang, include_synonyms))
                elif source_filter == 'gemet':
                    keywords.extend(keyword_generator.search_gemet(keyword))
                elif source_filter == 'wikidata':
                    keywords.extend(keyword_generator.search_wikidata(keyword))
        
        i14y_keywords = keyword_generator._convert_to_i14y_format(keywords)
        
        # Ensure all keywords are JSON serializable
        sanitized_keywords = []
        for kw in keywords:
            try:
                # Create a new dict with only necessary serializable data
                sanitized_kw = {
                    'source': kw.get('source', ''),
                    'multilingual_label': kw.get('multilingual_label', {}),
                    'uri': kw.get('uri', ''),
                    'description': kw.get('description', ''),
                    'entry_id': kw.get('entry_id', ''),
                    'query_lang': kw.get('query_lang', query_lang),
                    'available_languages': kw.get('available_languages', list(kw.get('multilingual_label', {}).keys())),
                    'is_synonym_result': kw.get('is_synonym_result', False),
                    'found_via_query': kw.get('found_via_query', query),
                    'extracted_from': kw.get('extracted_from'),
                    'search_keyword': kw.get('search_keyword')
                }
                sanitized_keywords.append(sanitized_kw)
            except Exception as e:
                logging.error(f"Error sanitizing keyword for JSON: {e}")
        
        # Determine if this was a multi-keyword extraction
        is_text_extraction = len(query.split()) > 3
        
        result = {
            'query': query,
            'query_lang': query_lang,
            'include_synonyms': include_synonyms,
            'is_text_extraction': is_text_extraction,
            'extracted_keywords': extracted_keywords if is_text_extraction else None,
            'keywords': sanitized_keywords,  # Use sanitized keywords
            'i14y_keywords': i14y_keywords,  # I14Y-ready format for upload
            'total': len(sanitized_keywords),
            'synonym_count': len([k for k in sanitized_keywords if k.get('is_synonym_result', False)])
        }
        
        # Test JSON serialization before caching
        try:
            json.dumps(result)
            cache.set(cache_key, result)
            return jsonify(result)
        except Exception as json_error:
            logging.error(f"JSON serialization error: {json_error}")
            return jsonify({'error': 'Error creating JSON response', 'details': str(json_error)}), 500
            
    except Exception as e:
        logging.error(f"Search error: {e}", exc_info=True)
        return jsonify({'error': f'Error during search: {str(e)}'}), 500

@app.route('/generate-progressive-keywords', methods=['POST'])
def generate_progressive_keywords():
    """Generate keywords using improved progressive algorithm for workflow 1"""
    data = request.json
    i14y_item = data.get('i14y_item', {})
    query_lang = data.get('lang', 'de').strip().lower()
    
    if not i14y_item:
        return jsonify({'error': 'I14Y item data is required'}), 400
    
    try:
        logging.info(f"Progressive keyword generation for item: {i14y_item.get('id', 'unknown')}")
        
        # Generate keywords using progressive algorithm
        keywords = keyword_generator.generate_keywords_progressive(i14y_item, query_lang)
        
        # Convert to I14Y format
        i14y_keywords = keyword_generator._convert_to_i14y_format(keywords)
        
        # Sanitize for JSON
        sanitized_keywords = []
        for kw in keywords:
            try:
                sanitized_kw = {
                    'source': kw.get('source', ''),
                    'multilingual_label': kw.get('multilingual_label', {}),
                    'uri': kw.get('uri', ''),
                    'description': kw.get('description', ''),
                    'entry_id': kw.get('entry_id', ''),
                    'query_lang': kw.get('query_lang', query_lang),
                    'available_languages': kw.get('available_languages', list(kw.get('multilingual_label', {}).keys())),
                }
                sanitized_keywords.append(sanitized_kw)
            except Exception as e:
                logging.error(f"Error sanitizing keyword: {e}")
        
        result = {
            'keywords': sanitized_keywords,
            'i14y_keywords': i14y_keywords,
            'total': len(sanitized_keywords),
            'unique_count': len(set(
                kw.get('multilingual_label', {}).get('de', '') or 
                kw.get('multilingual_label', {}).get('en', '') or 
                kw.get('multilingual_label', {}).get('fr', '')
                for kw in sanitized_keywords
            ))
        }
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Progressive keyword generation error: {e}", exc_info=True)
        return jsonify({'error': f'Error generating keywords: {str(e)}'}), 500

@app.route('/upload')
def upload_keywords():
    """Upload page - no authentication needed, prompts for token at upload time"""
    return render_template('upload.html')

@app.route('/upload-to-i14y', methods=['POST'])
def upload_to_i14y():
    """Server-side proxy to upload keywords to I14Y API (avoids CORS issues)"""
    data = request.json
    dataset_guid = data.get('dataset_guid', '').strip()
    i14y_token = data.get('i14y_token', '').strip()
    new_keywords = data.get('keywords', [])
    
    if not dataset_guid:
        return jsonify({'error': 'Dataset GUID is required'}), 400
    
    if not i14y_token:
        return jsonify({'error': 'I14Y token is required'}), 400
    
    if not new_keywords:
        return jsonify({'error': 'Keywords are required'}), 400
    
    # Set SSL verification based on network environment
    verify_ssl = NETWORK_ENV != "internal"
    
    try:
        # Clean the token - remove "Bearer " prefix if it exists
        clean_token = i14y_token
        if i14y_token.lower().startswith('bearer '):
            clean_token = i14y_token[7:]  # Remove "Bearer " (7 characters)
                
        # Use the correct I14Y API URL
        i14y_url = f"https://api.i14y.admin.ch/api/partner/v1/datasets/{dataset_guid}"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {clean_token}",
            'Accept': '*/*'
        }
        
        # Step 1: GET the existing dataset
        get_response = requests.get(i14y_url, headers=headers, timeout=30, verify=verify_ssl)
        
        if not get_response.ok:
            return jsonify({
                'error': f'Failed to retrieve dataset: HTTP {get_response.status_code}',
                'details': get_response.text
            }), get_response.status_code
        
        # Step 2: Parse the existing dataset
        dataset = get_response.json()
        
        # Ensure we have the data structure
        if 'data' not in dataset:
            return jsonify({'error': 'Invalid dataset structure received from I14Y'}), 500
        
        dataset_data = dataset['data']
        
        # Step 3: Fix missing email in contactPoints if needed
        if 'contactPoints' in dataset_data and len(dataset_data['contactPoints']) > 0:
            first_contact = dataset_data['contactPoints'][0]
            if not first_contact.get('hasEmail') or first_contact.get('hasEmail').strip() == '':
                first_contact['hasEmail'] = 'i14y@bfs.admin.ch'
        
        # Helpers to normalize keywords to new structure for comparison
        def normalize_keyword(kw):
            """Return {'label': {...}, 'uri': ...} regardless of incoming schema"""
            label = {}
            uri = ''
            if not isinstance(kw, dict):
                return None
            
            # New schema
            if isinstance(kw.get('label'), dict):
                for lang in ['de', 'fr', 'it', 'en', 'rm']:
                    if kw['label'].get(lang):
                        label[lang] = kw['label'][lang]
            
            # Legacy language keys
            for lang in ['de', 'fr', 'it', 'en', 'rm']:
                if kw.get(lang):
                    label.setdefault(lang, kw[lang])
            
            uri = kw.get('uri') or kw.get('_uri', '')
            
            if not label:
                return None
            return {'label': label, 'uri': uri}

        # Step 4: Find ALL keyword arrays in the dataset and merge them
        existing_keywords_raw = []
        
        # Check for keywords at the top level of dataset_data
        if 'keywords' in dataset_data and isinstance(dataset_data['keywords'], list):
            existing_keywords_raw.extend(dataset_data['keywords'])
        
        # Check for keywords in distributions (if they exist there)
        if 'distributions' in dataset_data:
            for dist in dataset_data['distributions']:
                if isinstance(dist, dict) and isinstance(dist.get('keywords'), list):
                    existing_keywords_raw.extend(dist['keywords'])
                
        # Detect whether existing dataset already uses new schema
        uses_new_schema = any(isinstance(k, dict) and 'label' in k for k in existing_keywords_raw)

        # Normalize existing keywords
        normalized_existing = []
        for kw in existing_keywords_raw:
            norm = normalize_keyword(kw)
            if norm:
                normalized_existing.append(norm)
            else:
                logging.debug(f"Skipping invalid keyword from dataset: {kw}")
        
        # Deduplicate by preferred language text
        def keyword_key(norm_kw):
            lbl = norm_kw.get('label', {})
            for lang in ['de', 'en', 'fr', 'it', 'rm']:
                if lbl.get(lang):
                    return lbl[lang].lower()
            return None
        
        existing_keys = set()
        for nkw in normalized_existing:
            key = keyword_key(nkw)
            if key:
                existing_keys.add(key)
        
        # Normalize incoming keywords (from client) and merge
        keywords_added = 0
        normalized_new = []
        for kw in new_keywords:
            norm = normalize_keyword(kw)
            if not norm:
                continue
            key = keyword_key(norm)
            if key and key not in existing_keys:
                normalized_new.append(norm)
                existing_keys.add(key)
                keywords_added += 1
        
        merged_keywords = normalized_existing + normalized_new

        # Decide output schema: keep original (legacy vs new)
        output_keywords = []
        if uses_new_schema:
            for kw in merged_keywords:
                output_keywords.append({
                    'label': kw.get('label', {}),
                    'uri': kw.get('uri', '')
                })
        else:
            # Legacy schema with language keys; include uri for forward compatibility
            for kw in merged_keywords:
                label = kw.get('label', {})
                legacy_kw = {lang: text for lang, text in label.items() if text}
                if kw.get('uri'):
                    legacy_kw['uri'] = kw['uri']
                if legacy_kw:
                    output_keywords.append(legacy_kw)
        
        # Update the dataset with the merged keywords - IMPORTANT: Set only once!
        dataset_data['keywords'] = output_keywords
        
        # Remove any other keywords arrays to avoid conflicts
        if 'distributions' in dataset_data:
            for dist in dataset_data['distributions']:
                if isinstance(dist, dict) and 'keywords' in dist:
                    del dist['keywords']  # Remove keywords from distributions
                
        # Step 5: PUT the complete dataset back
        put_payload = {
            'data': dataset_data
        }
        
        put_response = requests.put(i14y_url, headers=headers, json=put_payload, timeout=30, verify=verify_ssl)
        
        if put_response.ok:
            # Create the I14Y dataset link
            dataset_link = f"https://input.i14y.admin.ch/catalog/datasets/{dataset_guid}"
            
            return jsonify({
                'success': True, 
                'message': f'Successfully added {keywords_added} new keywords to I14Y dataset',
                'total_keywords': len(normalized_existing),
                'keywords_added': keywords_added,
                'status_code': put_response.status_code,
                'dataset_link': dataset_link
            })
        else:
            # Try to get error details from I14Y response
            try:
                error_data = put_response.json()
                error_message = error_data.get('message', f'HTTP {put_response.status_code}')
            except:
                error_message = f'HTTP {put_response.status_code}: {put_response.reason}'
            
            return jsonify({
                'error': f'I14Y API error during update: {error_message}',
                'status_code': put_response.status_code,
                'response_body': put_response.text
            }), put_response.status_code
            
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request to I14Y API timed out'}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Could not connect to I14Y API'}), 503
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/search-i14y-datasets', methods=['POST'])
def search_i14y_datasets():
    """Search I14Y datasets using the I14Y API search endpoint"""
    query = request.json.get('query', '').strip()
    page = request.json.get('page', 1)
    page_size = request.json.get('pageSize', 25)
    object_type_filter = request.json.get('objectType', '').strip().lower()  # 'dataset', 'dataservice', 'publicservice', 'concept'
    organisation_filter = request.json.get('organisation', '').strip()  # Organisation ID or name
    
    if not query:
        return jsonify({'error': 'Search query is required'}), 400
    
    try:
        # I14Y API search endpoint (selectable env)
        env = request.json.get('env', 'prod').lower()
        if env not in ['abnahme', 'prod']:
            env = 'prod'

        search_base = {
            'abnahme': "https://input-backend.i14y.a.c.bfs.admin.ch",
            'prod': "https://input-backend.i14y.c.bfs.admin.ch"
        }[env]

        i14y_search_url = f"{search_base}/api/Catalog/search"
        
        params = {
            'query': query,
            'page': page,
            'pageSize': page_size
        }
        
        logging.info(f"Searching I14Y for: '{query}' (page {page}, size {page_size})")
        
        # Make request to I14Y search API
        response = requests.get(i14y_search_url, params=params, timeout=10)
        
        if not response.ok:
            logging.error(f"I14Y API error: {response.status_code} - {response.text}")
            return jsonify({'error': f'I14Y API returned {response.status_code}'}), response.status_code
        
        data = response.json()
        
        # Extract relevant dataset information
        # The API response is a list of datasets directly
        datasets = []
        
        # Handle both list and dict responses
        if isinstance(data, list):
            data_list = data
        elif isinstance(data, dict):
            # Try different possible keys for datasets
            data_list = data.get('datasets', data.get('data', []))
            if isinstance(data_list, dict):
                data_list = []
        else:
            data_list = []
        
        logging.info(f"I14Y API response type: {type(data)}, raw data: {str(data)[:200]}")
        
        for dataset in data_list:
            if not isinstance(dataset, dict):
                continue
            
            # Helper function to extract text from potentially multilingual fields
            def extract_text(field):
                if isinstance(field, dict):
                    # Prefer EN, then DE, then FR, then IT
                    return field.get('en') or field.get('de') or field.get('fr') or field.get('it') or ''
                elif isinstance(field, str):
                    return field
                return ''

            def extract_status_field(field):
                if isinstance(field, dict):
                    return field.get('de') or field.get('fr') or field.get('en') or field.get('it') or ''
                elif isinstance(field, str):
                    return field
                return ''
            
            def extract_publisher_name(publisher_field):
                """Extract publisher name from nested structure: publisher.name[lang]"""
                if isinstance(publisher_field, dict):
                    name = publisher_field.get('name')
                    if name:
                        return extract_text(name)
                    # Fallback to direct multilingual field
                    return extract_text(publisher_field)
                elif isinstance(publisher_field, str):
                    return publisher_field
                return ''
            
            # Determine object type from the dataset structure
            # The I14Y API returns different fields based on object type
            # First check if there's an explicit type field
            object_type = 'dataset'  # Default
            
            # Check for explicit type indicators first
            if 'type' in dataset:
                type_value = dataset['type'].lower() if isinstance(dataset['type'], str) else str(dataset['type']).lower()
                if 'dataservice' in type_value or 'data-service' in type_value:
                    object_type = 'dataservice'
                elif 'publicservice' in type_value or 'public-service' in type_value:
                    object_type = 'publicservice'
                elif 'concept' in type_value:
                    object_type = 'concept'
                else:
                    object_type = 'dataset'
            # If no explicit type, use field presence detection
            elif 'endpointUrls' in dataset or 'servesDatasets' in dataset:
                object_type = 'dataservice'
            elif 'channels' in dataset or 'isDescribedAt' in dataset:
                object_type = 'publicservice'
            elif 'conceptType' in dataset and 'distribution' not in dataset and 'landingPage' not in dataset:
                # Only classify as concept if it has conceptType AND doesn't have dataset-specific fields
                object_type = 'concept'
            else:
                # Default to dataset if it has typical dataset fields
                object_type = 'dataset'
                
            dataset_info = {
                'id': dataset.get('guid', '') or dataset.get('id', '') or dataset.get('identifier', ''),
                'guid': dataset.get('guid', '') or dataset.get('id', '') or dataset.get('identifier', ''),
                'title': extract_text(dataset.get('title', '')) or extract_text(dataset.get('name', '')) or extract_text(dataset.get('prefLabel', '')),
                'description': extract_text(dataset.get('description', '')),
                'publisher': extract_publisher_name(dataset.get('publisher', '')) or 
                           extract_publisher_name(dataset.get('organisation', '')) or
                           extract_publisher_name(dataset.get('publisherName', '')),
                'status': extract_status_field(dataset.get('registrationStatus')) or
                          extract_status_field(dataset.get('status')) or
                          extract_status_field(dataset.get('lifeCycleStatus')) or
                          extract_status_field(dataset.get('lifecycleStatus')) or
                          extract_status_field(dataset.get('state')) or
                          extract_status_field(dataset.get('statusText')),
                'publicationLevel': dataset.get('publicationLevel', ''),
                'modified': dataset.get('modified', '') or dataset.get('modificationDate', ''),
                'url': f"https://input.i14y.admin.ch/catalog/datasets/{dataset.get('guid', '') or dataset.get('id', '')}" if (dataset.get('guid') or dataset.get('id')) else '',
                'objectType': object_type,
                'fullObject': dataset  # Store the full object for later use in updates
            }
            
            # Only include if we have at least a title or description
            if dataset_info['title'] or dataset_info['description']:
                datasets.append(dataset_info)
        
        # Apply filters
        filtered_datasets = []
        for dataset in datasets:
            # Filter by object type
            if object_type_filter and dataset['objectType'] != object_type_filter:
                continue
            
            # Filter by organisation
            if organisation_filter:
                # Check if organisation matches (by ID or name)
                publisher_name = dataset.get('publisher', '').lower()
                if organisation_filter.lower() not in publisher_name and dataset.get('organisation_id', '') != organisation_filter:
                    continue
            
            filtered_datasets.append(dataset)
        
        result = {
            'datasets': filtered_datasets,
            'total': len(filtered_datasets),
            'page': page,
            'pageSize': page_size,
            'query': query
        }
        
        logging.info(f"Found {len(datasets)} datasets in I14Y")
        return jsonify(result)
        
    except requests.exceptions.Timeout:
        return jsonify({'error': 'I14Y API request timed out'}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Could not connect to I14Y API'}), 503
    except Exception as e:
        logging.error(f"I14Y search error: {e}", exc_info=True)
        return jsonify({'error': f'Error searching I14Y: {str(e)}'}), 500


@app.route('/get-i14y-object', methods=['POST'])
def get_i14y_object():
    """Fetch a full I14Y object (dataset/dataservice/publicservice/concept) from the public API.
    Used to retrieve complete keyword arrays when the search response omits them."""
    try:
        req = request.get_json(force=True)
        object_id = req.get('objectId') or req.get('object_id')
        object_type = (req.get('objectType') or req.get('object_type') or 'dataset').lower()
        env = (req.get('env') or 'prod').lower()

        if not object_id:
            return jsonify({'error': 'objectId is required'}), 400

        if object_type not in ['dataset', 'dataservice', 'publicservice', 'concept']:
            return jsonify({'error': f'Unsupported objectType: {object_type}'}), 400

        if env not in ['abnahme', 'prod']:
            env = 'prod'

        public_base = {
            'abnahme': 'https://api-a.i14y.admin.ch/api/public/v1',
            'prod': 'https://api.i14y.admin.ch/api/public/v1'
        }[env]

        endpoint_map = {
            'dataset': '/datasets',
            'dataservice': '/dataservices',
            'publicservice': '/publicservices',
            'concept': '/concepts'
        }

        url = f"{public_base}{endpoint_map[object_type]}/{object_id}"

        verify_ssl = NETWORK_ENV != "internal"
        response = requests.get(url, timeout=10, verify=verify_ssl)

        if not response.ok:
            logging.error(f"Failed to fetch I14Y object {object_type}:{object_id} ({env}): HTTP {response.status_code} - {response.text}")
            return jsonify({'error': f'I14Y public API returned {response.status_code}'}), response.status_code

        payload = response.json()
        full_object = payload.get('data', payload)

        return jsonify({'fullObject': full_object})

    except Exception as e:
        logging.error(f"Error fetching I14Y object: {e}", exc_info=True)
        return jsonify({'error': f'Error fetching I14Y object: {str(e)}'}), 500

@app.route('/update-i14y-keywords', methods=['POST'])
def update_i14y_keywords():
    """Update keywords on I14Y via the Partner API using PUT request"""
    try:
        data = request.get_json()
        
        object_id = data.get('objectId') or data.get('object_id')
        object_type = (data.get('objectType') or data.get('object_type') or 'dataset').lower()
        keywords = data.get('keywords', [])
        api_token = data.get('apiToken') or data.get('api_token')
        full_dataset = data.get('fullDataset') or data.get('full_dataset') or {}
        if not isinstance(full_dataset, dict):
            full_dataset = {}
        
        if not object_id:
            return jsonify({'error': 'Object ID is required'}), 400
        
        if not api_token:
            return jsonify({'error': 'API token is required'}), 400
            
        if not api_token.startswith('Bearer '):
            return jsonify({'error': 'API token must start with "Bearer "'}), 400
        
        if not keywords:
            return jsonify({'error': 'At least one keyword is required'}), 400
        
        # Map object types to API endpoints
        endpoint_map = {
            'dataset': f'/datasets/{object_id}',
            'dataservice': f'/dataservices/{object_id}',
            'publicservice': f'/publicservices/{object_id}',
            'concept': f'/concepts/{object_id}'
        }
        
        endpoint = endpoint_map.get(object_type)
        if not endpoint:
            return jsonify({'error': f'Unknown object type: {object_type}'}), 400
        
        # Format keywords according to new I14Y structure
        # New structure: {"label": {"de": "...", "en": "...", "fr": "...", "it": "...", "rm": "..."}, "uri": "..."}
        formatted_keywords = []
        for kw in keywords:
            keyword_obj = {
                "label": {},
                "uri": kw.get('uri', '')
            }
            
            # Handle multilingual_label (from TERMDAT/GEMET)
            if 'multilingual_label' in kw:
                ml = kw['multilingual_label']
                keyword_obj['label'] = {
                    'de': ml.get('de', ''),
                    'en': ml.get('en', ''),
                    'fr': ml.get('fr', ''),
                    'it': ml.get('it', ''),
                    'rm': ml.get('rm', '')
                }
            # Handle plain label string (from Wikidata or fallback)
            elif 'label' in kw:
                label_text = kw['label']
                # Distribute to all languages if we have a plain string
                keyword_obj['label'] = {
                    'de': label_text,
                    'en': label_text,
                    'fr': label_text,
                    'it': label_text,
                    'rm': label_text
                }
            else:
                # Fallback: no label found
                logging.warning(f"Keyword has no label: {kw}")
                continue
                
            formatted_keywords.append(keyword_obj)
        
        if not formatted_keywords:
            return jsonify({'error': 'No valid keywords to update'}), 400
        
        # Prepare the PUT request body
        # We need to send the full object back with updated keywords
        # Get the fullObject from the dataset
        full_object = (
            full_dataset.get('fullObject') or
            full_dataset.get('full_object') or
            data.get('fullObject') or
            data.get('full_object') or {}
        )
        
        # If fullObject is not available, create minimal required structure
        if not full_object:
            logging.warning("fullObject not available in dataset, creating minimal structure")
            full_object = {
                'title': full_dataset.get('title', {}),
                'description': full_dataset.get('description', {}),
                'identifier': full_dataset.get('guid') or full_dataset.get('id'),
            }
        
        # Start with the existing data and update only the keywords field
        update_data = full_object.copy()
        
        # Update keywords in the payload
        update_data['keywords'] = formatted_keywords
        
        # Remove read-only fields that shouldn't be sent in PUT
        # These fields are returned by GET but should not be included in PUT
        readonly_fields = ['id', 'guid', 'publicationLevel', 'publicationLevelProposal', 
                          'registrationStatus', 'registrationStatusProposal',
                          'isLocked', 'codeListEntries']
        for field in readonly_fields:
            update_data.pop(field, None)
        
        # Wrap in data object as required by API
        update_payload = {
            "data": update_data
        }
        
        # Pick Partner API base by env
        env = data.get('env', 'prod').lower()
        if env not in ['abnahme', 'prod']:
            env = 'prod'

        partner_base = {
            'abnahme': 'https://api-a.i14y.admin.ch/api/partner/v1',
            'prod': 'https://api.i14y.admin.ch/api/partner/v1'
        }[env]

        partner_api_url = f'{partner_base}{endpoint}'
        
        headers = {
            'Authorization': api_token,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        logging.info(f"Updating {object_type} {object_id} on I14Y with {len(formatted_keywords)} keywords")
        logging.debug(f"PUT URL: {partner_api_url}")
        logging.debug(f"Keywords payload: {json.dumps(formatted_keywords, indent=2)}")
        
        response = requests.put(
            partner_api_url,
            json=update_payload,
            headers=headers,
            timeout=30
        )
        
        # Check response
        if response.status_code == 204:
            # Success - No Content
            logging.info(f"Successfully updated keywords for {object_type} {object_id}")
            return jsonify({
                'success': True,
                'message': f'Successfully updated {len(formatted_keywords)} keywords',
                'object_id': object_id,
                'object_type': object_type
            })
        elif response.status_code == 401:
            return jsonify({'error': 'Unauthorized - Invalid API token'}), 401
        elif response.status_code == 403:
            return jsonify({'error': 'Forbidden - You do not have permission to update this object'}), 403
        elif response.status_code == 404:
            return jsonify({'error': f'Object not found: {object_type} {object_id}'}), 404
        else:
            # Try to get error details from response
            try:
                error_data = response.json()
                error_msg = error_data.get('title') or error_data.get('detail') or str(error_data)
            except:
                error_msg = response.text or f'HTTP {response.status_code}'
            
            logging.error(f"I14Y API error: {response.status_code} - {error_msg}")
            return jsonify({'error': f'I14Y API error: {error_msg}'}), response.status_code
            
    except requests.exceptions.Timeout:
        logging.error("I14Y API request timed out")
        return jsonify({'error': 'Request to I14Y API timed out'}), 504
    except requests.exceptions.ConnectionError:
        logging.error("Could not connect to I14Y API")
        return jsonify({'error': 'Could not connect to I14Y API'}), 503
    except Exception as e:
        logging.error(f"Error updating keywords on I14Y: {e}", exc_info=True)
        return jsonify({'error': f'Error updating keywords: {str(e)}'}), 500

@app.route('/get-i14y-organisations', methods=['POST'])
def get_i14y_organisations():
    """Fetch organisations from I14Y agents endpoint"""
    try:
        env = request.json.get('env', 'prod').lower()
        if env not in ['abnahme', 'prod']:
            env = 'prod'

        agents_base = {
            'abnahme': "https://input-backend.i14y.a.c.bfs.admin.ch/api/Agent",
            'prod': "https://input-backend.i14y.c.bfs.admin.ch/api/Agent"
        }[env]

        response = requests.get(agents_base, timeout=10)
        
        if not response.ok:
            logging.error(f"Failed to fetch organisations: HTTP {response.status_code}")
            return jsonify({'error': f'Failed to fetch organisations: {response.status_code}'}), response.status_code
        
        data = response.json()
        
        # Extract organisations with all language variants
        organisations = []
        if isinstance(data, list):
            for agent in data:
                if isinstance(agent, dict):
                    # Extract all language variants
                    names = {}
                    if 'name' in agent and isinstance(agent['name'], dict):
                        names = {
                            'de': agent['name'].get('de', ''),
                            'fr': agent['name'].get('fr', ''),
                            'it': agent['name'].get('it', ''),
                            'en': agent['name'].get('en', ''),
                            'rm': agent['name'].get('rm', '')
                        }
                    elif 'name' in agent and isinstance(agent['name'], str):
                        names = {'de': agent['name']}
                    elif 'prefLabel' in agent and isinstance(agent['prefLabel'], dict):
                        names = {
                            'de': agent['prefLabel'].get('de', ''),
                            'fr': agent['prefLabel'].get('fr', ''),
                            'it': agent['prefLabel'].get('it', ''),
                            'en': agent['prefLabel'].get('en', ''),
                            'rm': agent['prefLabel'].get('rm', '')
                        }
                    
                    # Get primary display name (prefer German)
                    display_name = names.get('de') or names.get('en') or names.get('fr') or names.get('it') or ''
                    
                    if display_name:
                        organisations.append({
                            'id': agent.get('id', ''),
                            'name': display_name,
                            'names': names  # All language variants
                        })
        
        # Sort alphabetically
        organisations.sort(key=lambda x: x['name'].lower())
        
        return jsonify({'organisations': organisations})
        
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request to I14Y API timed out'}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Could not connect to I14Y API'}), 503
    except Exception as e:
        logging.error(f"Error fetching organisations: {e}", exc_info=True)
        return jsonify({'error': f'Error fetching organisations: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
