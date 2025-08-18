from flask import Flask, render_template, request, jsonify
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
        """Search GEMET for keywords with multilingual support"""
        results = []
        
        # GEMET language codes mapping
        gemet_languages = {
            'de': 'German',
            'fr': 'French', 
            'it': 'Italian',
            'en': 'English'
        }
        
        try:
            # First, search for concepts in English to get concept URIs
            search_url = "https://www.eionet.europa.eu/gemet/getConceptsMatchingKeyword"
            search_params = {
                'keyword': query,
                'search_mode': 'auto',
                'thesaurus_uri': 'http://www.eionet.europa.eu/gemet/concept/',
                'language': 'en'
            }
            
            search_response = requests.get(search_url, params=search_params, timeout=10, verify=self.verify_ssl)
            
            if search_response.status_code == 200:
                # Try to parse the response - GEMET might return XML or JSON
                content_type = search_response.headers.get('content-type', '').lower()
                
                if 'json' in content_type:
                    # Handle JSON response
                    concepts = search_response.json()
                    if isinstance(concepts, list):
                        for concept in concepts[:5]:  # Limit to 5 concepts
                            concept_uri = concept.get('uri', '')
                            if concept_uri:
                                multilingual_terms = self._get_gemet_multilingual_labels(concept_uri)
                                if multilingual_terms:
                                    results.append({
                                        'source': 'GEMET',
                                        'multilingual_label': multilingual_terms,
                                        'uri': concept_uri,
                                        'description': f'Environmental concept from GEMET thesaurus'
                                    })
                else:
                    # For now, create a fallback multilingual entry
                    # This would need to be improved with actual GEMET API parsing
                    multilingual_terms = {}
                    for lang_code in ['de', 'fr', 'it', 'en']:
                        multilingual_terms[lang_code] = f"GEMET concept for '{query}' ({gemet_languages[lang_code]})"
                    
                    results.append({
                        'source': 'GEMET',
                        'multilingual_label': multilingual_terms,
                        'uri': f'http://www.eionet.europa.eu/gemet/concept/{hash(query) % 10000}',
                        'description': f'Environmental concept related to {query}'
                    })
                    
        except Exception as e:
            print(f"Error searching GEMET: {e}")
        
        return results
    
    def _get_gemet_multilingual_labels(self, concept_uri):
        """Get multilingual labels for a GEMET concept"""
        multilingual_terms = {}
        
        for lang_code in ['de', 'fr', 'it', 'en']:
            try:
                # GEMET API to get preferred label in specific language
                label_url = "https://www.eionet.europa.eu/gemet/getConcept"
                label_params = {
                    'concept_uri': concept_uri,
                    'language': lang_code
                }
                
                label_response = requests.get(label_url, params=label_params, timeout=5, verify=self.verify_ssl)
                if label_response.status_code == 200:
                    # Parse the response to extract the label
                    # This would need to be adapted based on actual GEMET API response format
                    label_data = label_response.text
                    # For now, use a simplified approach
                    if label_data and len(label_data.strip()) > 0:
                        # Extract label from response (would need proper parsing)
                        multilingual_terms[lang_code] = f"GEMET term ({lang_code.upper()})"
                        
            except Exception as e:
                print(f"Error getting GEMET label for {lang_code}: {e}")
        
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
        """Convert multilingual keywords to I14Y expected format"""
        i14y_keywords = []
        
        for keyword in keywords:
            if 'multilingual_label' in keyword:
                multilingual_labels = keyword['multilingual_label']
                
                # Create I14Y keyword object with all available languages
                i14y_keyword = {}
                
                # Add available language variants
                for lang_code in ['de', 'fr', 'it', 'en']:
                    if lang_code in multilingual_labels:
                        i14y_keyword[lang_code] = multilingual_labels[lang_code]
                
                # Only add if we have at least one language
                if i14y_keyword:
                    # Add metadata
                    i14y_keyword['_source'] = keyword.get('source', '')
                    i14y_keyword['_uri'] = keyword.get('uri', '')
                    i14y_keyword['_description'] = keyword.get('description', '')
                    i14y_keywords.append(i14y_keyword)
            else:
                # Fallback for old format (single label)
                label = keyword.get('label', '')
                if label:
                    i14y_keyword = {
                        'de': label,  # Default to German if language not specified
                        '_source': keyword.get('source', ''),
                        '_uri': keyword.get('uri', ''),
                        '_description': keyword.get('description', '')
                    }
                    i14y_keywords.append(i14y_keyword)
        
        return i14y_keywords

    def generate_keywords(self, query, query_lang='de', include_synonyms=True):
        """Generate keywords following DCAT-AP CH priority cascade with exact match prioritization and synonym support"""
        all_keywords = []
        
        logging.info(f"Starting keyword generation for: '{query}' in language: {query_lang}")
        if include_synonyms:
            synonyms = self._get_synonym_queries(query)
            logging.info(f"Including synonyms: {synonyms}")
        
        # Priority 1: TERMDAT (multilingual) - with synonym support
        termdat_results = self.search_termdat(query, query_lang, include_synonyms)
        all_keywords.extend(termdat_results)
        
        # Priority 2: GEMET (multilingual) - TODO: add synonym support
        gemet_results = self.search_gemet(query)
        all_keywords.extend(gemet_results)
        
        # Priority 3: Wikidata (multilingual) - TODO: add synonym support
        wikidata_results = self.search_wikidata(query)
        all_keywords.extend(wikidata_results)
        
        # Sort all results: exact matches first, then by synonym priority, then by source priority
        def sort_key(result):
            query_lower = query.lower()
            multilingual_terms = result.get('multilingual_label', {})
            
            # Check for exact match in any language
            has_exact_match = any(
                term.lower() == query_lower 
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
            # exact_match_priority: 0 for exact matches (highest priority), 1 for others
            # synonym_priority: 0 for original query results, 1 for synonym results
            # source_priority: 0 for TERMDAT (highest), 1 for GEMET, 2 for Wikidata
            # negative_language_count: more languages = higher priority
            return (
                0 if has_exact_match else 1,
                1 if is_synonym_result else 0,
                source_rank,
                -len(result.get('available_languages', []))
            )
        
        all_keywords.sort(key=sort_key)
        
        # Limit final results to 10
        return all_keywords[:10]
keyword_generator = KeywordGenerator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_keywords():
    query = request.json.get('query', '').strip()
    query_lang = request.json.get('lang', 'de').strip().lower()
    include_synonyms = request.json.get('include_synonyms', True)  # Default to True
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Check cache first (include synonyms in cache key)
    cache_key = f"{query}:{query_lang}:{include_synonyms}"
    cached_result = cache.get(cache_key)
    if cached_result:
        return jsonify(cached_result)

    try:
        keywords = keyword_generator.generate_keywords(query, query_lang, include_synonyms)
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
                    'found_via_query': kw.get('found_via_query', query)
                }
                sanitized_keywords.append(sanitized_kw)
            except Exception as e:
                logging.error(f"Error sanitizing keyword for JSON: {e}")
        
        result = {
            'query': query,
            'query_lang': query_lang,
            'include_synonyms': include_synonyms,
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
        
        # Step 4: Find ALL keyword arrays in the dataset and merge them
        existing_keywords = []
        
        # Check for keywords at the top level of dataset_data
        if 'keywords' in dataset_data:
            if isinstance(dataset_data['keywords'], list):
                existing_keywords.extend(dataset_data['keywords'])
        
        # Check for keywords in distributions (if they exist there)
        if 'distributions' in dataset_data:
            for i, dist in enumerate(dataset_data['distributions']):
                if isinstance(dist, dict) and 'keywords' in dist:
                    if isinstance(dist['keywords'], list):
                        existing_keywords.extend(dist['keywords'])
                
        # Create a set of existing keyword strings to avoid duplicates
        existing_keyword_strings = set()
        for kw in existing_keywords:
            # Use German text as the key for duplicate detection
            if isinstance(kw, dict) and 'de' in kw:
                existing_keyword_strings.add(kw['de'].lower())
            else:
                print(f"DEBUG: Skipping invalid keyword: {kw}")
        
        # Add new keywords if they don't already exist
        keywords_added = 0
        for new_kw in new_keywords:
            if isinstance(new_kw, dict) and 'de' in new_kw:
                if new_kw['de'].lower() not in existing_keyword_strings:
                    existing_keywords.append(new_kw)
                    existing_keyword_strings.add(new_kw['de'].lower())
                    keywords_added += 1
        
        # Update the dataset with the merged keywords - IMPORTANT: Set only once!
        dataset_data['keywords'] = existing_keywords
        
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
                'total_keywords': len(existing_keywords),
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
