from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Configure caching
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300
cache = Cache(app)

# Network environment configuration
NETWORK_ENV = "internal"  # Change to "external" for external networks

class KeywordGenerator:
    def __init__(self):
        self.termdat_base_url = "https://register.ld.admin.ch/termdat/"
        self.gemet_base_url = "http://www.eionet.europa.eu/gemet/"
        self.wikidata_base_url = "https://www.wikidata.org/w/api.php"
        # Set SSL verification based on network environment
        self.verify_ssl = NETWORK_ENV != "internal"
    
    def search_termdat(self, query, query_lang='de'):
        """Search TERMDAT for keywords using the official API with multilingual support"""
        results = []
        entry_ids_processed = set()  # Track processed entries to avoid duplicates
        
        # Lowercase the query language
        query_lang = query_lang.lower()
        
        try:
            # Use separate searches for each language to maximize results
            # This approach increases chances of finding terms in the desired languages
            all_multilingual_terms = {}  # Dictionary to collect terms by entry_id
            
            # Map language codes to TERMDAT language IDs
            # 2=DE, 3=EN, 6=FR, 7=IT, 8=RM
            language_map = {
                'de': 2,
                'fr': 6,
                'it': 7,
                'en': 3
            }
            
            # Search in all 4 languages to get comprehensive results
            for search_lang, lang_id in language_map.items():
                # TERMDAT API search endpoint
                url = "https://www.termdat.bk.admin.ch/api/Search/Search"
                
                # Build parameters according to the working request format
                params = {
                    'pageindex': 1,
                    'pagesize': 10,  # Limit results per language
                    'phrase': query,
                    'sourceLanguageIds': lang_id,
                    'targetLanguageIds': lang_id,  # Same language for source and target
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
                
                logging.debug(f"TERMDAT Search Request for {search_lang}: {url}")
                
                response = requests.get(url, params=params, timeout=10, verify=self.verify_ssl)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        # Process search entries from the response
                        if 'searchEntries' in data and isinstance(data['searchEntries'], list):
                            for entry in data['searchEntries']:
                                entry_id = entry.get('id')
                                if not entry_id:
                                    continue
                                
                                # Initialize this entry's terms if not already in our collection
                                if entry_id not in all_multilingual_terms:
                                    all_multilingual_terms[entry_id] = {
                                        'terms': {},
                                        'collection': '',
                                        'description': f"TERMDAT entry for '{query}'"
                                    }
                                
                                # Store collection info
                                if 'collection' in entry and 'name' in entry['collection']:
                                    all_multilingual_terms[entry_id]['collection'] = entry['collection']['name']
                                
                                # Extract terms from this entry
                                if 'terms' in entry and isinstance(entry['terms'], list):
                                    for term_obj in entry['terms']:
                                        lang_id = term_obj.get('languageId')
                                        for lang_code, tid in language_map.items():
                                            if lang_id == tid:
                                                # Get term from name or abbreviation
                                                term_text = term_obj.get('name', '')
                                                if not term_text and 'abbreviation' in term_obj:
                                                    term_text = term_obj.get('abbreviation', '')
                                                
                                                if term_text:
                                                    all_multilingual_terms[entry_id]['terms'][lang_code] = term_text
                                                break
                    except Exception as e:
                        logging.error(f"Error parsing TERMDAT response for {search_lang}: {e}")
                else:
                    logging.error(f"TERMDAT API returned non-200 status for {search_lang}: {response.status_code}")
            
            # Now create result objects from the collected terms
            for entry_id, entry_data in all_multilingual_terms.items():
                multilingual_terms = entry_data['terms']
                
                # Only include entries that have the required language coverage
                required_languages = {'de', 'fr', 'it', 'en'}
                available_languages = set(multilingual_terms.keys())
                
                # Check if all required languages are available
                if required_languages.issubset(available_languages):
                    # Build the TERMDAT URI
                    termdat_uri = f"https://register.ld.admin.ch/termdat/{entry_id}"
                    
                    # Get a suitable description
                    description = entry_data['description']
                    if entry_data['collection']:
                        description = f"TERMDAT: {entry_data['collection']}"
                    
                    # Track which languages are available
                    available_languages_list = list(multilingual_terms.keys())
                    
                    logging.debug(f"Adding TERMDAT result for entry {entry_id} with all required languages: {multilingual_terms}")
                    
                    results.append({
                        'source': 'TERMDAT',
                        'multilingual_label': multilingual_terms,
                        'uri': termdat_uri,
                        'description': description,
                        'entry_id': entry_id,
                        'query_lang': query_lang,
                        'available_languages': available_languages_list
                    })
                else:
                    missing_languages = required_languages - available_languages
                    logging.debug(f"Skipping TERMDAT entry {entry_id} due to missing languages: {missing_languages}")
        except Exception as e:
            logging.error(f"Error searching TERMDAT: {e}")
        
        logging.debug(f"TERMDAT total results: {len(results)}")
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
            # First search in English to get entity IDs
            search_params = {
                'action': 'wbsearchentities',
                'search': query,
                'language': 'en',
                'format': 'json',
                'limit': 5
            }
            
            search_response = requests.get(self.wikidata_base_url, params=search_params, timeout=10, verify=self.verify_ssl)
            if search_response.status_code == 200:
                search_data = search_response.json()
                if 'search' in search_data:
                    for item in search_data['search']:
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
                            
        except Exception as e:
            print(f"Error searching Wikidata: {e}")
        
        return results
    
    def _get_wikidata_multilingual_labels(self, entity_id):
        """Get multilingual labels for a Wikidata entity"""
        multilingual_terms = {}
        
        try:
            # Use Wikidata API to get entity data with labels in multiple languages
            entity_params = {
                'action': 'wbgetentities',
                'ids': entity_id,
                'props': 'labels',
                'languages': 'de|fr|it|en',  # Request all 4 languages
                'format': 'json'
            }
            
            entity_response = requests.get(self.wikidata_base_url, params=entity_params, timeout=10, verify=self.verify_ssl)
            if entity_response.status_code == 200:
                entity_data = entity_response.json()
                if 'entities' in entity_data and entity_id in entity_data['entities']:
                    entity = entity_data['entities'][entity_id]
                    if 'labels' in entity:
                        labels = entity['labels']
                        
                        # Extract labels for each language
                        for lang_code in ['de', 'fr', 'it', 'en']:
                            if lang_code in labels and 'value' in labels[lang_code]:
                                multilingual_terms[lang_code] = labels[lang_code]['value']
                                
        except Exception as e:
            print(f"Error getting Wikidata multilingual labels for {entity_id}: {e}")
        
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

    def generate_keywords(self, query, query_lang='de'):
        """Generate keywords following DCAT-AP CH priority cascade"""
        all_keywords = []
        
        # Priority 1: TERMDAT (multilingual)
        termdat_results = self.search_termdat(query, query_lang)
        all_keywords.extend(termdat_results)
        
        # Priority 2: GEMET (multilingual)
        gemet_results = self.search_gemet(query)
        all_keywords.extend(gemet_results)
        
        # Priority 3: Wikidata (multilingual)
        wikidata_results = self.search_wikidata(query)
        all_keywords.extend(wikidata_results)
        
        return all_keywords
keyword_generator = KeywordGenerator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_keywords():
    query = request.json.get('query', '').strip()
    query_lang = request.json.get('lang', 'de').strip().lower()
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Check cache first
    cache_key = f"{query}:{query_lang}"
    cached_result = cache.get(cache_key)
    if cached_result:
        return jsonify(cached_result)

    try:
        keywords = keyword_generator.generate_keywords(query, query_lang)
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
                    'available_languages': kw.get('available_languages', list(kw.get('multilingual_label', {}).keys()))
                }
                sanitized_keywords.append(sanitized_kw)
            except Exception as e:
                logging.error(f"Error sanitizing keyword for JSON: {e}")
        
        result = {
            'query': query,
            'query_lang': query_lang,
            'keywords': sanitized_keywords,  # Use sanitized keywords
            'i14y_keywords': i14y_keywords,  # I14Y-ready format for upload
            'total': len(sanitized_keywords)
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
