# DCAT-AP CH Keyword Generator

A Flask web application that generates DCAT-AP CH compliant keywords by searching through multiple controlled vocabularies following the specified priority cascade.

## Features

- **Priority-based keyword search** following DCAT-AP CH specifications:
  1. TERMDAT (Swiss federal terminology database)
  2. GEMET (Environmental terminology)
  3. Wikidata (General knowledge base)
  4. Literal keywords (fallback)

- **Modern web interface** with responsive design
- **Real-time search** with loading indicators
- **Structured results** showing source, URI, and descriptions
- **Select and download keywords**: Click on generated keyword tiles to select them and download the selected keywords in JSON format.

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python app.py
   ```

3. Open your browser and go to `http://localhost:5000`

## Usage

1. Enter a search term in the input field.
2. Click "Generate Keywords" or press Enter.
3. Review the generated keywords sorted by priority.
4. Click on keyword tiles to select them.
5. Use the "Download Selected Keywords" button to download the selected keywords in JSON format.
6. Use the provided URIs for DCAT-AP CH compliance

## Implementation Notes

- **TERMDAT**: Currently uses a placeholder implementation as TERMDAT doesn't have a public API. In a production environment, you would need to implement web scraping or use their specific API if available.

- **GEMET**: Uses the GEMET API for environmental terminology searches. The current implementation includes a mock response structure that should be adapted based on the actual API response format.

- **Wikidata**: Implements the Wikidata API for entity searches, returning proper Wikidata URIs.

- **Literal fallback**: When no controlled vocabulary matches are found, the system falls back to literal keywords.

## API Endpoints

- `GET /` - Main application interface
- `POST /search` - Keyword search endpoint
  - Request: `{"query": "search term"}`
  - Response: `{"query": "...", "keywords": [...], "total": n}`

## Future Enhancements

- Implement proper TERMDAT API integration
- Add caching for frequently searched terms **(in progress)**
- Include multi-language support
- Add export functionality for different formats
- Implement user preference settings for keyword sources
- **Display search history** in the web interface
