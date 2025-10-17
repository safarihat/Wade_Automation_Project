import os
import json
import logging
import time
import datetime
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REGION_URLS = {
    "Auckland": "https://mapspublic.aucklandcouncil.govt.nz/arcgis3/rest/services/NonCouncil/LAWA/MapServer/0",
    "Bay of Plenty": "https://gis.boprc.govt.nz/server2/rest/services/emar/MonitoringSiteReferenceData/MapServer/0",
    "Canterbury": "https://gis.ecan.govt.nz/arcgis/rest/services/emar/MonitoringSiteReferenceData/MapServer/0",
    "Hawke's Bay": "https://services2.arcgis.com/0EFvUOzsI97G4Vdg/arcgis/rest/services/MonitoringSiteReferenceData/FeatureServer/0",
    "Horizons": "https://maps.horizons.govt.nz/arcgis/rest/services/emar/MonitoringSiteReferenceData/MapServer/0",
    "Northland": "https://nrcmaps.nrc.govt.nz/arcgis/rest/services/emar/MonitoringSiteReferenceData/MapServer/0",
    "Otago": "https://maps.orc.govt.nz/arcgis/rest/services/Public/Environmental_Monitoring/MapServer/0",
    "Southland": "https://maps.es.govt.nz/server/rest/services/emar/MonitoringSiteReferenceData/MapServer/0",
    "Taranaki": "https://maps.trc.govt.nz/arcgis/rest/services/LocalMaps/SamplingAndRecordingSites/MapServer/0",
    "Waikato": "https://data.waikatoregion.govt.nz/arcgis/rest/services/LAWA/MonitoringSiteReferenceData/MapServer/0",
    "West Coast": "https://maps.wcrc.govt.nz/arcgis/rest/services/emar/MonitoringSiteReferenceData/MapServer/0",
    "Gisborne": "https://maps.gdc.govt.nz/arcgis/rest/services/LAWA/MonitoringSiteReferenceData/MapServer/0",
    "Marlborough": "https://maps.marlborough.govt.nz/arcgis/rest/services/LAWA/MonitoringSiteReferenceData/MapServer/0",
    "Tasman": "https://maps.tasman.govt.nz/arcgis/rest/services/LAWA/MonitoringSiteReferenceData/MapServer/0",
    "Greater Wellington": "https://mapping.gw.govt.nz/arcgis/rest/services/emar/MonitoringSiteReferenceData/MapServer/0",
}

CACHE_DIR = os.path.join(os.getcwd(), 'doc_generator', 'data', 'region_cache')

def get_field_name(fields, candidates):
    field_map = {f['name'].lower(): f['name'] for f in fields}
    for candidate in candidates:
        if candidate.lower() in field_map:
            return field_map[candidate.lower()]
    return None

def fetch_sites_for_region(region, base_url, retries=3, delay=5):
    logging.info(f"  Fetching sites for {region}...")
    for attempt in range(retries):
        try:
            meta_response = requests.get(base_url, params={'f': 'json'}, timeout=60)
            meta_response.raise_for_status()
            metadata = meta_response.json()

            fields = metadata.get('fields', [])
            lawa_id_field = get_field_name(fields, ['LAWASiteID', 'LawaSiteID', 'LawaId'])
            site_name_field = get_field_name(fields, ['SiteName', 'SiteID', 'CouncilSiteID'])

            if not lawa_id_field:
                logging.error(f"    -> Could not find a LAWA ID field for {region}. Skipping.")
                return None

            query_url = f"{base_url}/query"
            all_features = []
            offset = 0
            max_records = metadata.get('maxRecordCount', 1000)
            if region == "Southland":
                max_records = 2000
            supports_pagination = metadata.get('advancedQueryCapabilities', {}).get('supports_pagination', False)

            while True:
                params = {'where': '1=1', 'outFields': '*' if site_name_field else lawa_id_field, 'returnGeometry': 'true', 'outSR': '4326', 'f': 'json'}
                if supports_pagination:
                    params['resultOffset'] = offset
                    params['resultRecordCount'] = max_records
                
                query_response = requests.get(query_url, params=params, timeout=60)
                query_response.raise_for_status()
                data = query_response.json()
                features = data.get('features', [])
                if not features:
                    break

                all_features.extend(features)
                if not data.get('exceededTransferLimit', False) and len(features) < max_records:
                    break
                offset += len(features)

            sites = []
            for feature in all_features:
                attrs = feature.get('attributes', {})
                geom = feature.get('geometry', {})
                lawa_id = attrs.get(lawa_id_field)
                if not lawa_id:
                    continue

                site_name = attrs.get(site_name_field, "") if site_name_field else ""
                lat = geom.get('y')
                lon = geom.get('x')

                if not all([lawa_id, lat, lon]):
                    logging.warning(f"    -> Skipping feature in {region} due to missing data (ID: {lawa_id}, Lat: {lat}, Lon: {lon})")
                    continue

                sites.append({"LawaId": lawa_id, "SiteName": site_name, "Lat": lat, "Long": lon, "Region": region})
            
            logging.info(f"    -> Successfully fetched {len(sites)} sites for {region}.")
            return sites

        except requests.RequestException as e:
            logging.error(f"    -> Attempt {attempt + 1}/{retries} failed for {region}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logging.error(f"    -> All attempts failed for {region}. Skipping.")
                return None
        except Exception as e:
            logging.error(f"    -> An unexpected error occurred for {region}: {e}", exc_info=True)
            return None

def aggregate_regional_data(output_path):
    logging.info("--- Aggregating Regional Data ---")
    all_sites = []
    seen_lawa_ids = set()
    successful_regions = []

    for region_file in os.listdir(CACHE_DIR):
        if region_file.endswith('.json'):
            region_name = region_file.replace('_sites.json', '')
            try:
                with open(os.path.join(CACHE_DIR, region_file), 'r') as f:
                    region_sites = json.load(f)
                    for site in region_sites:
                        if site['LawaId'] not in seen_lawa_ids:
                            all_sites.append(site)
                            seen_lawa_ids.add(site['LawaId'])
                    successful_regions.append(region_name)
            except (json.JSONDecodeError, IOError) as e:
                logging.error(f"Could not read or parse {region_file}: {e}")

    all_sites.sort(key=lambda x: x['LawaId'])
    
    final_data = {
        "last_updated_utc": datetime.datetime.utcnow().isoformat(),
        "sites": all_sites
    }

    with open(output_path, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    logging.info(f"Successfully aggregated data from {len(successful_regions)} regions: {successful_regions}")
    logging.info(f"Total unique sites collected: {len(all_sites)}")
    logging.info(f"Final aggregated data saved to: {output_path}")

def main():
    logging.info("--- Starting LAWA Monitoring Site Collection ---")
    os.makedirs(CACHE_DIR, exist_ok=True)
    failed_regions = []

    for region, url in REGION_URLS.items():
        region_sites = fetch_sites_for_region(region, url)
        if region_sites is not None:
            cache_path = os.path.join(CACHE_DIR, f"{region}_sites.json")
            with open(cache_path, 'w') as f:
                json.dump(region_sites, f, indent=2)
            logging.info(f"    -> Cached data for {region} to {cache_path}")
        else:
            failed_regions.append(region)

    if failed_regions:
        logging.warning(f"\nCould not fetch data for the following regions: {failed_regions}")
    
    # Aggregate all cached files into the final output
    final_output_path = os.path.join(os.getcwd(), 'doc_generator', 'data', 'lawa_sites.json')
    aggregate_regional_data(final_output_path)

if __name__ == "__main__":
    main()
