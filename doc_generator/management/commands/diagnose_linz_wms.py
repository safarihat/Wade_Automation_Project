import requests
import xml.etree.ElementTree as ET
from django.core.management.base import BaseCommand
from django.conf import settings

class Command(BaseCommand):
    help = "Diagnoses issues with LINZ Web Map Service (WMS) requests by checking capabilities and testing GetMap calls."

    def handle(self, *args, **options):
        api_key = settings.LINZ_API_KEY
        if not api_key:
            self.stdout.write(self.style.ERROR("LINZ_API_KEY is not set in your environment. Cannot proceed."))
            return
        
        # The WMS endpoint for NZTopo50 on data.linz.govt.nz.
        base_wms_url = "https://data.linz.govt.nz/services/wms/"
        target_layer_name = 'layer-50767' # The layer ID for NZTopo50 on data.linz.govt.nz WMS.
        target_crs = 'EPSG:2193'

        # --- 1. Fetch and Parse GetCapabilities ---
        self.stdout.write(self.style.HTTP_INFO(f"Fetching WMS Capabilities from {base_wms_url}..."))
        capabilities_url = f"{base_wms_url}?service=WMS&request=GetCapabilities&key={api_key}"
        try:
            response = requests.get(capabilities_url, timeout=30)
            response.raise_for_status()
            self.stdout.write(self.style.SUCCESS("Successfully fetched GetCapabilities XML."))
        except requests.exceptions.RequestException as e:
            self.stdout.write(self.style.ERROR(f"Failed to fetch GetCapabilities: {e}"))
            return

        # --- 2. Analyze Capabilities XML ---
        self.stdout.write("\n" + self.style.HTTP_INFO("--- Analyzing Capabilities ---"))
        try:
            # Register namespaces to correctly parse the XML
            # Note: WMTS and WMS have different namespaces. This is for WMS.
            namespaces = {'wms': 'http://www.opengis.net/wms'}
            root = ET.fromstring(response.content)


            # Find the target layer
            target_layer_node = None
            all_layers = root.findall('.//wms:Layer[wms:Name]', namespaces)
            for layer in all_layers:
                name_node = layer.find('wms:Name', namespaces)
                if name_node is not None and name_node.text == target_layer_name:
                    target_layer_node = layer
                    break
            
            if target_layer_node is None:
                self.stdout.write(self.style.ERROR(f"Validation failed: Layer '{target_layer_name}' not found in GetCapabilities response."))
                return
            
            self.stdout.write(self.style.SUCCESS(f"Validation successful: Found layer '{target_layer_name}'."))

            # Find the bounding box for the target CRS
            # WMS 1.3.0 uses 'CRS', while 1.1.1 uses 'SRS'. We check for both.
            supported_crs_nodes = target_layer_node.findall('.//wms:CRS', namespaces) or target_layer_node.findall('.//wms:SRS', namespaces)
            supported_crs_list = {node.text for node in supported_crs_nodes}
            self.stdout.write(f"Layer supports the following CRS/SRS: {', '.join(supported_crs_list)}")

            bbox_node = target_layer_node.find(f".//wms:BoundingBox[@CRS='{target_crs}']", namespaces) or target_layer_node.find(f".//wms:BoundingBox[@SRS='{target_crs}']", namespaces)

            if bbox_node is None:
                self.stdout.write(self.style.ERROR(f"Validation failed: Layer '{target_layer_name}' does not support CRS '{target_crs}'."))
                return

            bbox = {
                'minx': bbox_node.get('minx'),
                'miny': bbox_node.get('miny'),
                'maxx': bbox_node.get('maxx'),
                'maxy': bbox_node.get('maxy'),
            }
            bbox_str = f"{bbox['minx']},{bbox['miny']},{bbox['maxx']},{bbox['maxy']}"
            self.stdout.write(self.style.SUCCESS(f"Found Bounding Box for {target_crs}: {bbox_str}"))

        except ET.ParseError as e:
            self.stdout.write(self.style.ERROR(f"Failed to parse XML: {e}"))
            return

        # --- 3. Systematically Test GetMap Requests ---
        self.stdout.write("\n" + self.style.HTTP_INFO("--- Testing GetMap Requests ---"))
        
        test_matrix = [
            {'version': '1.1.1', 'param_name': 'srs'},
            {'version': '1.3.0', 'param_name': 'crs'},
            # --- Negative tests for diagnosis ---
            {'version': '1.1.1', 'param_name': 'crs'},
            {'version': '1.3.0', 'param_name': 'srs'},
        ]

        success = False
        for test in test_matrix:
            version = test['version']
            param_name = test['param_name']
            
            self.stdout.write(f"\nAttempting with version={version}, param='{param_name}'...")

            # Use the BBOX from your original request for a direct comparison
            user_bbox = "1215915.84427322,4861507.161655702,1216915.84427322,4862507.161655702"

            params = {
                'key': api_key,
                'service': 'WMS',
                'request': 'GetMap',
                'layers': target_layer_name,
                'styles': '',
                'format': 'image/png',
                'transparent': 'true',
                'version': version,
                'width': 800,
                'height': 800,
                param_name: target_crs,
                'bbox': user_bbox,
            }

            try:
                # Construct the URL with parameters
                query_string = "&".join([f"{k}={v}" for k, v in params.items() if v is not None and v != ''])
                full_url = f"{base_wms_url}?{query_string}"
                self.stdout.write(f"  URL: {full_url}")

                res = requests.get(base_wms_url, params=params, timeout=30)

                self.stdout.write(f"  Status Code: {res.status_code}")
                
                content_type = res.headers.get('Content-Type', '')
                self.stdout.write(f"  Content-Type: {content_type}")

                if res.status_code == 200 and 'image/png' in content_type:
                    self.stdout.write(self.style.SUCCESS("  SUCCESS: Received a valid PNG image."))
                    success = True
                elif 'xml' in content_type:
                    self.stdout.write(self.style.WARNING("  FAILURE: Received an XML error response from the server:"))
                    self.stdout.write(res.text[:500] + "...") # Print first 500 chars of error
                else:
                    self.stdout.write(self.style.ERROR(f"  FAILURE: Received an unexpected response (Status {res.status_code})."))

            except requests.exceptions.RequestException as e:
                self.stdout.write(self.style.ERROR(f"  Request failed: {e}"))

        # --- 4. Final Report ---
        self.stdout.write("\n" + self.style.HTTP_INFO("--- Diagnostic Report ---"))
        if success:
            self.stdout.write(self.style.SUCCESS(
                "A successful combination was found! Please check the logs above for the correct parameters.\n"
                "The most likely cause of your original error was using 'srs' with WMS version 1.3.0, or 'crs' with 1.1.1."
            ))
        else:
            self.stdout.write(self.style.ERROR(
                "All tested combinations failed. This suggests the issue might be one of the following:\n"
                "1. The API Key is invalid, disabled, or has not been activated for the Basemaps service.\n"
                "2. There is a network issue (firewall, proxy) blocking the connection to basemaps.linz.govt.nz.\n"
                "3. The LINZ service is temporarily unavailable.\n"
            ))