
import requests
from pyproj import Transformer
from collections import Counter

class ArcGISService:
    """
    A service to query ArcGIS REST APIs for catchment degradation data.
    """
    BASE_URL = "https://maps.es.govt.nz/server/rest/services/Public/WaterAndLand/MapServer"
    LAYERS = {
        'TP': {'id': 8, 'field': 'Degradation_TP'},
        'SuspendedSediment': {'id': 23, 'field': 'Degradation_SuspendedSediment'},
        'Ecoli': {'id': 24, 'field': 'Degradation_Ecoli'},
        'MCI': {'id': 25, 'field': 'Degradation_MCI'},
        'TN': {'id': 30, 'field': 'Degradation_TN'},
    }
    # WGS 84 to NZTM2000
    TRANSFORMER = Transformer.from_crs("epsg:4326", "epsg:2193", always_xy=True)

    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon
        self.nztm_x, self.nztm_y = self.TRANSFORMER.transform(lon, lat)

    def get_catchment_data(self):
        """
        Queries all layers and aggregates the results.
        """
        all_attributes = []
        for layer_key, layer_info in self.LAYERS.items():
            attributes = self._query_layer(layer_info['id'], layer_info['field'])
            if attributes:
                for attr in attributes:
                    attr['layer'] = layer_key
                all_attributes.extend(attributes)

        if not all_attributes:
            return None

        return self._aggregate_results(all_attributes)

    def _query_layer(self, layer_id, degradation_field):
        """
        Queries a single ArcGIS layer.
        """
        url = f"{self.BASE_URL}/{layer_id}/query"
        params = {
            'geometry': f"{self.nztm_x},{self.nztm_y}",
            'geometryType': 'esriGeometryPoint',
            'inSR': 2193,
            'spatialRel': 'esriSpatialRelIntersects',
            'outFields': f'nzsegment,{degradation_field},shape_area',
            'returnGeometry': 'false',
            'f': 'json'
        }
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if 'features' in data and data['features']:
                return [f['attributes'] for f in data['features']]
        except requests.RequestException as e:
            print(f"Error querying layer {layer_id}: {e}")
            return None
        return None

    def _aggregate_results(self, all_attributes):
        """
        Aggregates results from all layers to find the primary catchment
        and summarize degradation.
        """
        if not all_attributes:
            return None

        # Find the most common nzsegment
        nzsegment_counts = Counter(attr['nzsegment'] for attr in all_attributes if 'nzsegment' in attr)
        if not nzsegment_counts:
            return None
        
        primary_nzsegment = nzsegment_counts.most_common(1)[0][0]

        # Get area and degradation statuses for the primary catchment
        total_area_m2 = 0
        degradation_summary = {}

        # Find the first attribute that matches the primary segment to get the area
        primary_segment_attributes = [attr for attr in all_attributes if attr.get('nzsegment') == primary_nzsegment]

        if primary_segment_attributes:
            # Get the area from the first matching attribute that has it
            for attr in primary_segment_attributes:
                if 'shape_area' in attr and attr['shape_area'] is not None:
                    total_area_m2 = attr['shape_area']
                    break # Stop after finding the first valid area

            # Now, collect all degradation statuses for the primary segment
            for attr in primary_segment_attributes:
                layer = attr['layer']
                degradation_field = self.LAYERS[layer]['field']
                if degradation_field in attr and attr[degradation_field]:
                    degradation_summary[layer] = attr[degradation_field]

        total_area_ha = round(total_area_m2 / 10000, 2) if total_area_m2 > 0 else 0

        return {
            "primary_nzsegment": primary_nzsegment,
            "area_ha": total_area_ha,
            "degradation_summary": degradation_summary,
            "raw_attributes": all_attributes
        }

# Example usage:
if __name__ == '__main__':
    # Example farm location from the prompt
    lat, lon = -46.28693651, 168.02011830
    
    service = ArcGISService(lat, lon)
    catchment_data = service.get_catchment_data()

    if catchment_data:
        print("--- Aggregated Catchment Data ---")
        print(f"Primary NZ Segment: {catchment_data['primary_nzsegment']}")
        print(f"Approx. Area (ha): {catchment_data['area_ha']}")
        print("Degradation Status:")
        for layer, status in catchment_data['degradation_summary'].items():
            print(f"  - {layer}: {status}")
        
        print("\n--- Raw Attributes for RAG Ingestion ---")
        for raw_attr in catchment_data['raw_attributes']:
            print(raw_attr)

    else:
        print("No catchment data found for the given location.")
