import logging
from doc_generator.geospatial_utils import _query_arcgis_vector

logger = logging.getLogger(__name__)

class SoilDrainageService:
    """
    A service to query for Soil Drainage Class data using the consolidated
    freshwater farm plan contextual data service.
    """
    # This URL points to the consolidated ArcGIS service that is already
    # used elsewhere in the application and is known to be working.
    BASE_URL = "https://services3.arcgis.com/v5RzLI7nHYeFImL4/arcgis/rest/services/Freshwater_farm_plan_contextual_data_hosted/FeatureServer/5/query"

    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def get_soil_drainage_class(self):
        """
        Queries the ArcGIS service for the soil drainage class at the given coordinates.
        """
        try:
            # Use the existing utility function from geospatial_utils to query the service.
            # This promotes code reuse and consistency.
            features = _query_arcgis_vector(self.BASE_URL, self.lon, self.lat)

            if features:
                # The data is nested within the 'properties' of the first feature.
                properties = features[0].get('properties', {})
                
                # The correct field name in this service is 'Darg_Drain'.
                drainage_class = properties.get('Darg_Drain')

                if drainage_class:
                    logger.info(f"Successfully retrieved soil drainage class: {drainage_class}")
                    return drainage_class
                else:
                    logger.warning("Soil drainage field 'Darg_Drain' was empty or not found in the response.")
                    return "Unknown"
            else:
                logger.info("No soil drainage data found for the given location via ArcGIS.")
                return "Not available"

        except Exception as e:
            logger.error(f"An unexpected error occurred in SoilDrainageService: {e}", exc_info=True)
            return "Error"