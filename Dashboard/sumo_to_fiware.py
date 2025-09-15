import os
import sys
import time
import requests
import json
import xml.etree.ElementTree as ET

# --- Reference point delta between SUMO and GeoJSON ---
# SUMO coordinates for the reference place
sumo_ref_lat = 33.663404
sumo_ref_lon = -7.879069

# GeoJSON coordinates for the same place
geojson_ref_lat = 33.553842
geojson_ref_lon = -7.671395

# Compute deltas
delta_lat = geojson_ref_lat - sumo_ref_lat
delta_lon = geojson_ref_lon - sumo_ref_lon

# --- Function to correct SUMO coordinates ---
def correct_coordinates(lon, lat):
    """
    Adjust SUMO coordinates to align with GeoJSON map.
    """
    corrected_lon = lon + delta_lon
    corrected_lat = lat + delta_lat
    return corrected_lon, corrected_lat

# --- SUMO HOME ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Erreur : Veuillez définir la variable d'environnement 'SUMO_HOME'.")

import traci
from traci.constants import *

# --- CONFIGURATION ---
SUMO_CONFIG_FILE = "../episode1/osm.sumocfg"  # Update this path
SUMO_CMD = ["sumo-gui", "-c", SUMO_CONFIG_FILE]

ORION_HOST = "localhost"
ORION_PORT = "1026"
ORION_URL = f"http://{ORION_HOST}:{ORION_PORT}/v2/op/update"

FIWARE_HEADERS = {
    'Content-Type': 'application/json',
    'Fiware-Service': 'sumo_digital_twin',
    'Fiware-ServicePath': '/'
}

UPDATE_INTERVAL = 5  # Send updates every 5 simulation seconds
SUMO_NET_FILE = "../episode1/osm.net.xml"  # path to SUMO net file

# --- LOAD EDGE NAMES ---
def load_edge_names(net_file_path):
    tree = ET.parse(net_file_path)
    root = tree.getroot()
    edge_to_name = {}
    for edge in root.findall("edge"):
        edge_id = edge.attrib.get("id")
        name = edge.attrib.get("name", None)
        if name:
            edge_to_name[edge_id] = name
        else:
            edge_to_name[edge_id] = edge_id  # fallback
    print(f"Loaded street names for {len(edge_to_name)} edges")
    return edge_to_name

edge_to_name = load_edge_names(SUMO_NET_FILE)

# --- NGSI entity builder ---
def create_ngsi_v2_entity(veh_id, lon, lat, speed, waiting_time, co2_emission, street_name):
    """Formats vehicle data into NGSI-v2 entity format."""
    return {
        "id": f"urn:ngsi-ld:Vehicle:sumo:{veh_id}",
        "type": "Vehicle",
        "location": {"type": "geo:json", "value": {"type": "Point", "coordinates": [lon, lat]}},
        "speed": {"type": "Number", "value": round(speed, 2)},
        "waitingTime": {"type": "Number", "value": round(waiting_time, 2)},
        "co2Emission": {"type": "Number", "value": round(co2_emission, 2)},
        "streetName": {"type": "Text", "value": street_name}
    }

# --- MAIN SIMULATION LOOP ---
def run_simulation():
    print("Démarrage de la simulation SUMO (Mode Performant)...")
    traci.start(SUMO_CMD)
    
    step = 0
    subscribed_vehicles = set()

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        if step % UPDATE_INTERVAL == 0:
            # Identify new and departed vehicles
            current_vehicle_ids = set(traci.vehicle.getIDList())
            newly_arrived_vehicles = current_vehicle_ids - subscribed_vehicles
            departed_vehicles = subscribed_vehicles - current_vehicle_ids
            
            # Subscribe to new vehicles
            for veh_id in newly_arrived_vehicles:
                traci.vehicle.subscribe(veh_id, [
                    VAR_POSITION,
                    VAR_SPEED,
                    VAR_WAITING_TIME,
                    VAR_CO2EMISSION
                ])
            subscribed_vehicles.update(newly_arrived_vehicles)
            
            # Remove departed vehicles
            subscribed_vehicles -= departed_vehicles

            if not subscribed_vehicles:
                step += 1
                continue

            print(f"Step {step}: {len(subscribed_vehicles)} véhicules suivis. Mise à jour de FIWARE...")

            entities_payload = []
            for veh_id in subscribed_vehicles:
                results = traci.vehicle.getSubscriptionResults(veh_id)
                if results:
                    pos_x, pos_y = results[VAR_POSITION]
                    speed = results[VAR_SPEED]
                    waiting_time = results[VAR_WAITING_TIME]
                    co2_emission = results[VAR_CO2EMISSION]

                    lon, lat = traci.simulation.convertGeo(pos_x, pos_y)
                    lon, lat = correct_coordinates(lon, lat)

                    edge_id = traci.vehicle.getRoadID(veh_id)
                    street_name = edge_to_name.get(edge_id, edge_id)

                    entity = create_ngsi_v2_entity(veh_id, lon, lat, speed, waiting_time, co2_emission, street_name)
                    entities_payload.append(entity)

            # Send batch update to Orion
            if entities_payload:
                ngsi_payload = {"actionType": "APPEND", "entities": entities_payload}
                try:
                    requests.post(ORION_URL, data=json.dumps(ngsi_payload), headers=FIWARE_HEADERS)
                except requests.exceptions.RequestException as e:
                    print(f"  -> Erreur de connexion à Orion: {e}")

        step += 1

    print("Simulation terminée.")
    traci.close()

if __name__ == "__main__":
    if not os.path.exists(SUMO_CONFIG_FILE):
        print(f"Erreur Critique : Le fichier de configuration '{SUMO_CONFIG_FILE}' n'a pas été trouvé.")
    else:
        run_simulation()
