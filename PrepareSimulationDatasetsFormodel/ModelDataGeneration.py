import xml.etree.ElementTree as ET
import json
import numpy as np
import os
import hashlib

# ---------------------------
# Helper functions
# ---------------------------

def hash_string_to_float(s, max_val=1e6):
    """Convert a string (e.g., lane ID) to a float using hashing."""
    h = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % max_val
    return float(h)

def normalize_vector(vec):
    min_v = vec.min()
    max_v = vec.max()
    return (vec - min_v)/(max_v-min_v+1e-8)

# ---------------------------
# Input readers
# ---------------------------

def read_od_matrix(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    od_list = []
    for actor in root.findall('actorConfig'):
        for timeSlice in actor.findall('timeSlice'):
            for odPair in timeSlice.findall('odPair'):
                amount = float(odPair.get('amount',0))
                od_list.append([amount])
    return np.array(od_list, dtype=np.float32)

def read_vtypes(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    vtypes = root.findall('.//vType')
    vtype_list = []

    for v in vtypes:
        speed = float(v.get('speed', 0))
        personCap = float(v.get('personCapacity', 0))
        containerCap = float(v.get('containerCapacity', 0))
        prob = float(v.get('probability', 1))
        vclass_hash = hash_string_to_float(v.get('vClass','unknown'))
        id_hash = hash_string_to_float(v.get('id','unknown'))

        vtype_list.append([speed, personCap, containerCap, prob, vclass_hash, id_hash])

    return np.array(vtype_list, dtype=np.float32)

def read_activitygen(file_path):
    with open(file_path,'r') as f:
        data = json.load(f)

    all_trips = []
    activities = data.get('activities', {})
    slices = data.get('slices', {})

    for slice_info in slices.values():
        for chain in slice_info.get('activityChains', []):
            activity_list = chain[1]
            mode_list = chain[2] if len(chain) > 2 else []
            modes_prob = [prob for _, prob in mode_list]

            for act_name in activity_list:
                act_info = activities.get(act_name, {})
                duration = act_info.get('duration', {}).get('m', 0) * 60 + act_info.get('duration', {}).get('s', 0)
                start = act_info.get('start', {}).get('m', 0) * 60 + act_info.get('start', {}).get('s', 0)
                all_trips.append([duration, start] + (modes_prob if modes_prob else [0.0]*len(modes_prob)))

    return np.array(all_trips, dtype=np.float32)

def read_stops(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    stops = []
    for stop in root.findall('busStop'):
        startPos = float(stop.get('startPos',0.0))
        endPos = float(stop.get('endPos',0.0))
        stops.append([startPos,endPos])
    return np.array(stops,dtype=np.float32)

def read_parking(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    parks = []
    for p in root.findall('parkingArea'):
        startPos = float(p.get('startPos',0.0))
        endPos = float(p.get('endPos',0.0))
        capacity = float(p.get('roadsideCapacity',0.0))
        parks.append([startPos,endPos,capacity])
    return np.array(parks,dtype=np.float32)

def read_taxis(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    taxis = []
    for t in root.findall('parkingArea'):
        startPos = float(t.get('startPos',0.0))
        endPos = float(t.get('endPos',0.0))
        capacity = float(t.get('roadsideCapacity',0.0))
        friendly = 1.0 if t.get('friendlyPos')=='true' else 0.0
        taxis.append([startPos,endPos,capacity,friendly])
    return np.array(taxis,dtype=np.float32)

def read_network(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    edges = [e for e in root.findall('edge') if e.get('function') != 'internal']
    edge_features = []
    
    for e in edges:
        lanes = e.findall('lane')
        num_lanes = len(lanes)  
        for lane in lanes:
            speed = float(lane.get('speed', 0))
            length = float(lane.get('length', 0))
            lane_hash = hash_string_to_float(lane.get('id','0'))
            edge_features.append([speed, length, num_lanes, lane_hash])
    network_summary = np.array([[len(edges), len(root.findall('junction'))]], dtype=np.float32)
    return np.array(edge_features, dtype=np.float32), network_summary

def read_bus_trips(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    trips = []

    for trip in root.findall('trip'):
        if trip.get('type') != 'pt_bus': continue
        depart = float(trip.get('depart',0))
        stops = trip.findall('stop')
        num_stops = len(stops)
        total_stop_duration = sum(float(s.get('duration',0)) for s in stops)
        line_hash = hash_string_to_float(trip.get('id',''))
        trips.append([depart, num_stops, total_stop_duration, line_hash])

    return np.array(trips, dtype=np.float32)

# ---------------------------
# Outputs readers
# ---------------------------

def read_tripinfo(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    trips = []
    for trip in root.findall('tripinfo'):
        trips.append([
            hash_string_to_float(trip.get('id','')),
            hash_string_to_float(trip.get('vType','')),
            float(trip.get('depart',0)),
            float(trip.get('arrival',0)),
            float(trip.get('duration',0)),
            float(trip.get('routeLength',0)),
            float(trip.get('waitingTime',0)),
            float(trip.get('stopTime',0)),
            float(trip.get('timeLoss',0))
        ])
    return np.array(trips,dtype=np.float32)

def read_statistics(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    perf_attrs = ['clockBegin','clockEnd','clockDuration','traciDuration','realTimeFactor',
                  'vehicleUpdatesPerSecond','personUpdatesPerSecond','begin','end','duration']

    veh_attrs = ['loaded','inserted','running','waiting']
    tele_attrs = ['total','jam','yield','wrongLane']
    safety_attrs = ['collisions','emergencyStops','emergencyBraking']
    vehicleTrip_attrs = ['count','routeLength','speed','duration','waitingTime','timeLoss','departDelay','departDelayWaiting',
                          'totalTravelTime','totalDepartDelay']

    def get_values(tag_name, attrs):
        tag = root.find(tag_name)
        if tag is None:
            return [0.0]*len(attrs)
        return [float(tag.get(attr,0.0)) for attr in attrs]

    values = []
    values += get_values('performance', perf_attrs)
    values += get_values('vehicles', veh_attrs)
    values += get_values('teleports', tele_attrs)
    values += get_values('safety', safety_attrs)
    values += get_values('vehicleTripStatistics', vehicleTrip_attrs)

    return np.array(values, dtype=np.float32)

def read_person_summary(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    attrs = ['loaded','walking','waitingForRide','riding','stopping','jammed','ended','arrived','teleports','duration']
    totals = np.zeros(len(attrs), dtype=np.float32)
    for step in root.findall('step'):
        for i, attr in enumerate(attrs):
            totals[i] += float(step.get(attr, 0))
    return totals

# ---------------------------
# Logging 
# ---------------------------

def log_dataset_sizes_and_attributes(X_sizes, Y_sizes, total_X, total_Y, log_file='dataset_sizes.txt'):
    # Define attributes for each dataset part
    details = {
        'od_matrix': ['amount'],
        'vtypes': ['speed','personCapacity','containerCapacity','probability','vClass_hash','id_hash'],
        'activitygen': ['duration','start_time','mode_probs'],
        'stops': ['startPos','endPos'],
        'parking': ['startPos','endPos','roadsideCapacity'],
        'taxis': ['startPos','endPos','roadsideCapacity','friendlyPos'],
        'network_edges': ['speed','length','num_lanes','lane_id_hash'],
        'network_summary': ['num_edges','num_junctions'],
        'bus_trips': ['depart','num_stops','total_stop_duration','line_id_hash'],
        'tripinfo': ['trip_id_hash','vType_hash','depart','arrival','duration','routeLength','waitingTime','stopTime','timeLoss'],
        'statistics': ['performance','vehicles','teleports','safety','vehicleTripStatistics'],
        'person_summary': ['loaded','walking','waitingForRide','riding','stopping','jammed','ended','arrived','teleports','duration']
    }

    with open(log_file,'w', encoding='utf-8') as f:
        f.write("✅ Detailed dataset summary:\n\n")
        
        f.write("X data sizes, percentages, and attributes:\n")
        for k,v in X_sizes.items():
            pct = (v/total_X)*100
            attrs = ', '.join(details.get(k, []))
            f.write(f"{k:20} : {v:8} elements ({pct:6.2f}%) | attributes: {attrs}\n")
        f.write(f"Total X size: {total_X} elements\n\n")
        
        f.write("Y data sizes, percentages, and attributes:\n")
        for k,v in Y_sizes.items():
            pct = (v/total_Y)*100
            attrs = ', '.join(details.get(k, []))
            f.write(f"{k:20} : {v:8} elements ({pct:6.2f}%) | attributes: {attrs}\n")
        f.write(f"Total Y size: {total_Y} elements\n")
    
    print(f"✅ Detailed dataset summary saved to '{log_file}'")


# ---------------------------
# Prepare dataset
# ---------------------------
def prepare_XY(sim_folder, log_file='dataset_sizes.txt'):
    X_parts, X_sizes = [], {}

    # Inputs
    od = read_od_matrix(os.path.join(sim_folder,'osm_odmatrix_amitran.xml'))
    X_parts.append(od.flatten()); X_sizes['od_matrix'] = od.size

    vtypes = read_vtypes(os.path.join(sim_folder,'basic.vType.xml'))
    X_parts.append(vtypes.flatten()); X_sizes['vtypes'] = vtypes.size

    activitygen = read_activitygen(os.path.join(sim_folder,'osm_activitygen.json'))
    X_parts.append(activitygen.flatten()); X_sizes['activitygen'] = activitygen.size

    stops = read_stops(os.path.join(sim_folder,'osm_stops.add.xml'))
    X_parts.append(stops.flatten()); X_sizes['stops'] = stops.size

    parking = read_parking(os.path.join(sim_folder,'osm_complete_parking_areas.add.xml'))
    X_parts.append(parking.flatten()); X_sizes['parking'] = parking.size

    taxis = read_taxis(os.path.join(sim_folder,'osm_taxi_stands.add.xml'))
    X_parts.append(taxis.flatten()); X_sizes['taxis'] = taxis.size

    edge_features, network_summary = read_network(os.path.join(sim_folder,'osm.net.xml'))
    X_parts.append(edge_features.flatten()); X_sizes['network_edges'] = edge_features.size
    X_parts.append(network_summary.flatten()); X_sizes['network_summary'] = network_summary.size

    bus = read_bus_trips(os.path.join(sim_folder,'trips.trips.xml'))
    X_parts.append(bus.flatten()); X_sizes['bus_trips'] = bus.size

    X_raw = np.concatenate(X_parts)
    X_norm = normalize_vector(X_raw)

    # Outputs
    Y_parts, Y_sizes = [], {}

    tripinfo = read_tripinfo(os.path.join(sim_folder,'output.tripinfo.xml'))
    Y_parts.append(tripinfo.flatten()); Y_sizes['tripinfo'] = tripinfo.size

    statistics = read_statistics(os.path.join(sim_folder,'output.statistics.xml'))
    Y_parts.append(statistics); Y_sizes['statistics'] = statistics.size

    person = read_person_summary(os.path.join(sim_folder,'output.person.summary.xml'))
    Y_parts.append(person); Y_sizes['person_summary'] = person.size

    Y_raw = np.concatenate(Y_parts)
    Y_norm = normalize_vector(Y_raw)

    # Logging
    total_X, total_Y = X_raw.size, Y_raw.size
    log_dataset_sizes_and_attributes(X_sizes, Y_sizes, total_X, total_Y, log_file)
    
    return X_raw, X_norm, Y_raw, Y_norm, X_sizes, Y_sizes
    
# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    sim_folder = r'.'
    X_raw, X_norm, Y_raw, Y_norm, X_sizes, Y_sizes = prepare_XY(sim_folder)

    np.savez('simulation_dataset.npz',
             X_raw=X_raw,
             X_norm=X_norm,
             Y_raw=Y_raw,
             Y_norm=Y_norm)

    print("✅ Full dataset saved successfully to 'simulation_dataset.npz'!")
    