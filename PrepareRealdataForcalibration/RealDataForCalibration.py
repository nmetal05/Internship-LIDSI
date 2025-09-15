import json
import pandas as pd


with open("tomtom_data.json", "r") as f:
    data = json.load(f)

rows = []

network = data.get("network", {})
segments = network.get("segmentResults", [])

# Map timeSet ids to readable names
time_sets = {ts["@id"]: ts["name"] for ts in data.get("timeSets", [])}

for seg in segments:
    dist = seg.get("distance", 0)  
    speed_limit = seg.get("speedLimit", 0)  

    for res in seg.get("segmentTimeResults", []):
        time_set_id = res.get("timeSet")
        time_name = time_sets.get(time_set_id, str(time_set_id))

        # Only keep results for 10:00-16:00 timeline 
        if time_name != "10:00-16:00":
            continue

        avg_speed = res.get("harmonicAverageSpeed")
        avg_tt = res.get("averageTravelTime", None)  

        if avg_speed and avg_tt and speed_limit > 0:
            free_flow_time = (dist / 1000) / (speed_limit / 3600)
            mean_waiting_time = avg_tt - free_flow_time
            mean_waiting_time = max(0, mean_waiting_time)  

            speed_factor = avg_speed / speed_limit

            rows.append({
                "meanSpeed": avg_speed,
                "meanWaitingTime": mean_waiting_time,
                "speedFactor": speed_factor
            })

df = pd.DataFrame(rows)
df.to_csv("real_data.csv", index=False)

print(f"Dataset saved with {len(df)} rows")
