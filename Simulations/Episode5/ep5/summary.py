import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

# Replace with your file path
summary_file = "output.summary.xml"

times, running = [], []

tree = ET.parse(summary_file)
root = tree.getroot()

for step in root.findall("step"):
    t = int(float(step.get("time")))   # time in seconds
    r = int(step.get("running"))       # vehicles running
    times.append(datetime.datetime(2025, 1, 1) + datetime.timedelta(seconds=t))
    running.append(r)

plt.figure(figsize=(8,6))
plt.plot(times, running, label="output.summary.xml", color="black")

plt.xlabel("Time of day")
plt.ylabel("Running vehicles")
plt.title("SUMO Summary Report")
plt.legend()

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gcf().autofmt_xdate()
plt.show()
