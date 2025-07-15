import re
from datetime import datetime

import matplotlib.pyplot as plt

log_path = "carla_autopilot_log.txt"

timestamps = []
positions_x = []
positions_y = []
speeds = []
yaws = []

with open(log_path, "r") as f:
    for line in f:
        try:
            time_str = re.search(r"\[(.*?)\]", line).group(1)
            timestamp = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

            pos_match = re.search(r"Pos: \((.*?), (.*?)\)", line)
            x = float(pos_match.group(1))
            y = float(pos_match.group(2))

            speed_match = re.search(r"Speed: (.*?) m/s", line)
            speed = float(speed_match.group(1))

            yaw_match = re.search(r"Yaw: (.*?),", line)
            yaw = float(yaw_match.group(1))

            timestamps.append(timestamp)
            positions_x.append(x)
            positions_y.append(y)
            speeds.append(speed)
            yaws.append(yaw)
        except Exception as e:
            print(f"Parse error: {e}")

# Zaman farkları
time_secs = [(t - timestamps[0]).total_seconds() for t in timestamps]

# Yönelim hatası (örnek: sabit hedef yönelim 90 derece)
desired_yaw = 90.0
yaw_errors = [abs(y - desired_yaw) for y in yaws]

# Yörünge + Yaw + Yaw error tek grafik
fig, ax1 = plt.subplots(figsize=(12, 7))

# Yörünge (XY)
ax1.plot(positions_x, positions_y, "b-", label="Yörünge (X-Y)")
ax1.set_xlabel("X konumu (m)")
ax1.set_ylabel("Y konumu (m)", color="b")
ax1.tick_params(axis="y", labelcolor="b")
ax1.grid()

# Yönelim (Yaw)
ax2 = ax1.twinx()
ax2.plot(time_secs, yaws, "g--", label="Yönelim (Yaw)", alpha=0.6)
ax2.set_ylabel("Yaw (derece)", color="g")
ax2.tick_params(axis="y", labelcolor="g")

# Yönelim hatası
ax2.plot(time_secs, yaw_errors, "r-.", label="Yaw Hatası", alpha=0.8)

# Tüm çizgilerin legend'larını birleştir
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.title("Araç Yörüngesi + Yönelim ve Yönelim Hatası")
plt.tight_layout()
plt.savefig("trajectory_yaw_error.png")
plt.show()

# Hız Profili Ayrı Grafik
plt.figure(figsize=(10, 4))
plt.plot(time_secs, speeds, label="Hız (m/s)", color="purple")
plt.xlabel("Zaman (s)")
plt.ylabel("Hız (m/s)")
plt.title("Hız Profili")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("speed_profile.png")
plt.show()
