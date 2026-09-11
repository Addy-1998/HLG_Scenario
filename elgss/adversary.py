"""Scripted adversary that ENACTS the interaction spec.

Without this, the adversarial vehicle is a default IDM driver and the
'cut_in'/'brake' specifications are never executed, making scenarios
barely harder than ambient traffic. This subclass triggers the
manoeuvre once the adversary is within ``trigger_distance`` of the ego.
"""
from highway_env.vehicle.behavior import IDMVehicle

SEVERITY_GAIN = {"medium": 1.0, "high": 1.5}


class AdversaryVehicle(IDMVehicle):
    def __init__(self, road, position, heading=0.0, speed=0.0):
        super().__init__(road, position, heading, speed)
        self._ego = None
        self._itype = None
        self._trigger = 15.0
        self._gain = 1.0
        self._fired = False

    def setup_attack(self, ego, itype, trigger_distance, severity):
        self._ego = ego
        self._itype = itype
        self._trigger = float(trigger_distance)
        self._gain = SEVERITY_GAIN.get(severity, 1.0)

    def act(self, action=None):
        if self._ego is not None and not self._fired and not self.crashed:
            gap = abs(self.position[0] - self._ego.position[0])
            if gap < self._trigger:
                self._fired = True
                # aggressive car-following: small gaps, hard accel
                self.DISTANCE_WANTED = 2.0 / self._gain
                self.TIME_WANTED = 0.5 / self._gain
                if self._itype == "cut_in":
                    self.target_lane_index = self._ego.lane_index
                    self.target_speed = max(5.0, self._ego.speed - 6.0 * self._gain)
                elif self._itype == "brake":
                    self.target_speed = max(0.0, self.speed - 15.0 * self._gain)
                elif self._itype == "overtake":
                    self.target_speed = self.speed + 8.0 * self._gain
                elif self._itype == "follow":
                    self.target_speed = self._ego.speed + 4.0 * self._gain
        return super().act(action)
