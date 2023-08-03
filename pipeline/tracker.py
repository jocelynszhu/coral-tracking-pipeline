import sys, os

from tracker.sort import Sort


class ObjectTracker(object):
    def __init__(self, trackerObjectName):
        if trackerObjectName == 'sort':  # Add more trackers in elif whenever needed
            self.trackerObject = SortTracker()
        else:
            print("Invalid Tracker Name")
            self.trackerObject = None

class SortTracker(object):
    def __init__(self):
        self.mot_tracker = Sort()