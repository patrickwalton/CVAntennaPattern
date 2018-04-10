class Point:
    # Represents a point being tracked
    def __init__(self, pixel_coordinates):
        self.location = pixel_coordinates
        self.prior_location =self.location

    def update(self, pixel_coordinates):
        self.prior_location = self.location
        self.location = pixel_coordinates
