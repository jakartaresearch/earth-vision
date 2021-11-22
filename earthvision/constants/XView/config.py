"""Configuration file for labels, labelmap, data ids."""

class_id = [
    11, 12, 13, 15, 17, 18, 19, 20, 21, 23, 
    24, 25, 26, 27, 28, 29, 32, 33, 34, 35, 
    36, 37, 38, 40, 41, 42, 44, 45, 47, 49, 
    50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 
    61, 62, 63, 64, 65, 66, 71, 72, 73, 74, 
    76, 77, 79, 83, 84, 86, 89, 91, 93, 94
]

class_name = [
    'Fixed-wing Aircraft', 'Small Aircraft', 'Passenger/Cargo Plane', 'Helicopter', 
    'Passenger Vehicle', 'Small Car', 'Bus', 'Pickup Truck', 'Utility Truck', 'Truck', 
    'Cargo Truck', 'Truck Tractor w/ Box Trailer', 'Truck Tractor', 'Trailer', 
    'Truck Tractor w/ Flatbed Trailer', 'Truck Tractor w/ Liquid Tank', 'Crane Truck', 
    'Railway Vehicle', 'Passenger Car', 'Cargo/Container Car', 'Flat Car', 'Tank car', 
    'Locomotive', 'Maritime Vessel', 'Motorboat', 'Sailboat', 'Tugboat', 'Barge', 
    'Fishing Vessel', 'Ferry', 'Yacht', 'Container Ship', 'Oil Tanker', 
    'Engineering Vehicle', 'Tower crane', 'Container Crane', 'Reach Stacker', 
    'Straddle Carrier', 'Mobile Crane', 'Dump Truck', 'Haul Truck', 'Scraper/Tractor', 
    'Front loader/Bulldozer', 'Excavator', 'Cement Mixer', 'Ground Grader', 'Hut/Tent', 
    'Shed', 'Building', 'Aircraft Hangar', 'Damaged Building', 'Facility', 
    'Construction Site', 'Vehicle Lot', 'Helipad', 'Storage Tank', 
    'Shipping container lot', 'Shipping Container', 'Pylon', 'Tower'
]

idx = range(len(class_id))
index_mapping = dict(zip(class_id, idx))
CLASS_ENC = dict(zip(idx, class_name))
CLASS_DEC = dict(zip(class_name, idx))