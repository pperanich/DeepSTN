from pathlib import Path
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, box
import pickle


def build_grid_map(zones_by_id, root_bb_bounds, grid_w, grid_h):
    bb_w, bb_h = abs(root_bb_bounds[0] - root_bb_bounds[2]), abs(root_bb_bounds[1] - root_bb_bounds[3])
    step_x, step_y = bb_w / grid_w, bb_h / grid_h

    grid_map = []

    for row in range(0, grid_h):
        cols = []
        for col in range(0, grid_w):
            x_min, y_min = root_bb_bounds[0] + col * step_x, root_bb_bounds[1] + row * step_y
            x_max, y_max = x_min + step_x, y_min + step_y

            bb_df = gpd.GeoDataFrame(gpd.GeoSeries(box(x_min, y_min, x_max, y_max)), columns=['geometry'],
                                     crs=zones.crs)

            area_ratios = dict()

            matches = zones.cx[x_min:x_max, y_min:y_max]
            if len(matches) > 0:
                intersections = gpd.overlay(matches, bb_df)
                for _, item in intersections.iterrows():
                    zone = zones_by_id.loc[item['layer_id']]
                    # if zone.equals(root_zone):
                    #     continue
                    area_ratios[item['layer_id']] = item.geometry.area / zone.geometry.area

            cols.append(area_ratios)

        grid_map.append(cols)

    return grid_map


zones = gpd.read_file("/home/clinamen/school/ML2020/project/data/rome-zones.geojson")
datafiles_path = Path("/home/clinamen/school/ML2020/project/data/tim-presence-data/")

zones_by_id = zones.set_index('layer_id')
root_zone = zones_by_id.loc['12|058|091']
root_bb_bounds = root_zone.geometry.bounds

# grid size
grid_w, grid_h = 12, 21

# load or create grid map
grid_map_file = Path(f'grid_map_W{grid_w}xH{grid_h}.pickle')
if not grid_map_file.exists():
    grid_map = build_grid_map(zones_by_id, root_bb_bounds, grid_w, grid_h)
    with grid_map_file.open('wb') as f:
        pickle.dump(grid_map, f)
else:
    with grid_map_file.open('rb') as f:
        grid_map = pickle.load(f)


data = []

# loop over day files
for f in sorted(datafiles_path.iterdir()):
    with f.open() as json_file:
        day = dict()
        response = json.load(json_file)
        for item in response['Data']:
            date = item['DateFrom']
            zone_id = item['LayerId']

            if len(zone_id.split('|')) != 5:
                # skips higher-level layers
                continue

            values = [x['DataValue'] for x in item['PresenceData'] if x['DataType'] == "P"]
            if len(values) == 0:
                continue

            zones = day.get(date, dict())
            zones[zone_id] = values[0]  # [presence, inflow, outflow]
            day[date] = zones

        # calculate approximate in/out flow for each grid cell
        prev_interval = dict()
        for t in sorted(day):
            interval_grid = np.zeros((2, grid_h, grid_w))
            for zid in day[t]:
                flow = (day[t][zid] - prev_interval.get(zid, 0), 0)  # (inflow, outflow)
                if flow[0] < 0:
                    flow = (0, - flow[0])

                # accumulate in/out flows for each cell
                for x, y in np.ndindex(interval_grid.shape[1:]):
                    zone_weight = grid_map[x][y].get(zid, 0)
                    interval_grid[0, x, y] = interval_grid[0, x, y] + zone_weight * flow[0]
                    interval_grid[1, x, y] = interval_grid[1, x, y] + zone_weight * flow[1]

            data.append(interval_grid)
            prev_interval = day[t]

# save data in NumPy format
np.save(f"data_W{grid_w}xH{grid_h}.npy", np.array(data))
