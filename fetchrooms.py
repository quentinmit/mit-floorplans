#!/usr/bin/env python3

from netrc import netrc
import os
import os.path

import cx_Oracle

cx_Oracle.init_oracle_client(lib_dir=os.getenv("ORACLE_LIB_DIR") or os.path.join(os.path.dirname(__file__), 'oracle'))

def fetch_data():
    path = os.path.join(os.path.dirname(__file__), '.netrc')
    if os.path.exists(path):
        n = netrc(path)
    else:
        n = netrc()
    user, _, password = n.authenticators("warehouse")
    conn = cx_Oracle.connect(user=user, password=password, dsn="warehouse", encoding="UTF-8")

    out = dict()

    for key, query in {
            "buildings": "select * from fac_building order by fac_building_key asc", # buildings has building_street_address
            "floors": "select * from fac_floor order by building_key asc, floor_key asc",
            "organizations": "select * from fac_organization order by organization_key asc",
            "fac_rooms": "select * from fac_rooms order by fac_room_key asc",
            "rooms": "select building_number, building_component, floor, room_number, building_room, building_room_name, room_square_footage, space_unit_code, space_unit, dlc_key, fclt_organization_key as organization_key, space_usage from space_detail left join buildings using (building_key) left join space_floor using (floor_key) left join space_unit using (space_unit_key) left join space_usage using (space_usage_key) order by building_number asc, building_component asc, building_room asc",
    }.items():
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        cursor.rowfactory = lambda *args: {k.lower(): v for k,v in zip(columns, args) if k != 'WAREHOUSE_LOAD_DATE' and v != '' and v != None}

        out[key] = cursor.fetchall()
    return out

def main():
    import json

    out = fetch_data()
    print(json.dumps(out, indent=1))

if __name__ == '__main__':
    main()
