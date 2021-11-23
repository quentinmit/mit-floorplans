#!/usr/bin/env python3

from netrc import netrc
import os
import yaml

import cx_Oracle

cx_Oracle.init_oracle_client(lib_dir=os.getenv("ORACLE_LIB_DIR") or './oracle')

if os.path.exists('.netrc'):
    n = netrc('.netrc')
else:
    n = netrc()
user, _, password = n.authenticators("warehouse")
conn = cx_Oracle.connect(user=user, password=password, dsn="warehouse", encoding="UTF-8")

cursor = conn.cursor()
cursor.execute("select * from space_detail where rownum < 100")
columns = [col[0] for col in cursor.description]
cursor.rowfactory = lambda *args: dict(zip(columns, args))

rooms = cursor.fetchall()
print(yaml.dump(rooms))
