import numpy as np

_KNOWN_SHAPES = dict()

for shapes in [
        # 50_0 stair font
        ("U", (0,-27,1,-32,4,-36,8,-37,11,-37,15,-36,17,-32,19,-27,19,0)),
        ("P", (0,37,12,37,17,35,18,34,19,30,19,25,18,21,17,19,12,18,0,18)),
        # 50_0 room number font
        ("A", (12,31,24,0), (15,0)),
        ("B", (4,-1,5,-3,7,-6,7,-10,5,-13,4,-14,0,-16,-14,-16,-14,15,0,15,4,14,5,12,7,9,7,6,5,3,4,2,0,0,-14,0)),
        ("C", (-2,3,-5,6,-8,7,-14,7,-17,6,-20,3,-21,0,-23,-5,-23,-12,-21,-17,-20,-19,-17,-22,-14,-24,-8,-24,-5,-22,-2,-19,0,-17)),
        ("D", (0,31,10,31,15,29,18,26,19,23,21,19,21,11,19,7,18,4,15,1,10,0,0,0)),
        ("E", (0,32,20,32), (12,0), (20,0)),
        ("F", (0,31,20,31), (12,0)),
        ("G", (-2,3,-5,6,-8,7,-14,7,-17,6,-20,3,-21,0,-23,-5,-23,-12,-21,-17,-20,-20,-17,-23,-14,-24,-8,-24,-5,-23,-2,-20,0,-17,0,-12,-8,-12)),
        ("I", (0,-31)),
        ("J", (0,-24,-2,-28,-3,-30,-6,-31,-9,-31,-12,-30,-14,-28,-15,-24,-15,-21)),
        ("K", (0,-31), (-20,-21), (13,-18)),
        ("L", (0,-32,18,-32)),
        ("M", (0,31,12,0,24,31,24,0)),
        ("N", (0,31,21,0,21,31)),
        ("O", (-3,-1,-6,-4,-8,-7,-9,-11,-9,-19,-8,-23,-6,-26,-3,-29,0,-31,6,-31,9,-29,12,-26,13,-23,15,-19,15,-11,13,-7,12,-4,9,-1,6,0,0,0)),
        ("P", (0,31,13,31,18,29,19,28,21,25,21,20,19,17,18,16,13,14,0,14)),
        ("R", (0,31,13,31,18,30,19,28,21,25,21,22,19,19,18,18,13,16,0,16), (11,-16)),
        ("S", (-3,3,-8,5,-14,5,-18,3,-21,0,-21,-3,-20,-6,-18,-7,-15,-9,-6,-12,-3,-13,-2,-15,0,-18,0,-22,-3,-25,-8,-26,-14,-26,-18,-25,-21,-22)),
        ("T", (0,-32), (21,0)),
        ("U", (0,-22,2,-27,5,-30,9,-31,12,-31,17,-30,20,-27,21,-22,21,0)),
        ("V", (12,-32,24,0)),
        ("V", (22,-75,43,0)),
        ("W", (13,-75,27,0,40,-75,53,0)),
        ("Y", (22,-36,43,0), (0,-39)),
        ("0", (-5,-1,-8,-6,-9,-13,-9,-17,-8,-25,-5,-29,0,-31,3,-31,7,-29,10,-25,12,-17,12,-13,10,-6,7,-1,3,0,0,0)),
        ("1", (3,1,8,6,8,-26)),
        ("2", (0,1,1,4,3,6,6,7,12,7,15,6,16,4,18,1,18,-2,16,-5,13,-9,-1,-24,19,-24)),
        ("3", (16,0,8,-12,12,-12,15,-14,16,-15,18,-20,18,-23,16,-27,13,-30,9,-32,5,-32,0,-30,-1,-29,-3,-26)),
        ("4", (0,31,-15,11,7,11)),
        ("5", (-15,0,-16,-13,-15,-12,-10,-10,-6,-10,-1,-12,2,-15,3,-19,3,-22,2,-26,-1,-29,-6,-31,-10,-31,-15,-29,-16,-28,-18,-25)),
        ("6", (-1,3,-6,4,-9,4,-13,3,-16,-2,-17,-9,-17,-16,-16,-22,-13,-25,-9,-27,-7,-27,-3,-25,0,-22,2,-18,2,-16,0,-12,-3,-9,-7,-8,-9,-8,-13,-9,-16,-12,-17,-16)),
        ("7", (15,31,-6,31)),
        ("8", (-4,-2,-6,-5,-6,-8,-4,-11,-1,-12,5,-14,9,-15,12,-18,14,-21,14,-26,12,-29,11,-30,6,-32,0,-32,-4,-30,-6,-29,-7,-26,-7,-21,-6,-18,-3,-15,2,-14,8,-12,11,-11,12,-8,12,-5,11,-2,6,0,0,0)),
        ("9", (-1,-4,-4,-7,-9,-8,-10,-8,-15,-7,-18,-4,-19,0,-19,2,-18,6,-15,9,-10,11,-9,11,-4,9,-1,6,0,0,0,-7,-1,-14,-4,-19,-9,-20,-12,-20,-16,-19,-18,-16)),
        ("/", (-27,-48)),
        ("-", (27,0)),
        # 50_0 footer
        ("a", (0,-40), (-6,6,-12,9,-20,9,-26,6,-32,0,-34,-8,-34,-14,-32,-23,-26,-28,-20,-31,-12,-31,-6,-28,0,-23)),
        ("c", (-6,6,-12,9,-20,9,-26,6,-32,0,-35,-8,-35,-14,-32,-23,-26,-28,-20,-31,-12,-31,-6,-28,0,-23)),
        ("d", (0,-60), (-5,6,-11,9,-20,9,-25,6,-31,0,-34,-8,-34,-14,-31,-23,-25,-28,-20,-31,-11,-31,-5,-28,0,-23)),
        ("e", (34,0,34,5,31,11,28,14,23,17,14,17,8,14,3,8,0,0,0,-6,3,-15,8,-20,14,-23,23,-23,28,-20,34,-15)),
        ("f", (-6,0,-12,-3,-14,-12,-14,-60), (20,0)),
        ("g", (0,-46,-3,-54,-6,-57,-11,-60,-20,-60,-26,-57), (-6,6,-11,9,-20,9,-26,6,-31,0,-34,-8,-34,-14,-31,-23,-26,-28,-20,-31,-11,-31,-6,-28,0,-23)),
        ("h", (0,-60), (8,9,14,12,23,12,28,9,31,0,31,-28)),
        ("i", (3,-3,6,0,3,3,0,0), (0,-37)),
        ("n", (0,-40), (9,9,14,12,23,12,29,9,32,0,32,-28)),
        ("o", (-6,-3,-12,-9,-15,-17,-15,-23,-12,-32,-6,-37,0,-40,8,-40,14,-37,20,-32,23,-23,23,-17,20,-9,14,-3,8,0,0,0)),
        ("p", (0,-60), (5,6,11,9,20,9,25,6,31,0,34,-8,34,-14,31,-23,25,-28,20,-31,11,-31,5,-28,0,-23)),
        ("r", (0,-40), (2,8,8,14,14,17,22,17)),
        ("s", (-3,6,-11,9,-20,9,-28,6,-31,0,-28,-5,-23,-8,-8,-11,-3,-14,0,-20,0,-23,-3,-28,-11,-31,-20,-31,-28,-28,-31,-23)),
        ("t", (0,-49,3,-57,9,-60,14,-60), (20,0)),
        ("u", (0,-29,3,-37,8,-40,17,-40,23,-37,31,-29), (0,-40)),
        ("v", (17,-40,34,0)),
        ("y", (17,-40,34,0), (-5,-12,-11,-17,-17,-20,-20,-20)),
        (".", (-3,-3,0,-6,3,-3,0,0)),

        # 64_3 room number font
        ("B", (0,16,7,16,9,15,10,15,10,10,9,9,0,9), (2,-1,3,-2,3,-7,2,-8,0,-9,-7,-9)),
        ("D", (0,15,7,15,9,13,10,12,10,3,9,2,7,0,0,0,0,0)),
        ("D", (0,15,8,15,9,13,10,12,11,9,11,6,10,3,9,2,8,0,0,0,0,0)),
        ("D", (0,15,8,15,9,13,10,12,11,9,11,6,10,4,9,2,8,1,6,0,0,0,0,0)),
        ("D", (0,16,5,16,7,15,9,14,10,12,10,4,9,3,7,1,5,0,0,0,0,0)),
        ("E", (0,16,9,16), (6,0,0,-8,9,-8)), # discontinuous
        ("J", (0,-12,-1,-14,-2,-15,-6,-15,-7,-14,-8,-12,-8,-10)),
        ("P", (0,15,7,15,9,14,10,14,10,8,9,8,7,7,0,7)),
        ("R", (0,15,9,15,9,14,10,12,10,11,9,9,7,8,0,8), (5,-8)),
        ("R", (0,16,7,16,9,15,10,15,10,10,9,9,0,9), (5,-9)),
        ("S", (-1,1,-4,2,-7,2,-9,1,-10,0,-10,-3,-9,-4,-7,-5,-3,-6,-1,-7,-1,-8,0,-9,0,-12,-1,-13,-4,-14,-7,-14,-9,-13,-10,-12)),
        ("S", (-2,2,-9,2,-11,0,-11,-1,-10,-3,-9,-4,-8,-4,-3,-6,-2,-7,-1,-7,0,-9,0,-11,-2,-13,-9,-13,-11,-11)),
        ("S", (-2,1,-4,2,-7,2,-9,1,-11,0,-11,-2,-10,-3,-9,-4,-8,-5,-3,-6,-2,-7,-1,-8,0,-9,0,-11,-2,-13,-9,-13,-11,-11)),
        ("S", (-1,2,-9,2,-10,0,-10,-1,-9,-3,-9,-4,-7,-4,-3,-6,-1,-7,0,-7,0,-11,-1,-12,-3,-13,-6,-13,-9,-12,-10,-11)),
        ("S", (-1,2,-3,3,-6,3,-9,2,-10,0,-10,-1,-9,-3,-7,-4,-3,-6,-1,-6,0,-7,0,-11,-1,-12,-3,-13,-6,-13,-9,-12,-10,-11)),
        ("T", (0,-15,-5,0,6,0)),
        ("0", (-3,-1,-4,-3,-5,-7,-5,-9,-4,-13,-3,-15,0,-16,1,-16,3,-15,5,-13,5,-3,3,-1,1,0,0,0,0,0)),
        ("0", (-3,-1,-4,-3,-5,-6,-5,-9,-4,-12,-3,-15,3,-15,5,-12,6,-9,6,-6,5,-3,3,-1,1,0,0,0,0,0)),
        ("0", (-2,-1,-3,-3,-4,-6,-4,-9,-3,-12,-2,-15,4,-15,6,-12,6,-3,4,-1,2,0,0,0,0,0)),
        ("0", (-2,-1,-4,-3,-4,-13,-2,-15,4,-15,5,-13,6,-9,6,-7,5,-3,4,-1,2,0,0,0,0,0)),
        ("0", (-2,-1,-4,-3,-4,-13,-2,-15,0,-16,1,-16,4,-15,5,-13,6,-9,6,-7,5,-3,4,-1,1,0,0,0,0,0)),
        #("1", (0,1,-3,4,13,4)),
        ("2", (0,3,1,3,3,4,6,4,7,3,8,3,9,1,9,0,8,-2,6,-4,-1,-11,9,-11)),
        ("2", (0,3,1,3,2,4,5,4,7,3,8,3,8,-2,6,-4,-1,-11,9,-11)),
        #("2", (0,2,1,3,7,3,8,2,9,1,9,-1,8,-2,6,-5,-1,-12,9,-12)),
        ("3", (9,0,4,-6,6,-6,8,-7,9,-8,9,-14,7,-15,5,-16,3,-16,0,-15,0,-14,-1,-13)),
        ("3", (8,0,3,-6,7,-6,8,-7,9,-9,9,-11,8,-13,6,-15,0,-15,-1,-14,-2,-12)),
        ("3", (8,0,3,-6,7,-6,8,-7,8,-13,6,-15,0,-15,-1,-14,-2,-12)),
        ("3", (8,0,4,-6,6,-6,8,-7,9,-10,9,-11,8,-13,7,-15,5,-16,2,-16,0,-15,-1,-14,-1,-13)),
        ("5", (-8,0,-8,-6,-5,-5,-3,-5,-1,-6,1,-7,1,-13,-1,-15,-8,-15,-8,-14,-9,-12)),
        ("6", (-1,1,-3,2,-5,2,-7,1,-8,-1,-9,-4,-9,-8,-8,-11,-7,-13,-2,-13,0,-11,1,-9,1,-8,0,-6,-2,-4,-7,-4,-8,-6,-9,-8)),
        ("6", (-1,2,-3,3,-5,3,-7,2,-9,0,-9,-11,-7,-12,-5,-13,-4,-13,-2,-12,0,-11,0,-6,-2,-4,-4,-3,-5,-3,-7,-4,-9,-6,-9,-8)),
        ("9", (-1,-2,-2,-4,-4,-5,-5,-5,-7,-4,-9,-2,-9,3,-7,4,-5,5,-4,5,-2,4,-1,3,0,0,0,-4,-1,-8,-2,-10,-8,-10,-9,-8)),
        ("4T T", (-7,-10,4,-10), (0,-15,11,0,11,-15,6,0,17,0,-29,-23,-29,-39,-34,-23,-24,-23)),
        ("ATH", (6,16,12,0), (8,0,16,11,16,-5,11,11,22,11,25,11,25,-5,36,11,36,-5,25,3,36,3)),

        # W20_5 room number font
        ("B", (0,1.86,0.78,1.86,1.14,1.68,1.2,1.5,1.2,1.32,1.14,1.14,1.02,1.02,0.78,0.96,0,0.96), (0.24,-0.12,0.36,-0.18,0.42,-0.36,0.42,-0.6,0.36,-0.78,0.24,-0.9,0,-0.96,-0.78,-0.96)),
        ("C", (-0.12,0.18,-0.3,0.36,-0.48,0.42,-0.84,0.42,-1.02,0.36,-1.2,0.18,-1.26,0,-1.38,-0.3,-1.38,-0.72,-0.3,-1.38,-0.12,-1.2,0,-1.02)), # curve
        ("C", (-0.12,0.18,-0.3,0.36,-1.26,0,-1.32,-0.3,-1.32,-0.72,-1.26,-0.96,-1.14,-1.14,-0.96,-1.32,-0.78,-1.44,-0.48,-1.44,-0.3,-1.32,-0.12,-1.14,0,-0.96)), # curve
        ("C", (-0.06,0.18,-0.24,0.36,-1.26,0,-1.32,-0.3,-1.32,-0.72,-0.24,-1.38,-0.06,-1.2,0,-1.02)), # curve
        ("D", (0,1.86,0.66,1.86,0.9,1.8,1.08,1.62,1.2,1.44,1.26,1.14,1.26,0.72,1.2,0.42,1.08,0.24,0.9,0.06,0.66,0,0,0,0,0)),
        ("D", (0,1.92,0.66,1.92,0.66,0,0,0,0,0)), # curve
        ("F", (0,1.86,1.2,1.86), (0.72,0)),
        ("G", (-0.06,0.12,-0.24,0.3,-0.42,0.42,-0.78,0.42,-0.96,0.3,-1.26,0,-1.32,-0.3,-1.32,-0.72,-1.26,-1.02,-1.14,-1.2,-0.96,-1.38,-0.78,-1.44,-0.42,-1.44,-0.24,-1.38,-0.06,-1.2,0,-1.02,0,-0.72,-0.42,-0.72)),
        ("G", (-0.06,0.18,-0.24,0.36,-1.32,-0.24,-1.32,-0.72,-1.26,-0.96,-0.06,-1.14,0,-0.96,0,-0.72,-0.42,-0.72)), # curve
        ("G", (-0.24,0.36,-1.32,-0.3,-1.32,-0.72,-1.26,-1.02,-1.14,-1.2,-0.96,-1.38,-0.78,-1.44,-0.42,-1.44,-0.24,-1.38,0,-1.02,0,-0.72,-0.42,-0.72)), # curve
        ("O", (-0.18,-0.06,-0.36,-0.24,-0.42,-0.42,-0.54,-0.66,-0.54,-1.14,-0.42,-1.38,-0.36,-1.56,-0.18,-1.74,0,-1.86,0.36,-1.86,0.54,-1.74,0.72,-1.56,0.84,-1.38,0.9,-1.14,0.9,-0.66,0.84,-0.42,0.72,-0.24,0.54,-0.06,0.36,0,0,0,0,0)),
        ("O", (-0.18,-0.12,-0.3,-0.24,-0.42,-0.42,-0.48,-0.72,-0.48,-1.14,-0.42,-1.44,-0.18,-1.8,0,-1.86,0.36,-1.86,0.54,-1.8,0.72,-1.62,0.84,-1.44,0.9,-1.14,0.9,-0.72,0.84,-0.42,0.72,-0.24,0.54,-0.12,0.36,0,0,0,0,0)),
        ("O", (-0.18,-0.12,-0.36,-0.3,0,-1.92,0.3,-1.92,0.48,-1.8,0.66,-1.62,0.78,-1.44,0.84,-1.2,0.84,-0.72,0.78,-0.48,0.66,-0.3,0.48,-0.12,0.3,0,0,0,0,0)), # curve
        ("O", (-0.18,-0.06,-0.36,-0.24,-0.42,-0.42,-0.54,-0.72,-0.54,-1.14,-0.42,-1.44,-0.36,-1.62,-0.18,-1.8,0,-1.86,0.36,-1.86,0.54,-1.8,0.54,-0.06,0.36,0,0,0,0,0)), # curve
        ("O", (-0.18,-0.12,-0.36,-0.3,0,-1.92,0.36,-1.92,0.54,-1.8,0.54,-0.12,0.36,0,0,0,0,0)), # curve
        ("P", (0,1.92,0.84,1.92,1.2,1.74,1.26,1.56,1.26,1.26,1.2,1.08,1.08,1.02,0.84,0.9,0,0.9)),
        ("Q", (-0.18,-0.12,-0.36,-0.3,-0.42,-0.48,-0.54,-0.72,-0.54,-1.2,-0.42,-1.44,-0.36,-1.62,-0.18,-1.8,0,-1.92,0.36,-1.92,0.54,-1.8,0.72,-1.62,0.84,-1.44,0.9,-1.2,0.9,-0.72,0.84,-0.48,0.72,-0.3,0.54,-0.12,0.36,0,0,0,0,0), (0.54,-0.54)),
        ("Q", (-0.18,-0.12,-0.36,-0.24,-0.42,-0.42,-0.54,-0.72,-0.54,-1.14,-0.42,-1.44,-0.36,-1.62,-0.18,-1.8,0,-1.86,0.36,-1.86,0.54,-1.8,0.72,-1.62,0.84,-1.44,0.9,-1.14,0.9,-0.72,0.84,-0.42,0.72,-0.24,0.36,0,0,0,0,0), (0.54,-0.54)),
        ("R", (0,1.92,0.84,1.92,1.2,1.74,1.26,1.56,1.26,1.38,1.2,1.2,1.08,1.08,0.84,1.02,0,1.02), (0.6,-1.02)),
        ("S", (-0.18,0.18,-0.48,0.3,-0.84,0.3,-1.08,0.18,-1.26,0,-1.26,-0.18,-1.14,-0.36,-1.08,-0.42,-0.9,-0.54,-0.18,-0.78,-0.12,-0.9,0,-1.08,0,-1.32,-0.18,-1.5,-0.48,-1.62,-0.84,-1.62,-1.08,-1.5,-1.26,-1.32)),
        ("S", (-0.18,0.18,-0.42,0.24,-0.78,0.24,-1.02,0.18,-1.2,0,-1.2,-0.18,-1.14,-0.36,-1.02,-0.42,-0.84,-0.54,-0.36,-0.72,-0.78,-1.62,-1.02,-1.5,-1.2,-1.32)), # curve
        ("V", (0.66,-1.86,1.38,0)),
        ("Y", (0.72,-0.9,0.72,-1.86), (-0.66,-0.9)),
        ("0", (-0.24,-0.12,-0.42,-0.36,-0.48,-0.84,-0.48,-1.08,-0.42,-1.56,-0.24,-1.8,0,-1.92,0.18,-1.92,0.48,-1.8,0.66,-1.56,0.72,-1.08,0.72,-0.84,0.66,-0.36,0.48,-0.12,0.18,0,0,0,0,0)),
        ("0", (-0.3,-0.12,-0.48,-0.36,-0.54,-0.84,-0.54,-1.08,-0.48,-1.5,-0.3,-1.8,0,-1.86,0.18,-1.86,0.42,-1.8,0.42,-0.12,0.18,0,0,0,0,0)), # curve
        ("4", (-0.9,-1.26,0.48,-1.26), (0,-1.92)),
        ("5", (-0.84,0,-0.96,-0.78,-0.6,-0.6,-0.36,-0.6,-0.06,-0.72,0.12,-0.9,0.18,-1.14,0.18,-1.32,0.12,-1.62,-0.06,-1.8,-0.36,-1.86,-0.6,-1.86,-0.84,-1.8,-0.96,-1.68,-1.02,-1.5)),
        ("5", (-0.9,0,-1.02,-0.84,-0.9,-0.72,-0.66,-0.66,-0.36,-0.66,-0.12,-0.72,0.06,-0.9,0.18,-1.2,0.18,-1.38,0.06,-1.62,-0.12,-1.8,-0.36,-1.92,-0.66,-1.92,-1.02,-1.74,-1.08,-1.56)),
        ("5", (-0.9,0,-1.02,-0.78,-0.66,-0.6,-0.36,-0.6,-0.12,-0.72,0.06,-0.9,0.18,-1.14,0.18,-1.32,0.06,-1.56,-0.12,-1.74,-0.36,-1.86,-0.66,-1.86,-1.02,-1.68,-1.08,-1.5)),
        ("5", (-0.84,0,-0.96,-0.84,-0.84,-0.72,-0.96,-1.74,-1.02,-1.56)), # curve
        #("5", (-0.9,0,-0.96,-0.78,-0.9,-0.72,-0.6,-0.6,-0.36,-0.6,-0.06,-0.72,0.12,-0.9,0.18,-1.14,0.18,-1.32,0.12,-1.62,-0.06,-1.8,-0.36,-1.86,-0.6,-1.86,-0.9,-1.8,-0.96,-1.68,-1.08,-1.5)),
        # W20_5 stair font
        ("P", (0,2.28,0.72,2.28,0.72,1.08,0,1.08)), # curve
        ("N", (0,2.22,1.08,0,1.08,2.22)),
        ("S", (-0.06,0.12,-0.18,0.18,-0.36,0.18,-0.48,0.12,-0.54,0,-0.54,-0.18,-0.42,-0.3,-0.06,-0.48,-0.06,-0.54,0,-0.6,0,-0.78,-0.06,-0.9,-0.18,-0.96,-0.36,-0.96,-0.48,-0.9,-0.54,-0.78)),
        # NE83_4 room number font
        ("B", (0,1.86,0.84,1.86,1.08,1.8,1.2,1.68,1.26,1.5,1.26,1.32,1.2,1.14,0.84,0.96,0,0.96), (0.24,-0.06,0.36,-0.18,0.42,-0.36,0.42,-0.6,0.36,-0.78,0,-0.96,-0.84,-0.96)),
        ("B", (0,3.78,1.62,3.78,2.16,3.6,2.34,3.42,2.52,3.06,2.52,2.7,2.34,2.34,2.16,2.16,1.62,1.98,0,1.98), (0.54,-0.18,0.72,-0.36,0.9,-0.72,0.9,-1.26,0.72,-1.62,0.54,-1.8,0,-1.98,-1.62,-1.98)),
        ("Y", (1.74,-2.1,3.48,0), (0,-2.4)),
]:
        # # 50_0 stair font
        # ("U", (-27,0,-32,-1,-36,-4,-37,-8,-37,-11,-36,-15,-32,-17,-27,-19,0,-19)),
        # #("P", (37,0,37,-12,35,-17,34,-18,30,-19,25,-19,21,-18,19,-17,18,-12,18,0)),
        # # 50_0 room number font
        # ("A", (31,-12,0,-24), (0,-15)),
        # ("B", (-1,-4,-3,-5,-6,-7,-10,-7,-13,-5,-14,-4,-16,0,-16,14,15,14,15,0,14,-4,12,-5,9,-7,6,-7,3,-5,2,-4,0,0,0,14)),
        # ("C", (3,2,6,5,7,8,7,14,6,17,3,20,0,21,-5,23,-12,23,-17,21,-19,20,-22,17,-24,14,-24,8,-22,5,-19,2,-17,0)),
        # ("D", (31,0,31,-10,29,-15,26,-18,23,-19,19,-21,11,-21,7,-19,4,-18,1,-15,0,-10,0,0)),
        # ("E", (32,0,32,-20), (0,-12), (0,-20)),
        # ("F", (31,0,31,-20), (0,-12)),
        # ("G", (3,2,6,5,7,8,7,14,6,17,3,20,0,21,-5,23,-12,23,-17,21,-20,20,-23,17,-24,14,-24,8,-23,5,-20,2,-17,0,-12,0,-12,8)),
        # ("I", (-31,0)),
        # ("J", (-24,0,-28,2,-30,3,-31,6,-31,9,-30,12,-28,14,-24,15,-21,15)),
        # ("K", (-31,0), (-21,20), (-18,-13)),
        # ("L", (-32,0,-32,-18)),
        # ("M", (31,0,0,-12,31,-24,0,-24)),
        # ("N", (31,0,0,-21,31,-21)),
        # ("O", (-1,3,-4,6,-7,8,-11,9,-19,9,-23,8,-26,6,-29,3,-31,0,-31,-6,-29,-9,-26,-12,-23,-13,-19,-15,-11,-15,-7,-13,-4,-12,-1,-9,0,-6,0,0)),
        # #("P", (31,0,31,-13,29,-18,28,-19,25,-21,20,-21,17,-19,16,-18,14,-13,14,0)),
        # ("R", (31,0,31,-13,30,-18,28,-19,25,-21,22,-21,19,-19,18,-18,16,-13,16,0), (-16,-11)),
        # ("S", (3,3,5,8,5,14,3,18,0,21,-3,21,-6,20,-7,18,-9,15,-12,6,-13,3,-15,2,-18,0,-22,0,-25,3,-26,8,-26,14,-25,18,-22,21)),
        # ("T", (-32,0), (0,-21)),
        # ("U", (-22,0,-27,-2,-30,-5,-31,-9,-31,-12,-30,-17,-27,-20,-22,-21,0,-21)),
        # ("V", (-32,-12,0,-24)),
        # ("V", (-75,-22,0,-43)),
        # ("W", (-75,-13,0,-27,-75,-40,0,-53)),
        # ("Y", (-36,-22,0,-43), (-39,0)),
        # ("0", (-1,5,-6,8,-13,9,-17,9,-25,8,-29,5,-31,0,-31,-3,-29,-7,-25,-10,-17,-12,-13,-12,-6,-10,-1,-7,0,-3,0,0)),
        # ("1", (1,-3,6,-8,-26,-8)),
        # ("2", (1,0,4,-1,6,-3,7,-6,7,-12,6,-15,4,-16,1,-18,-2,-18,-5,-16,-9,-13,-24,1,-24,-19)),
        # ("3", (0,-16,-12,-8,-12,-12,-14,-15,-15,-16,-20,-18,-23,-18,-27,-16,-30,-13,-32,-9,-32,-5,-30,0,-29,1,-26,3)),
        # ("4", (31,0,11,15,11,-7)),
        # ("5", (0,15,-13,16,-12,15,-10,10,-10,6,-12,1,-15,-2,-19,-3,-22,-3,-26,-2,-29,1,-31,6,-31,10,-29,15,-28,16,-25,18)),
        # ("6", (3,1,4,6,4,9,3,13,-2,16,-9,17,-16,17,-22,16,-25,13,-27,9,-27,7,-25,3,-22,0,-18,-2,-16,-2,-12,0,-9,3,-8,7,-8,9,-9,13,-12,16,-16,17)),
        # ("7", (31,-15,31,6)),
        # ("8", (-2,4,-5,6,-8,6,-11,4,-12,1,-14,-5,-15,-9,-18,-12,-21,-14,-26,-14,-29,-12,-30,-11,-32,-6,-32,0,-30,4,-29,6,-26,7,-21,7,-18,6,-15,3,-14,-2,-12,-8,-11,-11,-8,-12,-5,-12,-2,-11,0,-6,0,0)),
        # ("9", (-4,1,-7,4,-8,9,-8,10,-7,15,-4,18,0,19,2,19,6,18,9,15,11,10,11,9,9,4,6,1,0,0,-7,0,-14,1,-19,4,-20,9,-20,12,-19,16,-16,18)),
        # ("/", (-48,27)),
        # ("-", (0,-27)),
        # # 50_0 footer
        # ("a", (-40,0), (6,6,9,12,9,20,6,26,0,32,-8,34,-14,34,-23,32,-28,26,-31,20,-31,12,-28,6,-23,0)),
        # ("c", (6,6,9,12,9,20,6,26,0,32,-8,35,-14,35,-23,32,-28,26,-31,20,-31,12,-28,6,-23,0)),
        # ("d", (-60,0), (6,5,9,11,9,20,6,25,0,31,-8,34,-14,34,-23,31,-28,25,-31,20,-31,11,-28,5,-23,0)),
        # ("e", (0,-34,5,-34,11,-31,14,-28,17,-23,17,-14,14,-8,8,-3,0,0,-6,0,-15,-3,-20,-8,-23,-14,-23,-23,-20,-28,-15,-34)),
        # ("f", (0,6,-3,12,-12,14,-60,14), (0,-20)),
        # ("g", (-46,0,-54,3,-57,6,-60,11,-60,20,-57,26), (6,6,9,11,9,20,6,26,0,31,-8,34,-14,34,-23,31,-28,26,-31,20,-31,11,-28,6,-23,0)),
        # ("h", (-60,0), (9,-8,12,-14,12,-23,9,-28,0,-31,-28,-31)),
        # ("i", (-3,-3,0,-6,3,-3,0,0), (-37,0)),
        # ("n", (-40,0), (9,-9,12,-14,12,-23,9,-29,0,-32,-28,-32)),
        # ("o", (-3,6,-9,12,-17,15,-23,15,-32,12,-37,6,-40,0,-40,-8,-37,-14,-32,-20,-23,-23,-17,-23,-9,-20,-3,-14,0,-8,0,0)),
        # ("p", (-60,0), (6,-5,9,-11,9,-20,6,-25,0,-31,-8,-34,-14,-34,-23,-31,-28,-25,-31,-20,-31,-11,-28,-5,-23,0)),
        # ("r", (-40,0), (8,-2,14,-8,17,-14,17,-22)),
        # ("s", (6,3,9,11,9,20,6,28,0,31,-5,28,-8,23,-11,8,-14,3,-20,0,-23,0,-28,3,-31,11,-31,20,-28,28,-23,31)),
        # ("t", (-49,0,-57,-3,-60,-9,-60,-14), (0,-20)),
        # ("u", (-29,0,-37,-3,-40,-8,-40,-17,-37,-23,-29,-31), (-40,0)),
        # ("v", (-40,-17,0,-34)),
        # ("y", (-40,-17,0,-34), (-12,5,-17,11,-20,17,-20,20)),
        # (".", (-3,3,-6,0,-3,-3,0,0)),

        # # 64_3 room number font
        # #("1", (1,0,4,3,4,-13)),
        # #("3", (9,0,4,-6,6,-6,8,-7,9,-8,9,-14,7,-15,5,-16,3,-16,0,-15,0,-14,-1,-13)),

        # # W20_5 room number font
        # ("B", (1.86,0,1.86,-0.78,1.68,-1.14,1.5,-1.2,1.32,-1.2,1.14,-1.14,1.02,-1.02,0.96,-0.78,0.96,0), (-0.12,-0.24,-0.18,-0.36,-0.36,-0.42,-0.6,-0.42,-0.78,-0.36,-0.9,-0.24,-0.96,0,-0.96,0.78)),
        # ("C", (0.18,0.12,0.36,0.3,0.42,0.48,0.42,0.84,0.36,1.02,0.18,1.2,0,1.26,-0.3,1.38,-0.72,1.38,-1.38,0.3,-1.2,0.12,-1.02,0)), # curve
        # ("C", (0.18,0.12,0.36,0.3,0,1.26,-0.3,1.32,-0.72,1.32,-0.96,1.26,-1.14,1.14,-1.32,0.96,-1.44,0.78,-1.44,0.48,-1.32,0.3,-1.14,0.12,-0.96,0)), # curve
        # ("C", (0.18,0.06,0.36,0.24,0,1.26,-0.3,1.32,-0.72,1.32,-1.38,0.24,-1.2,0.06,-1.02,0)), # curve
        # ("D", (1.86,0,1.86,-0.66,1.8,-0.9,1.62,-1.08,1.44,-1.2,1.14,-1.26,0.72,-1.26,0.42,-1.2,0.24,-1.08,0.06,-0.9,0,-0.66,0,0,0,0)),
        # ("D", (1.92,0,1.92,-0.66,0,-0.66,0,0,0,0)), # curve
        # ("F", (1.86,0,1.86,-1.2), (0,-0.72)),
        # ("G", (0.12,0.06,0.3,0.24,0.42,0.42,0.42,0.78,0.3,0.96,0,1.26,-0.3,1.32,-0.72,1.32,-1.02,1.26,-1.2,1.14,-1.38,0.96,-1.44,0.78,-1.44,0.42,-1.38,0.24,-1.2,0.06,-1.02,0,-0.72,0,-0.72,0.42)),
        # ("G", (0.18,0.06,0.36,0.24,-0.24,1.32,-0.72,1.32,-0.96,1.26,-1.14,0.06,-0.96,0,-0.72,0,-0.72,0.42)), # curve
        # ("G", (0.36,0.24,-0.3,1.32,-0.72,1.32,-1.02,1.26,-1.2,1.14,-1.38,0.96,-1.44,0.78,-1.44,0.42,-1.38,0.24,-1.02,0,-0.72,0,-0.72,0.42)), # curve
        # ("O", (-0.06,0.18,-0.24,0.36,-0.42,0.42,-0.66,0.54,-1.14,0.54,-1.38,0.42,-1.56,0.36,-1.74,0.18,-1.86,0,-1.86,-0.36,-1.74,-0.54,-1.56,-0.72,-1.38,-0.84,-1.14,-0.9,-0.66,-0.9,-0.42,-0.84,-0.24,-0.72,-0.06,-0.54,0,-0.36,0,0,0,0)),
        # ("O", (-0.12,0.18,-0.24,0.3,-0.42,0.42,-0.72,0.48,-1.14,0.48,-1.44,0.42,-1.8,0.18,-1.86,0,-1.86,-0.36,-1.8,-0.54,-1.62,-0.72,-1.44,-0.84,-1.14,-0.9,-0.72,-0.9,-0.42,-0.84,-0.24,-0.72,-0.12,-0.54,0,-0.36,0,0,0,0)),
        # ("O", (-0.12,0.18,-0.3,0.36,-1.92,0,-1.92,-0.3,-1.8,-0.48,-1.62,-0.66,-1.44,-0.78,-1.2,-0.84,-0.72,-0.84,-0.48,-0.78,-0.3,-0.66,-0.12,-0.48,0,-0.3,0,0,0,0)), # curve
        # ("O", (-0.06,0.18,-0.24,0.36,-0.42,0.42,-0.72,0.54,-1.14,0.54,-1.44,0.42,-1.62,0.36,-1.8,0.18,-1.86,0,-1.86,-0.36,-1.8,-0.54,-0.06,-0.54,0,-0.36,0,0,0,0)), # curve
        # ("O", (-0.12,0.18,-0.3,0.36,-1.92,0,-1.92,-0.36,-1.8,-0.54,-0.12,-0.54,0,-0.36,0,0,0,0)), # curve
        # ("P", (1.92,0,1.92,-0.84,1.74,-1.2,1.56,-1.26,1.26,-1.26,1.08,-1.2,1.02,-1.08,0.9,-0.84,0.9,0)),
        # ("Q", (-0.12,0.18,-0.3,0.36,-0.48,0.42,-0.72,0.54,-1.2,0.54,-1.44,0.42,-1.62,0.36,-1.8,0.18,-1.92,0,-1.92,-0.36,-1.8,-0.54,-1.62,-0.72,-1.44,-0.84,-1.2,-0.9,-0.72,-0.9,-0.48,-0.84,-0.3,-0.72,-0.12,-0.54,0,-0.36,0,0,0,0), (-0.54,-0.54)),
        # ("Q", (-0.12,0.18,-0.24,0.36,-0.42,0.42,-0.72,0.54,-1.14,0.54,-1.44,0.42,-1.62,0.36,-1.8,0.18,-1.86,0,-1.86,-0.36,-1.8,-0.54,-1.62,-0.72,-1.44,-0.84,-1.14,-0.9,-0.72,-0.9,-0.42,-0.84,-0.24,-0.72,0,-0.36,0,0,0,0), (-0.54,-0.54)),
        # ("R", (1.92,0,1.92,-0.84,1.74,-1.2,1.56,-1.26,1.38,-1.26,1.2,-1.2,1.08,-1.08,1.02,-0.84,1.02,0), (-1.02,-0.6)),
        # ("S", (0.18,0.18,0.3,0.48,0.3,0.84,0.18,1.08,0,1.26,-0.18,1.26,-0.36,1.14,-0.42,1.08,-0.54,0.9,-0.78,0.18,-0.9,0.12,-1.08,0,-1.32,0,-1.5,0.18,-1.62,0.48,-1.62,0.84,-1.5,1.08,-1.32,1.26)),
        # ("S", (0.18,0.18,0.24,0.42,0.24,0.78,0.18,1.02,0,1.2,-0.18,1.2,-0.36,1.14,-0.42,1.02,-0.54,0.84,-0.72,0.36,-1.62,0.78,-1.5,1.02,-1.32,1.2)), # curve
        # ("V", (-1.86,-0.66,0,-1.38)),
        # ("Y", (-0.9,-0.72,-1.86,-0.72), (-0.9,0.66)),
        # ("0", (-0.12,0.24,-0.36,0.42,-0.84,0.48,-1.08,0.48,-1.56,0.42,-1.8,0.24,-1.92,0,-1.92,-0.18,-1.8,-0.48,-1.56,-0.66,-1.08,-0.72,-0.84,-0.72,-0.36,-0.66,-0.12,-0.48,0,-0.18,0,0,0,0)),
        # ("0", (-0.12,0.3,-0.36,0.48,-0.84,0.54,-1.08,0.54,-1.5,0.48,-1.8,0.3,-1.86,0,-1.86,-0.18,-1.8,-0.42,-0.12,-0.42,0,-0.18,0,0,0,0)), # curve
        # ("4", (-1.26,0.9,-1.26,-0.48), (-1.92,0)),
        # ("5", (0,0.84,-0.78,0.96,-0.6,0.6,-0.6,0.36,-0.72,0.06,-0.9,-0.12,-1.14,-0.18,-1.32,-0.18,-1.62,-0.12,-1.8,0.06,-1.86,0.36,-1.86,0.6,-1.8,0.84,-1.68,0.96,-1.5,1.02)),
        # ("5", (0,0.9,-0.84,1.02,-0.72,0.9,-0.66,0.66,-0.66,0.36,-0.72,0.12,-0.9,-0.06,-1.2,-0.18,-1.38,-0.18,-1.62,-0.06,-1.8,0.12,-1.92,0.36,-1.92,0.66,-1.74,1.02,-1.56,1.08)),
        # ("5", (0,0.9,-0.78,1.02,-0.6,0.66,-0.6,0.36,-0.72,0.12,-0.9,-0.06,-1.14,-0.18,-1.32,-0.18,-1.56,-0.06,-1.74,0.12,-1.86,0.36,-1.86,0.66,-1.68,1.02,-1.5,1.08)),
        # ("5", (0,0.84,-0.84,0.96,-0.72,0.84,-1.74,0.96,-1.56,1.02)), # curve
        # #("5", (0,0.9,-0.78,0.96,-0.72,0.9,-0.6,0.6,-0.6,0.36,-0.72,0.06,-0.9,-0.12,-1.14,-0.18,-1.32,-0.18,-1.62,-0.12,-1.8,0.06,-1.86,0.36,-1.86,0.6,-1.8,0.9,-1.68,0.96,-1.5,1.08)),
        # # W20_5 stair font
        # ("P", (2.28,0,2.28,-0.72,1.08,-0.72,1.08,0)), # curve
        # ("N", (2.22,0,0,-1.08,2.22,-1.08)),
        # ("S", (0.12,0.06,0.18,0.18,0.18,0.36,0.12,0.48,0,0.54,-0.18,0.54,-0.3,0.42,-0.48,0.06,-0.54,0.06,-0.6,0,-0.78,0,-0.9,0.06,-0.96,0.18,-0.96,0.36,-0.9,0.48,-0.78,0.54)),
        # # NE83_4 room number font
        # ("B", (1.86,0,1.86,-0.84,1.8,-1.08,1.68,-1.2,1.5,-1.26,1.32,-1.26,1.14,-1.2,0.96,-0.84,0.96,0), (-0.06,-0.24,-0.18,-0.36,-0.36,-0.42,-0.6,-0.42,-0.78,-0.36,-0.96,0,-0.96,0.84)),
        # ("B", (3.78,0,3.78,-1.62,3.6,-2.16,3.42,-2.34,3.06,-2.52,2.7,-2.52,2.34,-2.34,2.16,-2.16,1.98,-1.62,1.98,0), (-0.18,-0.54,-0.36,-0.72,-0.72,-0.9,-1.26,-0.9,-1.62,-0.72,-1.8,-0.54,-1.98,0,-1.98,1.62)),
        # ("Y", (-2.1,-1.74,0,-3.48), (-2.4,0)),
    #]:
    shapes = list(shapes)
    k = shapes.pop(0)
    l = len(shapes[0])//2
    shapes = [np.array(v, dtype='f').reshape((-1, 2)) for v in shapes]
    r = np.min(shapes[0])-np.max(shapes[0])
    shapes = [s/r for s in shapes]
    if l not in _KNOWN_SHAPES:
        _KNOWN_SHAPES[l] = list()
    _KNOWN_SHAPES[l].append((k, shapes))

for shapes in _KNOWN_SHAPES.values():
    shapes.sort(key=lambda t: len(t[1]), reverse=True)
