from collections import namedtuple
import numpy as np

from pdf2svg import Path

CHAR_POINTS = 32
KNOWN_SHAPES = list()

_SAMPLE_CHARS = r"""
# N.B. path coordinates are in PDF coordinates, so positive Y is pointing up.

# 50_0 large
A L22,75L43,0 M8,25L35,25
D L0,75L19,75L27,71L33,64L35,57L38,46L38,29L35,18L33,11L27,4L19,0L0,0
K L0,37 M18,37L0,12 M6,21L18,0
E L0,75L35,75 M0,39L21,39 M0,0L35,0
R L0,75L24,75L32,71L35,68L37,61L37,54L35,46L32,43L24,39L0,39 M18,39L37,0
I L0,75

# 50_0 stair font
N L0,38L19,0L19,38
P L0,38L12,38L16,36L17,34L18,31L18,25L17,22L16,20L12,18L0,18
P L0,38L12,38L16,36L17,34L19,31L19,25L17,22L16,20L12,18L0,18

# 50_0 room number font
A L12,31L24,0 M4,10L19,10
B L4,-1L5,-3L7,-6L7,-10L5,-13L4,-14L0,-16L-14,-16L-14,15L0,15L4,14L5,12L7,9L7,6L5,3L4,2L0,0L-14,0
C L-1,3L-4,6L-7,8L-13,8L-16,6L-19,3L-21,0L-22,-4L-22,-12L-21,-16L-19,-19L-16,-22L-13,-24L-7,-24L-4,-22L-1,-19L0,-16
D L0,31L11,31L15,29L18,26L20,23L21,19L21,11L20,7L18,4L15,1L11,0L0,0
E L0,31L19,31 M0,16L12,16 M0,0L19,0
F L0,31L19,31 M0,16L12,16
G L-1,3L-4,7L-7,8L-12,8L-15,7L-18,3L-19,0L-20,-6L-20,-15L-19,-20L-18,-24L-15,-27L-12,-29L-7,-29L-4,-27L-1,-24L0,-20L0,-15L-7,-15
G L-2,3L-5,6L-8,7L-14,7L-17,6L-20,3L-21,0L-23,-5L-23,-12L-21,-17L-20,-20L-17,-23L-14,-24L-8,-24L-5,-23L-2,-20L0,-17L0,-12L-8,-12
H L0,38 M19,38L19,0 M0,20L19,20
H L0,-31 M21,0L21,-31 M0,-15L21,-15
#I L0,-31
J L0,-24L-1,-28L-3,-30L-6,-31L-9,-31L-12,-30L-13,-28L-15,-24L-15,-21
K L0,-31 M20,0L0,-21 M7,-13L20,-31
K L0,37 M18,37L0,18 M6,21L18,0
L L0,-31L18,-31
M L0,32L11,0L23,32L23,0
N L0,31L21,0L21,31
O L-3,-1L-6,-4L-7,-7L-9,-11L-9,-19L-7,-23L-6,-26L-3,-29L0,-31L6,-31L9,-29L12,-26L14,-23L15,-19L15,-11L14,-7L12,-4L9,-1L6,0L0,0
P L0,32L14,32L18,30L20,29L21,26L21,21L20,18L18,17L14,15L0,15
Q L-0.36,-0.24C-0.694,-0.626,-0.635,-1.759,0,-1.86L0.36,-1.86L0.54,-1.8C0.998,-1.444,1.063,-0.421,0.54,-0.12L0.36,0L0,0Z M0.24,-1.5L0.78,-2.04
R L0,31L13,31L17,30L19,28L20,25L20,23L19,20L17,18L13,17L0,17 M10,17L20,0
S L-3,3L-8,5L-14,5L-18,3L-21,0L-21,-3L-20,-6L-18,-7L-15,-9L-6,-12L-3,-13L-2,-15L0,-18L0,-22L-3,-25L-8,-26L-14,-26L-18,-25L-21,-22
T L0,-32 M-11,0L10,0
U L0,-22L2,-27L4,-30L9,-31L12,-31L16,-30L19,-27L21,-22L21,0
V L12,-31L24,0
W L7,-37L14,0L20,-37L27,0
Y L11,-17L21,0 M11,-17L11,-37
Z L-21,0L0,31L-21,31

0 L-4,-2L-7,-6L-9,-14L-9,-18L-7,-25L-4,-30L0,-31L3,-31L8,-30L11,-25L12,-18L12,-14L11,-6L8,-2L3,0L0,0
1 L3,2L7,6L7,-25
2 L0,1L1,4L3,6L6,7L12,7L15,6L16,4L18,1L18,-2L16,-5L13,-9L-1,-24L19,-24
3 L17,0L8,-12L12,-12L15,-14L17,-15L18,-19L18,-22L17,-27L14,-30L9,-31L5,-31L0,-30L-1,-28L-3,-25
4 L0,31L-15,11L7,11
5 L-15,0L-16,-13L-15,-12L-10,-10L-6,-10L-1,-12L2,-15L3,-19L3,-22L2,-26L-1,-29L-6,-31L-10,-31L-15,-29L-16,-28L-18,-25
6 L-1,3L-6,4L-9,4L-13,3L-16,-2L-17,-9L-17,-16L-16,-22L-13,-25L-9,-27L-7,-27L-3,-25L0,-22L2,-18L2,-16L0,-12L-3,-9L-7,-8L-9,-8L-13,-9L-16,-12L-17,-16
7 L15,31L-6,31
8 L-4,-1L-6,-4L-6,-7L-4,-10L-1,-12L5,-13L9,-15L12,-18L14,-21L14,-25L12,-28L11,-29L6,-31L0,-31L-4,-29L-6,-28L-7,-25L-7,-21L-6,-18L-3,-15L2,-13L8,-12L11,-10L12,-7L12,-4L11,-1L6,0L0,0
9 L-1,-4L-4,-7L-9,-8L-10,-8L-15,-7L-18,-4L-19,0L-19,2L-18,6L-15,9L-10,11L-9,11L-4,9L-1,6L0,0L0,-7L-1,-14L-4,-19L-9,-20L-12,-20L-16,-19L-18,-16

/ L-27,-48
- L26,0

# 50_0 footer font
i L3,-3L6,0L3,3L0,0 M3,-20L3,-57
o L-6,-3L-12,-9L-15,-17L-15,-23L-12,-32L-6,-37L0,-40L8,-40L14,-37L20,-32L23,-23L23,-17L20,-9L14,-3L8,0L0,0
. L-3,-3L0,-6L3,-3L0,0

# W20 footer font
a L0,-2.4 M0,-0.54L-0.36,-0.18C-2.561,0.979,-2.561,-3.379,-0.36,-2.22L0,-1.86
c L-0.3,0.36C-2.595,1.451,-2.595,-2.771,-0.3,-1.68L0,-1.32
d L0,-3.6 M0,-1.74L-0.36,-1.38C-2.606,-0.259,-2.606,-4.541,-0.36,-3.42L0,-3.06
e L2.04,0L2.04,0.36L1.86,0.66L1.68,0.84L1.38,1.02L0.84,1.02L0.48,0.84L0.18,0.48L0,0L0,-0.36L0.18,-0.84L0.48,-1.2L0.84,-1.38L1.38,-1.38L1.68,-1.2L2.04,-0.84
f L-0.36,0L-0.72,-0.18L-0.9,-0.66L-0.9,-3.6 M-1.38,-1.2L-0.18,-1.2
g L0,-2.76L-0.18,-3.24L-0.36,-3.42L-0.72,-3.6L-1.2,-3.6L-1.56,-3.42 M0,-0.54L-0.36,-0.18C-2.606,0.941,-2.606,-3.341,-0.36,-2.22L0,-1.86
h L0,-3.6 M0,-1.86L0.54,-1.38L0.9,-1.2L1.38,-1.2L1.74,-1.38L1.92,-1.86L1.92,-3.6
i L0.18,-0.18L0.3,0L0.18,0.18L0,0Z M0.18,-1.2L0.18,-3.42
i L0.18,-0.18L0.36,0L0.18,0.18L0,0Z M0.18,-1.2L0.18,-3.42
l L0,-3.6
n L0,-2.4 M0,-0.66L0.54,-0.18L0.84,0L1.38,0L1.74,-0.18L1.86,-0.66L1.86,-2.4
o L-0.36,-0.18L-0.72,-0.54L-0.9,-1.02L-0.9,-1.38L-0.72,-1.86L-0.36,-2.22L0,-2.4L0.48,-2.4L0.84,-2.22L1.2,-1.86L1.32,-1.38L1.32,-1.02L1.2,-0.54L0.84,-0.18L0.48,0L0,0Z
o L-0.36,-0.18L-0.66,-0.54L-0.84,-1.02L-0.84,-1.38L-0.66,-1.92L-0.36,-2.22L0,-2.4L0.54,-2.4L0.84,-2.22L1.2,-1.92L1.38,-1.38L1.38,-1.02L1.2,-0.54L0.84,-0.18L0.54,0L0,0Z
p L0,-3.6 M0,-0.54L0.36,-0.18C2.651,0.907,2.651,-3.307,0.36,-2.22L0,-1.86
r L0,-2.4 M0,-1.02L0.18,-0.54L0.54,-0.18L0.9,0L1.38,0
s L-0.18,0.36L-0.66,0.54L-1.2,0.54L-1.68,0.36L-1.86,0L-1.68,-0.3L-1.38,-0.48L-0.48,-0.66L-0.18,-0.84L0,-1.2L0,-1.32L-0.18,-1.68L-0.66,-1.86L-1.2,-1.86L-1.68,-1.68L-1.86,-1.32
t L0,-2.94L0.18,-3.42L0.48,-3.6L0.84,-3.6 M-0.54,-1.2L0.66,-1.2
u L0,-1.74L0.18,-2.22L0.48,-2.4L1.02,-2.4L1.38,-2.22L1.86,-1.74 M1.86,0L1.86,-2.4
v L1.02,-2.4L2.04,0
y L1.02,-2.4L2.04,0 M1.02,-2.4L0.66,-3.06L0.36,-3.42L0,-3.6L-0.18,-3.6
. L-0.18,-0.18L0,-0.36L0.18,-0.18L0,0Z

# W20 large font
O L-0.12,-0.12L-0.3,-0.36C-0.645,-1.035,-0.53,-1.775,0,-2.28L0.36,-2.28L0.48,-2.16C0.978,-1.556,0.99,-0.721,0.48,-0.12L0.36,0L0,0Z

# W20_0 stair font
S L-0.06,0.12L-0.18,0.18L-0.36,0.18L-0.48,0.12L-0.54,0L-0.54,-0.18L-0.42,-0.3L-0.06,-0.48L-0.06,-0.54L0,-0.6L0,-0.78L-0.06,-0.9L-0.18,-0.96L-0.36,-0.96L-0.48,-0.9L-0.54,-0.78
T L0,-1.14 M-0.3,0L0.3,0
U L0,-1.62L0.06,-1.98L0.24,-2.16L0.48,-2.28L0.6,-2.28L0.84,-2.16L1.02,-1.98L1.08,-1.62L1.08,0
\x23 L-0.3,-1.5 M0.24,0L-0.06,-1.5 M-0.3,-0.6L0.3,-0.6 M-0.3,-0.9L0.24,-0.9
4 L-0.42,-0.72L0.18,-0.72 M0,0L0,-1.14

# W20_0 room number font
B L0,1.86L0.78,1.86L1.08,1.8L1.14,1.68L1.26,1.5L1.26,1.32L1.14,1.14L1.08,1.08L0.78,0.96L0,0.96 M0.78,0.96L1.08,0.9L1.14,0.78L1.26,0.6L1.26,0.36L1.14,0.18L1.08,0.06L0.78,0L0,0
C L-0.12,0.18L-0.3,0.36L-0.48,0.42L-0.84,0.42L-1.02,0.36L-1.2,0.18L-1.26,0L-1.38,-0.3L-1.38,-0.72C-1.191,-1.305,-1.052,-1.613,-0.3,-1.38L-0.12,-1.2L0,-1.02
L L0,-1.86L1.08,-1.86
O L-0.18,-0.06L-0.36,-0.24L-0.42,-0.42L-0.48,-0.72L-0.48,-1.14L-0.42,-1.44L-0.36,-1.62L-0.18,-1.8L0,-1.86L0.36,-1.86L0.54,-1.8L0.72,-1.62L0.84,-1.44L0.9,-1.14L0.9,-0.72L0.84,-0.42L0.72,-0.24L0.54,-0.06L0.36,0L0,0Z
X L1.26,-1.86 M1.26,0L0,-1.86
Z L1.26,1.86L0,1.86 M0,0L1.26,0
0 L-0.24,-0.12L-0.42,-0.36L-0.54,-0.84L-0.54,-1.08L-0.42,-1.5L-0.24,-1.8L0,-1.86L0.18,-1.86L0.48,-1.8L0.66,-1.5L0.72,-1.08L0.72,-0.84L0.66,-0.36L0.48,-0.12L0.18,0L0,0Z
4 L-0.84,-1.26L0.48,-1.26 M0,0L0,-1.92

( L-0.18,-0.18C-0.645,-1.234,-0.717,-2.13,-0.18,-3.18L0,-3.42
) L0.18,-0.18C0.701,-1.092,0.768,-2.283,0.18,-3.18L0,-3.42

# 64 room number font
D L0,15L7,15L9,13L9,12L10,9L10,6L9,3L9,2L7,0L0,0Z
E L0,15L10,15 M0,8L6,8M0,0L10,0
E L0,16L9,16 M0,8L6,8M0,0L9,0
T L0,-15M-5,0L5,0

0 L-3,-1L-4,-3L-5,-7L-5,-9L-4,-13L-3,-15L0,-16L1,-16L3,-15L5,-13L5,-3L3,-1L1,0L0,0Z
0 L-2,-1L-3,-3L-4,-7L-4,-9L-3,-13L-2,-15L0,-16L2,-16L4,-15L5,-13L6,-9L6,-7L5,-3L4,-1L2,0L0,0Z

AI L6,15L12,0 M2,5L9,5M15,15L15,0
ATH L6,16L12,0 M2,6L10,6M19,16L19,0M13,16L24,16M28,16L28,0M38,16L38,0M28,9L38,9
ATH L6,15L11,0 M2,5L9,5M18,15L18,0M13,15L23,15M27,15L27,0M38,15L38,0M27,8L38,8
TT L0,-56M-14,0L14,0M36,0L36,-56M22,0L50,0
4T\nT L-7,-10L4,-10 M0,0L0,-15M11,0L11,-15M6,0L17,0M-29,-23L-29,-39M-34,-23L-24,-23

# 64 large font
B L0,56L18,56L24,53L26,51L28,45L28,40L26,34L24,32L18,29L0,29 M18,29L24,26L26,24L28,18L28,10L26,5L24,2L18,0L0,0
E L0,56L27,56 M0,30L17,30M0,0L27,0
M L0,56L16,0L32,56L32,0
O L-4,-2L-8,-8L-10,-13L-12,-21L-12,-35L-10,-43L-8,-48L-4,-53L0,-56L8,-56L12,-53L16,-48L18,-43L20,-35L20,-21L18,-13L16,-8L12,-2L8,0L0,0Z
# 54 room number font
Y L2.88,-3.6L2.88,-7.5 M5.76,0L2.88,-3.6

& L0,0.36L-0.36,0.72L-0.72,0.72L-1.08,0.36L-1.44,-0.36L-2.1,-2.1L-2.82,-3.18L-3.54,-3.9L-4.26,-4.26L-5.7,-4.26L-6.42,-3.9L-6.78,-3.54L-7.14,-2.82L-7.14,-2.1L-6.78,-1.38L-6.42,-1.02L-3.9,0.36L-3.54,0.72L-3.18,1.44L-3.18,2.16L-3.54,2.88L-4.26,3.24L-4.98,2.88L-5.34,2.16L-5.34,1.44C-4.972,-0.062,-2.484,-3.817,-1.08,-4.26L-0.36,-4.26L0,-3.9L0,-3.54

# 10 label font
C L-0.12,0.36L-0.42,0.72L-0.66,0.9L-1.2,0.9L-1.44,0.72L-1.74,0.36L-1.86,0L-1.98,-0.54L-1.98,-1.44L-1.86,-1.98L-1.74,-2.34L-1.44,-2.7L-1.2,-2.88L-0.66,-2.88L-0.42,-2.7L-0.12,-2.34L0,-1.98
T L0,-3.78 M-0.9,0L0.96,0

# 10 stair font
D L0,2.28L0.54,2.28C1.389,1.948,1.357,0.357,0.54,0L0,0Z

# 10 scale font
0 L-0.06,-0.06L-0.12,-0.18L-0.12,-0.24L-0.18,-0.36L-0.24,-0.42L-0.24,-0.54L-0.3,-0.6L-0.3,-0.96L-0.36,-1.08C-0.372,-1.528,-0.357,-1.558,-0.3,-1.98L-0.3,-2.1L-0.24,-2.16L-0.24,-2.34L-0.18,-2.46L-0.18,-2.52L-0.12,-2.58L-0.12,-2.64L0,-2.76L0,-2.82L0.06,-2.88L0.18,-2.94L0.24,-3L0.3,-3L0.36,-3.06L0.48,-3.06L0.54,-3.12L0.96,-3.12L1.08,-3.06L1.14,-3.06L1.2,-3L1.32,-2.94L1.38,-2.94L1.38,-2.88L1.5,-2.76L0.54,-2.76L0.24,-2.46L0.24,-2.4L0.18,-2.34C0.195,-2.025,0.086,-2.415,0.12,-1.8L0.06,-1.68L0.06,-1.32C0.117,-0.965,0.132,-1.079,0.12,-0.66L0.18,-0.6L0.18,-0.48C0.2,-0.309,0.354,-0.188,0.48,-0.06L0.6,-0.06L0.66,0L0.78,0 M0.78,0.3L0.66,0.3C0.436,0.332,0.221,0.197,0.06,0.06L0,0L0.9,0L0.96,-0.06L1.02,-0.12L1.08,-0.12L1.2,-0.18L1.26,-0.3L1.26,-0.36L1.32,-0.42L1.32,-0.54L1.38,-0.6L1.38,-0.9L1.44,-1.02L1.44,-1.14C1.466,-1.687,1.402,-1.898,1.32,-2.4L1.26,-2.46L1.26,-2.52L1.2,-2.52C1.095,-2.609,0.994,-2.786,0.84,-2.76L0.78,-2.76L1.5,-2.76L1.62,-2.64C1.676,-2.505,1.778,-2.237,1.8,-2.1L1.8,-1.92L1.86,-1.8L1.86,-0.9L1.8,-0.84L1.8,-0.6L1.74,-0.54L1.74,-0.42L1.68,-0.36L1.68,-0.24L1.56,-0.12L1.5,0L1.32,0.18L1.26,0.24L1.14,0.24L1.02,0.3L0.84,0.3
1 L-0.12,-0.06L-0.24,-0.18L-0.24,-0.54L-0.18,-0.54L-0.12,-0.48L0,-0.42L0.06,-0.42L0.3,-0.3L0.36,-0.24L0.54,-0.06 M0.96,0.66L0.72,0.66L0.54,0.48L0.48,0.36L0.42,0.3L0.36,0.24L0.3,0.18L0.24,0.12L0.12,0.06L0.06,0L0,0L0.54,-0.06L0.54,-2.7L0.96,-2.7
/ L0.96,3.48L1.32,3.48L0.36,0L0,0
3 L-0.42,-0.06L-0.42,-0.12L-0.36,-0.18L-0.36,-0.3L-0.3,-0.36L-0.3,-0.48L-0.18,-0.6L-0.12,-0.72L0.66,-0.66L0.48,-0.66L0.42,-0.6L0.3,-0.6L0.18,-0.48L0.12,-0.36L0.12,-0.3L0.06,-0.24L0.06,-0.12 M1.5,0.72L0.72,0.66L0.84,0.66C1.144,0.671,1.384,0.275,1.38,0L1.38,-0.06L1.32,-0.18L1.32,-0.24L1.26,-0.3L1.2,-0.42L1.08,-0.54L1.02,-0.6L0.96,-0.6L0.84,-0.66L0.66,-0.66L-0.12,-0.72L-0.06,-0.78L0,-0.78L0.12,-0.9L0.18,-0.96L0.36,-0.96L0.48,-1.02L0.66,-1.02C0.95,-1.036,0.922,-1.005,1.2,-0.9L1.26,-0.9L1.32,-0.84L1.44,-0.78L1.56,-0.66L1.62,-0.54L1.74,-0.42L1.74,-0.3L1.8,-0.24L1.8,0.24L1.74,0.36L1.74,0.42L1.68,0.48L1.62,0.6L1.56,0.66 M0.48,1.02L0.42,0.66L0.72,0.66L1.5,0.72L1.44,0.78L1.38,0.78L1.26,0.84L1.2,0.84L1.32,0.96L0.54,1.02 M-0.06,2.16L-0.12,2.1L-0.18,1.98L-0.3,1.86L-0.3,1.74L-0.36,1.68L-0.36,1.56L0.06,1.5L0.06,1.62L0.12,1.74L0.18,1.8L0.18,1.86L0.3,1.98L0.42,2.04L0.48,2.04L0.54,2.1L0.66,2.1 M0.66,2.4L0.3,2.4L0.24,2.34L0.12,2.28L0.06,2.28L-0.06,2.16L0.66,2.1L0.72,2.1L0.84,2.04L0.9,2.04L1.02,1.98L1.08,1.92L1.14,1.8L1.2,1.74L1.2,1.32L1.14,1.26L1.08,1.2L1.02,1.14L0.96,1.14L0.9,1.08L0.78,1.02L0.54,1.02L1.32,0.96L1.38,1.02L1.5,1.08L1.5,1.14L1.62,1.26L1.62,1.8L1.56,1.92L1.5,1.98C1.291,2.401,1.106,2.275,0.96,2.4L0.78,2.4
2 L-0.84,0L-0.72,0L-0.6,-0.06L-0.54,-0.06L-0.48,-0.12L-0.42,-0.18L-0.36,-0.18L-0.3,-0.24L-0.3,-0.36L-0.24,-0.42L-0.24,-0.84L-0.3,-0.9L-0.36,-1.02L-0.42,-1.08L-0.42,-1.14L-0.48,-1.2L-0.54,-1.26L-0.78,-1.5L-0.9,-1.56L-0.96,-1.68L-1.08,-1.74L-1.14,-1.8L-1.26,-1.86L-1.32,-1.92L-1.38,-2.04L-1.86,-2.52L-1.92,-2.64L-1.92,-2.7L-1.98,-2.76L-1.98,-2.88L-2.04,-2.94L-2.04,-3.06L0.24,-3.06L0.24,-2.64L-1.44,-2.64L-1.26,-2.46L-1.26,-2.4L-1.2,-2.34L-1.14,-2.34L-1.08,-2.28L-0.9,-2.1L-0.78,-2.04L-0.78,-1.98L-0.66,-1.92C-0.5,-1.76,-0.34,-1.6,-0.18,-1.44L-0.12,-1.44L-0.12,-1.38L0,-1.26L0.06,-1.14L0.12,-1.08L0.12,-0.96L0.18,-0.9L0.18,-0.3L0.12,-0.24L0.12,-0.18L0.06,-0.06 M-0.84,0.3L-1.2,0.3L-1.32,0.24L-1.38,0.24L-1.5,0.18L-1.62,0.06L-1.68,0L-1.74,-0.06L-1.74,-0.12L-1.86,-0.24L-1.86,-0.36L-1.92,-0.42L-1.92,-0.66L-1.5,-0.72L-1.5,-0.48L-1.44,-0.42L-1.44,-0.36L-1.38,-0.24L-1.32,-0.18L-1.26,-0.18L-1.14,-0.06L-1.02,-0.06L-0.96,0L0,0L-0.12,0.12L-0.24,0.18L-0.3,0.18L-0.36,0.24L-0.48,0.3L-0.72,0.3
" L-0.48,0L-0.48,-0.54L-0.36,-1.2L-0.12,-1.2L0,-0.54 M-0.78,0L-1.2,0L-1.2,-0.54L-1.08,-1.2L-0.84,-1.2L-0.78,-0.54
= L0,0.35999L-2.28,0.35999L-2.28,0Z M0,1.02L0,1.44001L-2.28,1.44001L-2.28,1.02Z
' L-0.12,0.66L-0.12,1.2L0.36,1.2L0.36,0.66L0.24,0L0,0
- L0,0.41998L-1.32,0.41998L-1.32,0Z
"""

class KnownShape(namedtuple('KnownShape', 'char divisor shapes'.split())):
    pass

for line in _SAMPLE_CHARS.splitlines():
    line = line.split('#')[0].strip()
    if not line:
        continue
    line = line.split(' ')
    k = line.pop(0).encode('ascii', 'backslashreplace').decode('unicode-escape')

    paths = [Path(d='M0,0'+d) for d in line]
    shapes = [path.quantize(CHAR_POINTS) for path in paths]

    r = np.max(shapes[0])-np.min(shapes[0])
    shapes = [s/r for s in shapes]

    KNOWN_SHAPES.append(KnownShape(k, r, shapes))

KNOWN_SHAPES.sort(key=lambda t: len(t.shapes), reverse=True)

for shapes in [
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
    ]:
    pass
