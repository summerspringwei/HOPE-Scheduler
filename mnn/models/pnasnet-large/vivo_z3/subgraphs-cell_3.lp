Minimize
	value: tt

Subject to
tt + 1000 s_1_1 > 0.0
tt - t_1_1 - 1000.000000 s_1_1  + 0.000000 s_1_7  > -982.617213
tt + 1000 s_2_1 > 0.0
tt - t_2_1 - 1000.000000 s_2_1  + 0.000000 s_2_7  > -963.082301
tt + 1000 s_1_2 > 0.0
tt - t_1_2 - 1000.000000 s_1_2  + 0.000000 s_1_8  > -980.436164
tt + 1000 s_2_2 > 0.0
tt - t_2_2 - 1000.000000 s_2_2  + 0.000000 s_2_8  > -956.428720
tt + 1000 s_1_3 > 0.0
tt - t_1_3 - 1000.000000 s_1_3  + 0.000000 s_1_8  > -970.168617
tt + 1000 s_2_3 > 0.0
tt - t_2_3 - 1000.000000 s_2_3  + 0.000000 s_2_8  > -955.741440
tt + 1000 s_1_4 > 0.0
tt - t_1_4 - 1000.000000 s_1_4  + 0.000000 s_1_3  + 0.000000 s_1_8  > -983.972539
tt + 1000 s_2_4 > 0.0
tt - t_2_4 - 1000.000000 s_2_4  + 0.000000 s_2_3  + 0.000000 s_2_8  > -968.748268
tt + 1000 s_1_5 > 0.0
tt - t_1_5 - 1000.000000 s_1_5  + 0.000000 s_1_7  + 0.000000 s_1_8  > -985.605838
tt + 1000 s_2_5 > 0.0
tt - t_2_5 - 1000.000000 s_2_5  + 0.000000 s_2_7  + 0.000000 s_2_8  > -976.627260
tt + 1000 s_1_6 > 0.0
tt - t_1_6 - 1000.000000 s_1_6  + 0.000000 s_1_1  + 0.000000 s_1_4  + 0.000000 s_1_2  + 0.000000 s_1_3  + 0.000000 s_1_5  > -996.957700
tt + 1000 s_2_6 > 0.0
tt - t_2_6 - 1000.000000 s_2_6  + 0.000000 s_2_1  + 0.000000 s_2_4  + 0.000000 s_2_2  + 0.000000 s_2_3  + 0.000000 s_2_5  > -995.144300
tt + 1000 s_1_7 > 0.0
tt - t_1_7 - 1000.000000 s_1_7  > -962.426000
tt + 1000 s_2_7 > 0.0
tt - t_2_7 - 1000.000000 s_2_7  > -946.791687
tt + 1000 s_1_8 > 0.0
tt - t_1_8 - 1000.000000 s_1_8  > -962.349300
tt + 1000 s_2_8 > 0.0
tt - t_2_8 - 1000.000000 s_2_8  > -946.176400
s_1_1 + s_2_1 = 1
s_1_2 + s_2_2 = 1
s_1_3 + s_2_3 = 1
s_1_4 + s_2_4 = 1
s_1_5 + s_2_5 = 1
s_1_6 + s_2_6 = 1
s_1_7 + s_2_7 = 1
s_1_8 + s_2_8 = 1
s_1_1 + s_1_2 + 1000 r_1_1_2 >= 2
- s_1_1 - s_1_2 - 1000 r_1_1_2 >= -1001
t_1_1 - t_1_2 - 1000.000000 u_1_1_2 - 0.000000 s_1_2  + 0.000000 s_1_8  + 2000.000000 r_1_1_2 > -980.436164
s_2_1 + s_2_2 + 1000 r_2_1_2 >= 2
- s_2_1 - s_2_2 - 1000 r_2_1_2 >= -1001
t_2_1 - t_2_2 - 1000.000000 u_2_1_2 - 0.000000 s_2_2  + 0.000000 s_2_8  + 2000.000000 r_2_1_2 > -956.428720
s_1_1 + s_1_3 + 1000 r_1_1_3 >= 2
- s_1_1 - s_1_3 - 1000 r_1_1_3 >= -1001
t_1_1 - t_1_3 - 1000.000000 u_1_1_3 - 0.000000 s_1_3  + 0.000000 s_1_8  + 2000.000000 r_1_1_3 > -970.168617
s_2_1 + s_2_3 + 1000 r_2_1_3 >= 2
- s_2_1 - s_2_3 - 1000 r_2_1_3 >= -1001
t_2_1 - t_2_3 - 1000.000000 u_2_1_3 - 0.000000 s_2_3  + 0.000000 s_2_8  + 2000.000000 r_2_1_3 > -955.741440
s_1_1 + s_1_4 + 1000 r_1_1_4 >= 2
- s_1_1 - s_1_4 - 1000 r_1_1_4 >= -1001
t_1_1 - t_1_4 - 1000.000000 u_1_1_4 - 0.000000 s_1_4  + 0.000000 s_1_3  + 0.000000 s_1_8  + 2000.000000 r_1_1_4 > -983.972539
s_2_1 + s_2_4 + 1000 r_2_1_4 >= 2
- s_2_1 - s_2_4 - 1000 r_2_1_4 >= -1001
t_2_1 - t_2_4 - 1000.000000 u_2_1_4 - 0.000000 s_2_4  + 0.000000 s_2_3  + 0.000000 s_2_8  + 2000.000000 r_2_1_4 > -968.748268
s_1_1 + s_1_5 + 1000 r_1_1_5 >= 2
- s_1_1 - s_1_5 - 1000 r_1_1_5 >= -1001
t_1_1 - t_1_5 - 1000.000000 u_1_1_5 - 0.000000 s_1_5  + 0.000000 s_1_7  + 0.000000 s_1_8  + 2000.000000 r_1_1_5 > -985.605838
s_2_1 + s_2_5 + 1000 r_2_1_5 >= 2
- s_2_1 - s_2_5 - 1000 r_2_1_5 >= -1001
t_2_1 - t_2_5 - 1000.000000 u_2_1_5 - 0.000000 s_2_5  + 0.000000 s_2_7  + 0.000000 s_2_8  + 2000.000000 r_2_1_5 > -976.627260
t_1_6 - t_1_1 - 1000.000000 s_1_1  + 0.000000 s_1_7  > -982.617213
t_1_6 - t_2_1 - 1000.000000 s_2_1  + 0.000000 s_2_7  > -963.082301
t_2_6 - t_1_1 - 1000.000000 s_1_1  + 0.000000 s_1_7  > -982.617213
t_2_6 - t_2_1 - 1000.000000 s_2_1  + 0.000000 s_2_7  > -963.082301
t_1_1 - t_1_7 - 1000.000000 s_1_7  > -962.426000
t_1_1 - t_2_7 - 1000.000000 s_2_7  > -946.791687
t_2_1 - t_1_7 - 1000.000000 s_1_7  > -962.426000
t_2_1 - t_2_7 - 1000.000000 s_2_7  > -946.791687
s_1_1 + s_1_8 + 1000 r_1_1_8 >= 2
- s_1_1 - s_1_8 - 1000 r_1_1_8 >= -1001
t_1_1 - t_1_8 - 1000.000000 u_1_1_8 - 0.000000 s_1_8  + 2000.000000 r_1_1_8 > -962.349300
s_2_1 + s_2_8 + 1000 r_2_1_8 >= 2
- s_2_1 - s_2_8 - 1000 r_2_1_8 >= -1001
t_2_1 - t_2_8 - 1000.000000 u_2_1_8 - 0.000000 s_2_8  + 2000.000000 r_2_1_8 > -946.176400
s_1_2 + s_1_1 + 1000 r_1_1_2 >= 2
- s_1_2 - s_1_1 - 1000 r_1_1_2 >= -1001
t_1_2 - t_1_1 + 1000.000000 u_1_1_2 - 0.000000 s_1_1  + 0.000000 s_1_7  + 2000.000000 r_1_1_2 > 17.382787
s_2_2 + s_2_1 + 1000 r_2_1_2 >= 2
- s_2_2 - s_2_1 - 1000 r_2_1_2 >= -1001
t_2_2 - t_2_1 + 1000.000000 u_2_1_2 - 0.000000 s_2_1  + 0.000000 s_2_7  + 2000.000000 r_2_1_2 > 36.917699
s_1_2 + s_1_3 + 1000 r_1_2_3 >= 2
- s_1_2 - s_1_3 - 1000 r_1_2_3 >= -1001
t_1_2 - t_1_3 - 1000.000000 u_1_2_3 - 0.000000 s_1_3  + 0.000000 s_1_8  + 2000.000000 r_1_2_3 > -970.168617
s_2_2 + s_2_3 + 1000 r_2_2_3 >= 2
- s_2_2 - s_2_3 - 1000 r_2_2_3 >= -1001
t_2_2 - t_2_3 - 1000.000000 u_2_2_3 - 0.000000 s_2_3  + 0.000000 s_2_8  + 2000.000000 r_2_2_3 > -955.741440
s_1_2 + s_1_4 + 1000 r_1_2_4 >= 2
- s_1_2 - s_1_4 - 1000 r_1_2_4 >= -1001
t_1_2 - t_1_4 - 1000.000000 u_1_2_4 - 0.000000 s_1_4  + 0.000000 s_1_3  + 0.000000 s_1_8  + 2000.000000 r_1_2_4 > -983.972539
s_2_2 + s_2_4 + 1000 r_2_2_4 >= 2
- s_2_2 - s_2_4 - 1000 r_2_2_4 >= -1001
t_2_2 - t_2_4 - 1000.000000 u_2_2_4 - 0.000000 s_2_4  + 0.000000 s_2_3  + 0.000000 s_2_8  + 2000.000000 r_2_2_4 > -968.748268
s_1_2 + s_1_5 + 1000 r_1_2_5 >= 2
- s_1_2 - s_1_5 - 1000 r_1_2_5 >= -1001
t_1_2 - t_1_5 - 1000.000000 u_1_2_5 - 0.000000 s_1_5  + 0.000000 s_1_7  + 0.000000 s_1_8  + 2000.000000 r_1_2_5 > -985.605838
s_2_2 + s_2_5 + 1000 r_2_2_5 >= 2
- s_2_2 - s_2_5 - 1000 r_2_2_5 >= -1001
t_2_2 - t_2_5 - 1000.000000 u_2_2_5 - 0.000000 s_2_5  + 0.000000 s_2_7  + 0.000000 s_2_8  + 2000.000000 r_2_2_5 > -976.627260
t_1_6 - t_1_2 - 1000.000000 s_1_2  + 0.000000 s_1_8  > -980.436164
t_1_6 - t_2_2 - 1000.000000 s_2_2  + 0.000000 s_2_8  > -956.428720
t_2_6 - t_1_2 - 1000.000000 s_1_2  + 0.000000 s_1_8  > -980.436164
t_2_6 - t_2_2 - 1000.000000 s_2_2  + 0.000000 s_2_8  > -956.428720
s_1_2 + s_1_7 + 1000 r_1_2_7 >= 2
- s_1_2 - s_1_7 - 1000 r_1_2_7 >= -1001
t_1_2 - t_1_7 - 1000.000000 u_1_2_7 - 0.000000 s_1_7  + 2000.000000 r_1_2_7 > -962.426000
s_2_2 + s_2_7 + 1000 r_2_2_7 >= 2
- s_2_2 - s_2_7 - 1000 r_2_2_7 >= -1001
t_2_2 - t_2_7 - 1000.000000 u_2_2_7 - 0.000000 s_2_7  + 2000.000000 r_2_2_7 > -946.791687
t_1_2 - t_1_8 - 1000.000000 s_1_8  > -962.349300
t_1_2 - t_2_8 - 1000.000000 s_2_8  > -946.176400
t_2_2 - t_1_8 - 1000.000000 s_1_8  > -962.349300
t_2_2 - t_2_8 - 1000.000000 s_2_8  > -946.176400
s_1_3 + s_1_1 + 1000 r_1_1_3 >= 2
- s_1_3 - s_1_1 - 1000 r_1_1_3 >= -1001
t_1_3 - t_1_1 + 1000.000000 u_1_1_3 - 0.000000 s_1_1  + 0.000000 s_1_7  + 2000.000000 r_1_1_3 > 17.382787
s_2_3 + s_2_1 + 1000 r_2_1_3 >= 2
- s_2_3 - s_2_1 - 1000 r_2_1_3 >= -1001
t_2_3 - t_2_1 + 1000.000000 u_2_1_3 - 0.000000 s_2_1  + 0.000000 s_2_7  + 2000.000000 r_2_1_3 > 36.917699
s_1_3 + s_1_2 + 1000 r_1_2_3 >= 2
- s_1_3 - s_1_2 - 1000 r_1_2_3 >= -1001
t_1_3 - t_1_2 + 1000.000000 u_1_2_3 - 0.000000 s_1_2  + 0.000000 s_1_8  + 2000.000000 r_1_2_3 > 19.563836
s_2_3 + s_2_2 + 1000 r_2_2_3 >= 2
- s_2_3 - s_2_2 - 1000 r_2_2_3 >= -1001
t_2_3 - t_2_2 + 1000.000000 u_2_2_3 - 0.000000 s_2_2  + 0.000000 s_2_8  + 2000.000000 r_2_2_3 > 43.571280
t_1_4 - t_1_3 - 1000.000000 s_1_3  + 0.000000 s_1_8  > -970.168617
t_1_4 - t_2_3 - 1000.000000 s_2_3  + 0.000000 s_2_8  > -955.741440
t_2_4 - t_1_3 - 1000.000000 s_1_3  + 0.000000 s_1_8  > -970.168617
t_2_4 - t_2_3 - 1000.000000 s_2_3  + 0.000000 s_2_8  > -955.741440
s_1_3 + s_1_5 + 1000 r_1_3_5 >= 2
- s_1_3 - s_1_5 - 1000 r_1_3_5 >= -1001
t_1_3 - t_1_5 - 1000.000000 u_1_3_5 - 0.000000 s_1_5  + 0.000000 s_1_7  + 0.000000 s_1_8  + 2000.000000 r_1_3_5 > -985.605838
s_2_3 + s_2_5 + 1000 r_2_3_5 >= 2
- s_2_3 - s_2_5 - 1000 r_2_3_5 >= -1001
t_2_3 - t_2_5 - 1000.000000 u_2_3_5 - 0.000000 s_2_5  + 0.000000 s_2_7  + 0.000000 s_2_8  + 2000.000000 r_2_3_5 > -976.627260
t_1_6 - t_1_3 - 1000.000000 s_1_3  + 0.000000 s_1_8  > -970.168617
t_1_6 - t_2_3 - 1000.000000 s_2_3  + 0.000000 s_2_8  > -955.741440
t_2_6 - t_1_3 - 1000.000000 s_1_3  + 0.000000 s_1_8  > -970.168617
t_2_6 - t_2_3 - 1000.000000 s_2_3  + 0.000000 s_2_8  > -955.741440
s_1_3 + s_1_7 + 1000 r_1_3_7 >= 2
- s_1_3 - s_1_7 - 1000 r_1_3_7 >= -1001
t_1_3 - t_1_7 - 1000.000000 u_1_3_7 - 0.000000 s_1_7  + 2000.000000 r_1_3_7 > -962.426000
s_2_3 + s_2_7 + 1000 r_2_3_7 >= 2
- s_2_3 - s_2_7 - 1000 r_2_3_7 >= -1001
t_2_3 - t_2_7 - 1000.000000 u_2_3_7 - 0.000000 s_2_7  + 2000.000000 r_2_3_7 > -946.791687
t_1_3 - t_1_8 - 1000.000000 s_1_8  > -962.349300
t_1_3 - t_2_8 - 1000.000000 s_2_8  > -946.176400
t_2_3 - t_1_8 - 1000.000000 s_1_8  > -962.349300
t_2_3 - t_2_8 - 1000.000000 s_2_8  > -946.176400
s_1_4 + s_1_1 + 1000 r_1_1_4 >= 2
- s_1_4 - s_1_1 - 1000 r_1_1_4 >= -1001
t_1_4 - t_1_1 + 1000.000000 u_1_1_4 - 0.000000 s_1_1  + 0.000000 s_1_7  + 2000.000000 r_1_1_4 > 17.382787
s_2_4 + s_2_1 + 1000 r_2_1_4 >= 2
- s_2_4 - s_2_1 - 1000 r_2_1_4 >= -1001
t_2_4 - t_2_1 + 1000.000000 u_2_1_4 - 0.000000 s_2_1  + 0.000000 s_2_7  + 2000.000000 r_2_1_4 > 36.917699
s_1_4 + s_1_2 + 1000 r_1_2_4 >= 2
- s_1_4 - s_1_2 - 1000 r_1_2_4 >= -1001
t_1_4 - t_1_2 + 1000.000000 u_1_2_4 - 0.000000 s_1_2  + 0.000000 s_1_8  + 2000.000000 r_1_2_4 > 19.563836
s_2_4 + s_2_2 + 1000 r_2_2_4 >= 2
- s_2_4 - s_2_2 - 1000 r_2_2_4 >= -1001
t_2_4 - t_2_2 + 1000.000000 u_2_2_4 - 0.000000 s_2_2  + 0.000000 s_2_8  + 2000.000000 r_2_2_4 > 43.571280
s_1_4 + s_1_5 + 1000 r_1_4_5 >= 2
- s_1_4 - s_1_5 - 1000 r_1_4_5 >= -1001
t_1_4 - t_1_5 - 1000.000000 u_1_4_5 - 0.000000 s_1_5  + 0.000000 s_1_7  + 0.000000 s_1_8  + 2000.000000 r_1_4_5 > -985.605838
s_2_4 + s_2_5 + 1000 r_2_4_5 >= 2
- s_2_4 - s_2_5 - 1000 r_2_4_5 >= -1001
t_2_4 - t_2_5 - 1000.000000 u_2_4_5 - 0.000000 s_2_5  + 0.000000 s_2_7  + 0.000000 s_2_8  + 2000.000000 r_2_4_5 > -976.627260
t_1_6 - t_1_4 - 1000.000000 s_1_4  + 0.000000 s_1_3  + 0.000000 s_1_8  > -983.972539
t_1_6 - t_2_4 - 1000.000000 s_2_4  + 0.000000 s_2_3  + 0.000000 s_2_8  > -968.748268
t_2_6 - t_1_4 - 1000.000000 s_1_4  + 0.000000 s_1_3  + 0.000000 s_1_8  > -983.972539
t_2_6 - t_2_4 - 1000.000000 s_2_4  + 0.000000 s_2_3  + 0.000000 s_2_8  > -968.748268
s_1_4 + s_1_7 + 1000 r_1_4_7 >= 2
- s_1_4 - s_1_7 - 1000 r_1_4_7 >= -1001
t_1_4 - t_1_7 - 1000.000000 u_1_4_7 - 0.000000 s_1_7  + 2000.000000 r_1_4_7 > -962.426000
s_2_4 + s_2_7 + 1000 r_2_4_7 >= 2
- s_2_4 - s_2_7 - 1000 r_2_4_7 >= -1001
t_2_4 - t_2_7 - 1000.000000 u_2_4_7 - 0.000000 s_2_7  + 2000.000000 r_2_4_7 > -946.791687
t_1_4 - t_1_8 - 1000.000000 s_1_8  > -962.349300
t_1_4 - t_2_8 - 1000.000000 s_2_8  > -946.176400
t_2_4 - t_1_8 - 1000.000000 s_1_8  > -962.349300
t_2_4 - t_2_8 - 1000.000000 s_2_8  > -946.176400
s_1_5 + s_1_1 + 1000 r_1_1_5 >= 2
- s_1_5 - s_1_1 - 1000 r_1_1_5 >= -1001
t_1_5 - t_1_1 + 1000.000000 u_1_1_5 - 0.000000 s_1_1  + 0.000000 s_1_7  + 2000.000000 r_1_1_5 > 17.382787
s_2_5 + s_2_1 + 1000 r_2_1_5 >= 2
- s_2_5 - s_2_1 - 1000 r_2_1_5 >= -1001
t_2_5 - t_2_1 + 1000.000000 u_2_1_5 - 0.000000 s_2_1  + 0.000000 s_2_7  + 2000.000000 r_2_1_5 > 36.917699
s_1_5 + s_1_2 + 1000 r_1_2_5 >= 2
- s_1_5 - s_1_2 - 1000 r_1_2_5 >= -1001
t_1_5 - t_1_2 + 1000.000000 u_1_2_5 - 0.000000 s_1_2  + 0.000000 s_1_8  + 2000.000000 r_1_2_5 > 19.563836
s_2_5 + s_2_2 + 1000 r_2_2_5 >= 2
- s_2_5 - s_2_2 - 1000 r_2_2_5 >= -1001
t_2_5 - t_2_2 + 1000.000000 u_2_2_5 - 0.000000 s_2_2  + 0.000000 s_2_8  + 2000.000000 r_2_2_5 > 43.571280
s_1_5 + s_1_3 + 1000 r_1_3_5 >= 2
- s_1_5 - s_1_3 - 1000 r_1_3_5 >= -1001
t_1_5 - t_1_3 + 1000.000000 u_1_3_5 - 0.000000 s_1_3  + 0.000000 s_1_8  + 2000.000000 r_1_3_5 > 29.831383
s_2_5 + s_2_3 + 1000 r_2_3_5 >= 2
- s_2_5 - s_2_3 - 1000 r_2_3_5 >= -1001
t_2_5 - t_2_3 + 1000.000000 u_2_3_5 - 0.000000 s_2_3  + 0.000000 s_2_8  + 2000.000000 r_2_3_5 > 44.258560
s_1_5 + s_1_4 + 1000 r_1_4_5 >= 2
- s_1_5 - s_1_4 - 1000 r_1_4_5 >= -1001
t_1_5 - t_1_4 + 1000.000000 u_1_4_5 - 0.000000 s_1_4  + 0.000000 s_1_3  + 0.000000 s_1_8  + 2000.000000 r_1_4_5 > 16.027461
s_2_5 + s_2_4 + 1000 r_2_4_5 >= 2
- s_2_5 - s_2_4 - 1000 r_2_4_5 >= -1001
t_2_5 - t_2_4 + 1000.000000 u_2_4_5 - 0.000000 s_2_4  + 0.000000 s_2_3  + 0.000000 s_2_8  + 2000.000000 r_2_4_5 > 31.251732
t_1_6 - t_1_5 - 1000.000000 s_1_5  + 0.000000 s_1_7  + 0.000000 s_1_8  > -985.605838
t_1_6 - t_2_5 - 1000.000000 s_2_5  + 0.000000 s_2_7  + 0.000000 s_2_8  > -976.627260
t_2_6 - t_1_5 - 1000.000000 s_1_5  + 0.000000 s_1_7  + 0.000000 s_1_8  > -985.605838
t_2_6 - t_2_5 - 1000.000000 s_2_5  + 0.000000 s_2_7  + 0.000000 s_2_8  > -976.627260
t_1_5 - t_1_7 - 1000.000000 s_1_7  > -962.426000
t_1_5 - t_2_7 - 1000.000000 s_2_7  > -946.791687
t_2_5 - t_1_7 - 1000.000000 s_1_7  > -962.426000
t_2_5 - t_2_7 - 1000.000000 s_2_7  > -946.791687
t_1_5 - t_1_8 - 1000.000000 s_1_8  > -962.349300
t_1_5 - t_2_8 - 1000.000000 s_2_8  > -946.176400
t_2_5 - t_1_8 - 1000.000000 s_1_8  > -962.349300
t_2_5 - t_2_8 - 1000.000000 s_2_8  > -946.176400
s_1_7 + s_1_2 + 1000 r_1_2_7 >= 2
- s_1_7 - s_1_2 - 1000 r_1_2_7 >= -1001
t_1_7 - t_1_2 + 1000.000000 u_1_2_7 - 0.000000 s_1_2  + 0.000000 s_1_8  + 2000.000000 r_1_2_7 > 19.563836
s_2_7 + s_2_2 + 1000 r_2_2_7 >= 2
- s_2_7 - s_2_2 - 1000 r_2_2_7 >= -1001
t_2_7 - t_2_2 + 1000.000000 u_2_2_7 - 0.000000 s_2_2  + 0.000000 s_2_8  + 2000.000000 r_2_2_7 > 43.571280
s_1_7 + s_1_3 + 1000 r_1_3_7 >= 2
- s_1_7 - s_1_3 - 1000 r_1_3_7 >= -1001
t_1_7 - t_1_3 + 1000.000000 u_1_3_7 - 0.000000 s_1_3  + 0.000000 s_1_8  + 2000.000000 r_1_3_7 > 29.831383
s_2_7 + s_2_3 + 1000 r_2_3_7 >= 2
- s_2_7 - s_2_3 - 1000 r_2_3_7 >= -1001
t_2_7 - t_2_3 + 1000.000000 u_2_3_7 - 0.000000 s_2_3  + 0.000000 s_2_8  + 2000.000000 r_2_3_7 > 44.258560
s_1_7 + s_1_4 + 1000 r_1_4_7 >= 2
- s_1_7 - s_1_4 - 1000 r_1_4_7 >= -1001
t_1_7 - t_1_4 + 1000.000000 u_1_4_7 - 0.000000 s_1_4  + 0.000000 s_1_3  + 0.000000 s_1_8  + 2000.000000 r_1_4_7 > 16.027461
s_2_7 + s_2_4 + 1000 r_2_4_7 >= 2
- s_2_7 - s_2_4 - 1000 r_2_4_7 >= -1001
t_2_7 - t_2_4 + 1000.000000 u_2_4_7 - 0.000000 s_2_4  + 0.000000 s_2_3  + 0.000000 s_2_8  + 2000.000000 r_2_4_7 > 31.251732
s_1_7 + s_1_8 + 1000 r_1_7_8 >= 2
- s_1_7 - s_1_8 - 1000 r_1_7_8 >= -1001
t_1_7 - t_1_8 - 1000.000000 u_1_7_8 - 0.000000 s_1_8  + 2000.000000 r_1_7_8 > -962.349300
s_2_7 + s_2_8 + 1000 r_2_7_8 >= 2
- s_2_7 - s_2_8 - 1000 r_2_7_8 >= -1001
t_2_7 - t_2_8 - 1000.000000 u_2_7_8 - 0.000000 s_2_8  + 2000.000000 r_2_7_8 > -946.176400
s_1_8 + s_1_1 + 1000 r_1_1_8 >= 2
- s_1_8 - s_1_1 - 1000 r_1_1_8 >= -1001
t_1_8 - t_1_1 + 1000.000000 u_1_1_8 - 0.000000 s_1_1  + 0.000000 s_1_7  + 2000.000000 r_1_1_8 > 17.382787
s_2_8 + s_2_1 + 1000 r_2_1_8 >= 2
- s_2_8 - s_2_1 - 1000 r_2_1_8 >= -1001
t_2_8 - t_2_1 + 1000.000000 u_2_1_8 - 0.000000 s_2_1  + 0.000000 s_2_7  + 2000.000000 r_2_1_8 > 36.917699
s_1_8 + s_1_7 + 1000 r_1_7_8 >= 2
- s_1_8 - s_1_7 - 1000 r_1_7_8 >= -1001
t_1_8 - t_1_7 + 1000.000000 u_1_7_8 - 0.000000 s_1_7  + 2000.000000 r_1_7_8 > 37.574000
s_2_8 + s_2_7 + 1000 r_2_7_8 >= 2
- s_2_8 - s_2_7 - 1000 r_2_7_8 >= -1001
t_2_8 - t_2_7 + 1000.000000 u_2_7_8 - 0.000000 s_2_7  + 2000.000000 r_2_7_8 > 53.208313
Binary
s_1_1
s_2_1
s_1_2
s_2_2
s_1_3
s_2_3
s_1_4
s_2_4
s_1_5
s_2_5
s_1_6
s_2_6
s_1_7
s_2_7
s_1_8
s_2_8
u_1_1_2
r_1_1_2
u_2_1_2
r_2_1_2
u_1_1_3
r_1_1_3
u_2_1_3
r_2_1_3
u_1_1_4
r_1_1_4
u_2_1_4
r_2_1_4
u_1_1_5
r_1_1_5
u_2_1_5
r_2_1_5
u_1_1_8
r_1_1_8
u_2_1_8
r_2_1_8
u_1_2_3
r_1_2_3
u_2_2_3
r_2_2_3
u_1_2_4
r_1_2_4
u_2_2_4
r_2_2_4
u_1_2_5
r_1_2_5
u_2_2_5
r_2_2_5
u_1_2_7
r_1_2_7
u_2_2_7
r_2_2_7
u_1_3_5
r_1_3_5
u_2_3_5
r_2_3_5
u_1_3_7
r_1_3_7
u_2_3_7
r_2_3_7
u_1_4_5
r_1_4_5
u_2_4_5
r_2_4_5
u_1_4_7
r_1_4_7
u_2_4_7
r_2_4_7
u_1_7_8
r_1_7_8
u_2_7_8
r_2_7_8

End
