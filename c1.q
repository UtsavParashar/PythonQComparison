Operators

// Division
85%2 /- 42.5, float, -9h
84%2 /- 42, float, -9h

// Modulus, remainder
84 mod 5 /- 4, long

// Quotient
floor 18%4 /- 4, long

// Exponentiation
6+6 xexp 2 /- 42, float

//- To get the type of a variable or value
type[2.0]       /- -9h
type["string"]  /- 10h
type@"2.0"      /- 10h
type 2          /- -7h

//- Questions
/- How many seconds are there in 10mins 20secs
20+10*60 /- 620j

/- if you ran 10 kms in 10mins 20sec, then how much you ran in miles/hr, 1mile = 1.62 km
(10%1.62)%10+20%60 /- 0.5973716

/- Volume of sphere formula 4/3*pi(r)3, if radius is 10
(4%3)*3.14159*10 xexp 3

/- Copy Px - 29.9, total copies - 80, discount - 30%, first copy shipping cost - 5, rest copies shipping cost - 0.8
(79*0.8)+5+29.9*80*.7

/- start run at 6:52, first and last lap = 8 min 15 secs, between 3 laps = 7 min 12 secs, Reached home at what time?
sth:6;stm:52;
rt:(2*15+8*60)+3*12+7*60;
rtm:floor rt%60;
rts:0.1*60;
mi:52+rtm;
if[60<mi;sth+:1;mi-:60];
raze (string(sth),":",(string(mi)),":",string(rts))





