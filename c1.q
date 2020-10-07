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
