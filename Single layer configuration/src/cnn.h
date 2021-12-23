#ifndef CNN_H_
#define CNN_H_

typedef short DTYPE;

const int R = 14;
const int C = 14;
const int M = 512;
const int N = 512;
const int K = 3;

const int BUSWIDTH = 32;

const int Tr = 14;
const int Tc = 14;
const int Tm = 32;
const int Tn = 32;



short Max(short a, short b, short c, short d)
{
	short arr[4] = {a, b, c, d};
	short max = a;
	for (int ii = 0; ii < 4; ii++) {
		if(arr[ii] > max) max = arr[ii];
		else max = max;
	}	
	return max;
}




#endif 
