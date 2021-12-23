#ifndef CNN_H_
#define CNN_H_

typedef short DTYPE;

const int IR = 224;
const int IC = 224;
const int IM = 64;
const int IN = 64;
const int IK = 3;

const int BUSWIDTH = 32;

// 2x2 Maxpooling
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
