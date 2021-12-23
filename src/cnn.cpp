#include "hls_vector.h"
#include "hls_stream.h"
#include "ap_int.h"
#include "assert.h"

#include "cnn.h"


static void load_input(hls::vector<short, BUSWIDTH> *inp, hls::stream<hls::vector<short, Tn>> & inp_stream) {
	
	hls::vector<short, Tn> tinp;
	hls::vector<short, BUSWIDTH> temp_inp;
	
	r_loop: for(int row = 0; row < R; row+=Tr) {
		c_loop: for(int col = 0; col < C; col+=Tc) {
			m_loop: for(int cho = 0; cho < M; cho+=Tm) {

				n_loop: for(int chi = 0; chi < N; chi+=Tn) {

					init_tinp_r: for (int tr = 0; tr < Tr+K-1; tr++) {
						int r = row + tr;
						init_tinp_c: for (int tc = 0; tc < Tc+K-1; tc++) {
#pragma HLS pipeline II = 1
							int c = col + tc;
							init_tinp_n: for (int tn = 0; tn < Tn; tn+=BUSWIDTH) {
#pragma HLS unroll
								int n = chi + tn*BUSWIDTH;
								temp_inp = inp[( r*N*(C+K-1) + c*N + n)/BUSWIDTH];
								for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
									tinp[tn+b] = temp_inp[b];
								}
							}
							inp_stream.write(tinp);
						}
					}		
				}
			}
		}
	}
}
static void load_weight(hls::vector<short, BUSWIDTH> *ker, hls::stream<hls::vector<short, Tn>> & ker_stream) {

	hls::vector<short, Tn> tker;
	hls::vector<short, BUSWIDTH> temp_ker;
	
	r_loop: for(int row = 0; row < R; row+=Tr) {
		c_loop: for(int col = 0; col < C; col+=Tc) {
			m_loop: for(int cho = 0; cho < M; cho+=Tm) {

				n_loop: for(int chi = 0; chi < N; chi+=Tn) {
					init_tker_ki: for (int ki = 0; ki < K; ki++) {
						init_tker_kj: for (int kj = 0; kj < K; kj++) {
							init_tker_m: for (int tm = 0; tm < Tm; tm++) {
#pragma HLS pipeline II = 1
								int m = cho + tm;
								init_tker_n: for (int tn = 0; tn < Tn; tn+=BUSWIDTH) {
#pragma HLS unroll
									int n = chi + tn*BUSWIDTH;
									temp_ker = ker[( ki*N*M*K + kj*N*M + m*N + n )/BUSWIDTH];
									for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
										tker[tn+b] = temp_ker[b];
									}
								}
								ker_stream.write(tker);
							}
						}
					}
				}
			}
		}
	}
}
/*
short Max(short a, short b, short c, short d)
{
	DTYPE arr[4] = {a, b, c, d};
	DTYPE max = 0;
	for (int ii = 0; ii < 4; ii++) {
		if(arr[ii] > max) max = arr[ii];
		else max = max;
	}	
	return max;
}

*/

static void Tiled_cnn(
		hls::stream<hls::vector<short, Tn>> & ker_stream,
		hls::stream<hls::vector<short, Tn>> & inp_stream,
		hls::stream<hls::vector<short, Tm>> & out_stream) {

	static short tinp[Tr+K-1][Tc+K-1][Tn];
	static short tker[K][K][Tm][Tn];
	static short tout[Tr][Tc][Tm];
	static short maxx[Tr/2][Tc/2][Tm];

	hls::vector<short, Tn> temp_inp;
	hls::vector<short, Tn> temp_ker;
	hls::vector<short, Tm> temp_out;

#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=tinp
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=tker
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=tout


	r_loop: for(int row = 0; row < R; row+=Tr) {
		c_loop: for(int col = 0; col < C; col+=Tc) {
			m_loop: for(int cho = 0; cho < M; cho+=Tm) {
				
				init_tout_r: for (int tr = 0; tr < Tr; tr++) {
					init_tout_c: for (int tc = 0; tc < Tc; tc++) {
#pragma HLS pipeline II = 1
						init_tout_m: for (int tm = 0; tm < Tm; tm++) {
#pragma HLS unroll
							tout[tr][tc][tm] = 0;
						}
					}
				}

				n_loop: for(int chi = 0; chi < N; chi+=Tn) {
					// Initialize tile of input
					init_tinp_r: for (int tr = 0; tr < Tr+K-1; tr++) {
						init_tinp_c: for (int tc = 0; tc < Tc+K-1; tc++) {
#pragma HLS pipeline II = 1
							temp_inp = inp_stream.read();
							init_tinp_n: for (int tn = 0; tn < Tn; tn++) {
#pragma HLS unroll
								tinp[tr][tc][tn] = temp_inp[tn];
					}}}
					// Initialize tile of kernel
					init_tker_ki: for (int ki = 0; ki < K; ki++) {
						init_tker_kj: for (int kj = 0; kj < K; kj++) {
							init_tker_m: for (int tm = 0; tm < Tm; tm++) {
#pragma HLS pipeline II = 1
								temp_ker = ker_stream.read();
								init_tker_n: for (int tn = 0; tn < Tn; tn++) {
#pragma HLS unroll
									tker[ki][kj][tm][tn]= temp_ker[tn];
					}}}}
					// Main computation
					ki: for (int ki = 0; ki < K; ki++) {
						kj: for (int kj = 0; kj < K; kj++) {
							tr: for (int tr = 0; tr < Tr; tr++) {
								tc: for (int tc = 0; tc < Tc; tc++) {
#pragma HLS pipeline II = 1
									tm: for (int tm = 0; tm < Tm; tm++) {
#pragma HLS unroll
										tn: for (int tn = 0; tn < Tn; tn++) {
#pragma HLS unroll
											L: tout[tr][tc][tm] += tker[ki][kj][tm][tn] * tinp[tr+ki][tc+kj][tn];

					}}}}}}
//start
					p_tm: for (int pm = 0; pm < Tm; pm++) {
						p_tr: for (int pr = 0; pr < Tr/2; pr++) {
							p_tc: for (int pc = 0; pc < Tc/2; pc++) {
								   int relu = Max(tout[2*pr][2*pc][pm], tout[2*pr][2*pc+1][pm], tout[2*pr+1][2*pc][pm], tout[2*pr+1][2*pc+1][pm]);							
								P: maxx[pr][pc][pm] = relu > 0 ? relu : 0;
							//	printf("%d\n",maxx[pr][pc][pm]);
				}}}
//end
				}

				// Writeback tile of output (Loop promotion)
				wb_tout_r: for (int tr = 0; tr < Tr/2; tr++) {
					wb_tout_c: for (int tc = 0; tc < Tc/2; tc++) {
#pragma HLS pipeline II = 1
						wb_tout_m: for (int tm = 0; tm < Tm; tm++) {
#pragma HLS unroll
							temp_out[tm] = maxx[tr][tc][tm];
						}
						out_stream.write(temp_out);
				}}

			
	}}}

}
static void store_result(short OUT1[41472], hls::stream<hls::vector<short, Tm>>&out_stream) {
	
	hls::vector<short, Tm> tout;
	hls::vector<short, BUSWIDTH> temp_out;
	
	int sss = 0;
	static short TEMP_OUT[R/2*C/2*M];

	r_loop: for(int row = 0; row < R/2; row+=Tr/2) {
		c_loop: for(int col = 0; col < C/2; col+=Tc/2) {
			m_loop: for(int cho = 0; cho < M; cho+=Tm) {
				
				wb_tout_r: for (int tr = 0; tr < Tr/2; tr++) {
					int r = row + tr;
					wb_tout_c: for (int tc = 0; tc < Tc/2; tc++) {
#pragma HLS pipeline II = 1
						int c = col + tc;
						tout = out_stream.read();
						wb_tout_m: for (int tm = 0; tm < Tm; tm+=BUSWIDTH) {
#pragma HLS unroll
							for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
								int m = cho + tm;
								temp_out[b] = tout[tm+b];
								TEMP_OUT[r*M*C/2 + c*M + m + b] = tout[tm+b];
							}
							//int m = cho + tm*BUSWIDTH;
							//out[( r*M*C/2 + c*M + m )/BUSWIDTH] = temp_out;

						}
					}
				}
			}
		}
	}
	for(int row = 0; row < (R/2 + (K-1)); row++){
		for(int col = 0; col < (C/2 + (K-1)); col++){
			for(int chi = 0; chi < M; chi++) {
				if(row > 0 && row < (R/2 + (K-1) -1) && col > 0 && col < (C/2 + (K-1) -1)) {
					OUT1[row*(C/2 + (K-1)) * M + col * M + chi] = TEMP_OUT[(row-1)*C/2*M + (col-1)*M +chi];
				}
				else OUT1[row*(C/2 + (K-1))*M + col*M + chi] = 0;
			}
		}
	}
}

static void store_result_to_host(hls::vector<short, BUSWIDTH>* out, short OUT1[41472]) {
	hls::vector<short, BUSWIDTH> tout;

		r_loop: for(int row = 0; row < (R/2+(K-1)); row+=Tr/2) {
		c_loop: for(int col = 0; col < (C/2+(K-1)); col+=Tc/2) {
			m_loop: for(int cho = 0; cho < M; cho+=Tm) {
				
				wb_tout_r: for (int tr = 0; tr < Tr/2; tr++) {
					int r = row + tr;
					wb_tout_c: for (int tc = 0; tc < Tc/2; tc++) {
#pragma HLS pipeline II = 1
						int c = col + tc;
						//tout = out_stream.read();
						wb_tout_m: for (int tm = 0; tm < Tm; tm+=BUSWIDTH) {
#pragma HLS unroll
							for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
								//temp_out[b] = tout[tm+b];
								tout[b] = OUT1[r * (C / 2 + (K - 1)) * M + c * M + tm + b];
							}
							int m = cho + tm*BUSWIDTH;
							out[( r*M*(C/2+(K-1)) + c*M + m )/BUSWIDTH] = tout;

						}
					}
				}
			}
		}
	}
}

extern "C" {

void cnn(
		hls::vector<short, BUSWIDTH>* inp,
		hls::vector<short, BUSWIDTH>* ker,
		hls::vector<short, BUSWIDTH>* out) {

#pragma HLS INTERFACE m_axi port = inp bundle = gmem0
#pragma HLS INTERFACE m_axi port = ker bundle = gmem1
#pragma HLS INTERFACE m_axi port = out bundle = gmem3

	static short OUT1[41472];

	static hls::stream<hls::vector<short, Tn> > inp_stream("input_stream");
	static hls::stream<hls::vector<short, Tn> > ker_stream("weight_stream");
	static hls::stream<hls::vector<short, Tm> > out_stream("output_stream");

#pragma HLS dataflow
	
	load_weight(ker, ker_stream);
	load_input(inp, inp_stream);
	Tiled_cnn(ker_stream, inp_stream, out_stream);
	store_result(OUT1, out_stream);
	store_result_to_host(out, OUT1);

}


}

