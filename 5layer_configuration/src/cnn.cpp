#include "hls_vector.h"
#include "hls_stream.h"
#include "ap_int.h"
#include "assert.h"

#include "cnn.h"

// input load
static void load_input1(hls::vector<short, BUSWIDTH> *inp, hls::stream<hls::vector<short, BUSWIDTH>> & inp_stream) {
		
	const int R = 224;
	const int C = 224;
	const int M = 64;
	const int N = 64;
	const int K = 3;
	const int Tr = 56;
	const int Tc = 56;
	const int Tm = 32;
	const int Tn = 32;


	hls::vector<short, BUSWIDTH> tinp;
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

// Tiled_cnn1 kernel load
static void load_weight1(hls::vector<short, BUSWIDTH> *ker1, hls::stream<hls::vector<short, BUSWIDTH>> & ker_stream) {
		
	const int R = 224;
	const int C = 224;
	const int M = 64;
	const int N = 64;
	const int K = 3;
	const int Tr = 56;
	const int Tc = 56;
	const int Tm = 32;
	const int Tn = 32;


	hls::vector<short, BUSWIDTH> tker;
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
									temp_ker = ker1[( ki*N*M*K + kj*N*M + m*N + n )/BUSWIDTH];
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

// Tiled_cnn2 kernel load
static void load_weight2(hls::vector<short, BUSWIDTH> *ker2, hls::stream<hls::vector<short, BUSWIDTH>> & ker_stream) {
		
	const int R = 112;
	const int C = 112;
	const int M = 128;
	const int N = 64;
	const int K = 3;
	const int Tr = 56;
	const int Tc = 56;
	const int Tm = 32;
	const int Tn = 32;


	hls::vector<short, BUSWIDTH> tker;
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
									temp_ker = ker2[( ki*N*M*K + kj*N*M + m*N + n )/BUSWIDTH];
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

// Tiled_cnn3 kernel load
static void load_weight3(hls::vector<short, BUSWIDTH> *ker3, hls::stream<hls::vector<short, BUSWIDTH>> & ker_stream) {
		
	const int R = 112;
	const int C = 112;
	const int M = 128;
	const int N = 128;
	const int K = 3;
	const int Tr = 56;
	const int Tc = 56;
	const int Tm = 32;
	const int Tn = 32;


	hls::vector<short, BUSWIDTH> tker;
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
									temp_ker = ker3[( ki*N*M*K + kj*N*M + m*N + n )/BUSWIDTH];
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

// Tiled_cnn4 kernel load
static void load_weight4(hls::vector<short, BUSWIDTH> *ker4, hls::stream<hls::vector<short, BUSWIDTH>> & ker_stream) {
		
	const int R = 56;
	const int C = 56;
	const int M = 256;
	const int N = 128;
	const int K = 3;
	const int Tr = 56;
	const int Tc = 56;
	const int Tm = 32;
	const int Tn = 32;


	hls::vector<short, BUSWIDTH> tker;
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
									temp_ker = ker4[( ki*N*M*K + kj*N*M + m*N + n )/BUSWIDTH];
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

// Tiled_cnn5 kernel load
static void load_weight5(hls::vector<short, BUSWIDTH> *ker5, hls::stream<hls::vector<short, BUSWIDTH>> & ker_stream) {
		
	const int R = 56;
	const int C = 56;
	const int M = 256;
	const int N = 256;
	const int K = 3;
	const int Tr = 56;
	const int Tc = 56;
	const int Tm = 32;
	const int Tn = 32;


	hls::vector<short, BUSWIDTH> tker;
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
									temp_ker = ker5[( ki*N*M*K + kj*N*M + m*N + n )/BUSWIDTH];
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

// Tiled_cnn6 kernel load
static void load_weight6(hls::vector<short, BUSWIDTH> *ker6, hls::stream<hls::vector<short, BUSWIDTH>> & ker_stream) {
		
	const int R = 28;
	const int C = 28;
	const int M = 512;
	const int N = 256;
	const int K = 3;
	const int Tr = 56;
	const int Tc = 56;
	const int Tm = 32;
	const int Tn = 32;


	hls::vector<short, BUSWIDTH> tker;
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
									temp_ker = ker6[( ki*N*M*K + kj*N*M + m*N + n )/BUSWIDTH];
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

// Tiled_cnn7 kernel load
static void load_weight7(hls::vector<short, BUSWIDTH> *ker7, hls::stream<hls::vector<short, BUSWIDTH>> & ker_stream) {
		
	const int R = 28;
	const int C = 28;
	const int M = 512;
	const int N = 512;
	const int K = 3;
	const int Tr = 56;
	const int Tc = 56;
	const int Tm = 32;
	const int Tn = 32;


	hls::vector<short, BUSWIDTH> tker;
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
									temp_ker = ker7[( ki*N*M*K + kj*N*M + m*N + n )/BUSWIDTH];
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

// Tiled_cnn8 kernel load
static void load_weight8(hls::vector<short, BUSWIDTH> *ker8, hls::stream<hls::vector<short, BUSWIDTH>> & ker_stream) {
		
	const int R = 14;
	const int C = 14;
	const int M = 512;
	const int N = 512;
	const int K = 3;
	const int Tr = 56;
	const int Tc = 56;
	const int Tm = 32;
	const int Tn = 32;


	hls::vector<short, BUSWIDTH> tker;
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
									temp_ker = ker8[( ki*N*M*K + kj*N*M + m*N + n )/BUSWIDTH];
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

// Tiled_cnn1 with 2X2 Maxpooling
static void Tiled_cnn1(
		hls::stream<hls::vector<short, BUSWIDTH>> & ker_stream,
		hls::stream<hls::vector<short, BUSWIDTH>> & inp_stream,
		hls::stream<hls::vector<short, BUSWIDTH>> & out_stream
		) {
	
	const int R = 224;
	const int C = 224;
	const int M = 64;
	const int N = 64;
	const int K = 3;
	const int Tr = 56;
	const int Tc = 56;
	const int Tm = 32;
	const int Tn = 32;

	static short tinp[Tr+K-1][Tc+K-1][Tn];
	static short tker[K][K][Tm][Tn];
	static short tout[Tr][Tc][Tm];
	static short maxx[Tr/2][Tc/2][Tm];

	hls::vector<short, BUSWIDTH> temp_inp;
	hls::vector<short, BUSWIDTH> temp_ker;
	hls::vector<short, BUSWIDTH> temp_out;

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
//start 2X2 Maxpooling
					p_tm: for (int pm = 0; pm < Tm; pm++) {
						p_tr: for (int pr = 0; pr < Tr/2; pr++) {
							p_tc: for (int pc = 0; pc < Tc/2; pc++) {
								
								P: maxx[pr][pc][pm] = Max(tout[2*pr][2*pc][pm], tout[2*pr][2*pc+1][pm], tout[2*pr+1][2*pc][pm], tout[2*pr+1][2*pc+1][pm]);							
					}}}
//end
				}
				// Writeback tile of output (Loop promotion)
				wb_tout_r: for (int tr = 0; tr < Tr/2; tr++) {
					wb_tout_c: for (int tc = 0; tc < Tc/2; tc++) {
#pragma HLS pipeline II = 1
						wb_tout_m: for (int tm = 0; tm < Tm; tm++) {
#pragma HLS unroll
							if (maxx[tr][tc][tm] < 0) maxx[tr][tc][tm] = 0;
							temp_out[tm] = maxx[tr][tc][tm];
						}
						out_stream.write(temp_out);
				}}
			
	}}}

}

// Tiled_cnn2 without 2X2 Maxpooling
static void Tiled_cnn2(
		hls::stream<hls::vector<short, BUSWIDTH>> & ker_stream,
		short OUT1[831744],
		hls::stream<hls::vector<short, BUSWIDTH>> & out_stream
		) {
	
	const int R = 112;
	const int C = 112;
	const int M = 128;
	const int N = 64;
	const int K = 3;
	const int Tr = 56;
	const int Tc = 56;
	const int Tm = 32;
	const int Tn = 32;

	static short tinp[Tr][Tc][Tn];
	static short tker[K][K][Tm][Tn];
	static short tout[Tr-(K-1)][Tc-(K-1)][Tm];

	hls::vector<short, BUSWIDTH> temp_inp;
	hls::vector<short, BUSWIDTH> temp_ker;
	hls::vector<short, BUSWIDTH> temp_out;

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
					init_tinp_r: for (int tr = 0; tr < Tr; tr++) {
						init_tinp_c: for (int tc = 0; tc < Tc; tc++) {
#pragma HLS pipeline II = 1
							init_tinp_n: for (int tn = 0; tn < Tn; tn++) {
#pragma HLS unroll
							tinp[tr][tc][tn] = OUT1[(tr+row)*N*C + (tc+col)*N + (tn+chi)];

	
							}
					}}
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
				}
				// Writeback tile of output (Loop promotion)
				wb_tout_r: for (int tr = 0; tr < Tr; tr++) {
					wb_tout_c: for (int tc = 0; tc < Tc; tc++) {
#pragma HLS pipeline II = 1
						wb_tout_m: for (int tm = 0; tm < Tm; tm++) {
#pragma HLS unroll
							if (tout[tr][tc][tm] < 0) tout[tr][tc][tm] = 0;
							temp_out[tm] = tout[tr][tc][tm];
						}
						out_stream.write(temp_out);
				}}
			
	}}}

}

// Tiled_cnn3 with 2X2 Maxpooling
static void Tiled_cnn3(
		hls::stream<hls::vector<short, BUSWIDTH>> & ker_stream,
		short OUT2[1663488],
		hls::stream<hls::vector<short, BUSWIDTH>> & out_stream
		) {
	
	const int R = 112;
	const int C = 112;
	const int M = 128;
	const int N = 128;
	const int K = 3;
	const int Tr = 56;
	const int Tc = 56;
	const int Tm = 32;
	const int Tn = 32;

	static short tinp[Tr][Tc][Tn];
	static short tker[K][K][Tm][Tn];
	static short tout[Tr-(K-1)][Tc-(K-1)][Tm];
	static short maxx[(Tr-(K-1))/2][(Tc-(K-1))/2][Tm];

	hls::vector<short, BUSWIDTH> temp_inp;
	hls::vector<short, BUSWIDTH> temp_ker;
	hls::vector<short, BUSWIDTH> temp_out;

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
					init_tinp_r: for (int tr = 0; tr < Tr; tr++) {
						init_tinp_c: for (int tc = 0; tc < Tc; tc++) {
#pragma HLS pipeline II = 1
							init_tinp_n: for (int tn = 0; tn < Tn; tn++) {
#pragma HLS unroll
							tinp[tr][tc][tn] = OUT2[(tr+row)*N*C + (tc+col)*N + (tn+chi)];
							}
					}}
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
//start 2X2 Maxpooling
					p_tm: for (int pm = 0; pm < Tm; pm++) {
						p_tr: for (int pr = 0; pr < Tr/2; pr++) {
							p_tc: for (int pc = 0; pc < Tc/2; pc++) {
								
								P: maxx[pr][pc][pm] = Max(tout[2*pr][2*pc][pm], tout[2*pr][2*pc+1][pm], tout[2*pr+1][2*pc][pm], tout[2*pr+1][2*pc+1][pm]);							
					}}}
//end
				}
				// Writeback tile of output (Loop promotion)
				wb_tout_r: for (int tr = 0; tr < Tr/2; tr++) {
					wb_tout_c: for (int tc = 0; tc < Tc/2; tc++) {
#pragma HLS pipeline II = 1
						wb_tout_m: for (int tm = 0; tm < Tm; tm++) {
#pragma HLS unroll
							if (maxx[tr][tc][tm] < 0) maxx[tr][tc][tm] = 0;
							temp_out[tm] = maxx[tr][tc][tm];
						}
						out_stream.write(temp_out);
				}}
			
	}}}

}

// Tiled_cnn4 without 2X2 Maxpooling
static void Tiled_cnn4(
		hls::stream<hls::vector<short, BUSWIDTH>> & ker_stream,
		short OUT3[430592],
		hls::stream<hls::vector<short, BUSWIDTH>> & out_stream
		) {
	
	const int R = 56;
	const int C = 56;
	const int M = 256;
	const int N = 128;
	const int K = 3;
	const int Tr = 56;
	const int Tc = 56;
	const int Tm = 32;
	const int Tn = 32;

	static short tinp[Tr][Tc][Tn];
	static short tker[K][K][Tm][Tn];
	static short tout[Tr-(K-1)][Tc-(K-1)][Tm];

	hls::vector<short, BUSWIDTH> temp_inp;
	hls::vector<short, BUSWIDTH> temp_ker;
	hls::vector<short, BUSWIDTH> temp_out;

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
					init_tinp_r: for (int tr = 0; tr < Tr; tr++) {
						init_tinp_c: for (int tc = 0; tc < Tc; tc++) {
#pragma HLS pipeline II = 1
							init_tinp_n: for (int tn = 0; tn < Tn; tn++) {
#pragma HLS unroll
							tinp[tr][tc][tn] = OUT3[(tr+row)*N*C + (tc+col)*N + (tn+chi)];
							}
					}}
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
				}
				// Writeback tile of output (Loop promotion)
				wb_tout_r: for (int tr = 0; tr < Tr; tr++) {
					wb_tout_c: for (int tc = 0; tc < Tc; tc++) {
#pragma HLS pipeline II = 1
						wb_tout_m: for (int tm = 0; tm < Tm; tm++) {
#pragma HLS unroll
							if (tout[tr][tc][tm] < 0) tout[tr][tc][tm] = 0;
							temp_out[tm] = tout[tr][tc][tm];
						}
						out_stream.write(temp_out);
				}}
			
	}}}

}

// Tiled_cnn5 with 2X2 Maxpooling
static void Tiled_cnn5(
		hls::stream<hls::vector<short, BUSWIDTH>> & ker_stream,
		short OUT4[861184],
		hls::stream<hls::vector<short, BUSWIDTH>> & out_stream
		) {
	
	const int R = 56;
	const int C = 56;
	const int M = 256;
	const int N = 256;
	const int K = 3;
	const int Tr = 56;
	const int Tc = 56;
	const int Tm = 32;
	const int Tn = 32;

	static short tinp[Tr][Tc][Tn];
	static short tker[K][K][Tm][Tn];
	static short tout[Tr-(K-1)][Tc-(K-1)][Tm];
	static short maxx[(Tr-(K-1))/2][(Tc-(K-1))/2][Tm];

	hls::vector<short, BUSWIDTH> temp_inp;
	hls::vector<short, BUSWIDTH> temp_ker;
	hls::vector<short, BUSWIDTH> temp_out;

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
					init_tinp_r: for (int tr = 0; tr < Tr; tr++) {
						init_tinp_c: for (int tc = 0; tc < Tc; tc++) {
#pragma HLS pipeline II = 1
							init_tinp_n: for (int tn = 0; tn < Tn; tn++) {
#pragma HLS unroll
							tinp[tr][tc][tn] = OUT4[(tr+row)*N*C + (tc+col)*N + (tn+chi)];
							}
					}}
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
//start 2X2 Maxpooling
					p_tm: for (int pm = 0; pm < Tm; pm++) {
						p_tr: for (int pr = 0; pr < Tr/2; pr++) {
							p_tc: for (int pc = 0; pc < Tc/2; pc++) {
								
								P: maxx[pr][pc][pm] = Max(tout[2*pr][2*pc][pm], tout[2*pr][2*pc+1][pm], tout[2*pr+1][2*pc][pm], tout[2*pr+1][2*pc+1][pm]);							
					}}}
//end
				}
				// Writeback tile of output (Loop promotion)
				wb_tout_r: for (int tr = 0; tr < Tr/2; tr++) {
					wb_tout_c: for (int tc = 0; tc < Tc/2; tc++) {
#pragma HLS pipeline II = 1
						wb_tout_m: for (int tm = 0; tm < Tm; tm++) {
#pragma HLS unroll
							if (maxx[tr][tc][tm] < 0) maxx[tr][tc][tm] = 0;
							temp_out[tm] = maxx[tr][tc][tm];
						}
						out_stream.write(temp_out);
				}}
			
	}}}

}

// Tiled_cnn6 without 2X2 Maxpooling
static void Tiled_cnn6(
		hls::stream<hls::vector<short, BUSWIDTH>> & ker_stream,
		short OUT5[230400],
		hls::stream<hls::vector<short, BUSWIDTH>> & out_stream
		) {
	
	const int R = 28;
	const int C = 28;
	const int M = 512;
	const int N = 256;
	const int K = 3;
	const int Tr = 28;
	const int Tc = 28;
	const int Tm = 32;
	const int Tn = 32;

	static short tinp[Tr][Tc][Tn];
	static short tker[K][K][Tm][Tn];
	static short tout[Tr-(K-1)][Tc-(K-1)][Tm];

	hls::vector<short, BUSWIDTH> temp_inp;
	hls::vector<short, BUSWIDTH> temp_ker;
	hls::vector<short, BUSWIDTH> temp_out;

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
					init_tinp_r: for (int tr = 0; tr < Tr; tr++) {
						init_tinp_c: for (int tc = 0; tc < Tc; tc++) {
#pragma HLS pipeline II = 1
							init_tinp_n: for (int tn = 0; tn < Tn; tn++) {
#pragma HLS unroll
							tinp[tr][tc][tn] = OUT5[(tr+row)*N*C + (tc+col)*N + (tn+chi)];
							}
					}}
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
				}
				// Writeback tile of output (Loop promotion)
				wb_tout_r: for (int tr = 0; tr < Tr; tr++) {
					wb_tout_c: for (int tc = 0; tc < Tc; tc++) {
#pragma HLS pipeline II = 1
						wb_tout_m: for (int tm = 0; tm < Tm; tm++) {
#pragma HLS unroll
							if (tout[tr][tc][tm] < 0) tout[tr][tc][tm] = 0;
							temp_out[tm] = tout[tr][tc][tm];
						}
						out_stream.write(temp_out);
				}}
			
	}}}

}

// Tiled_cnn7 with 2X2 Maxpooling
static void Tiled_cnn7(
		hls::stream<hls::vector<short, BUSWIDTH>> & ker_stream,
		short OUT6[460800],
		hls::stream<hls::vector<short, BUSWIDTH>> & out_stream
		) {
	
	const int R = 28;
	const int C = 28;
	const int M = 512;
	const int N = 512;
	const int K = 3;
	const int Tr = 28;
	const int Tc = 28;
	const int Tm = 32;
	const int Tn = 32;

	static short tinp[Tr][Tc][Tn];
	static short tker[K][K][Tm][Tn];
	static short tout[Tr-(K-1)][Tc-(K-1)][Tm];
	static short maxx[(Tr-(K-1))/2][(Tc-(K-1))/2][Tm];

	hls::vector<short, BUSWIDTH> temp_inp;
	hls::vector<short, BUSWIDTH> temp_ker;
	hls::vector<short, BUSWIDTH> temp_out;

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
					init_tinp_r: for (int tr = 0; tr < Tr; tr++) {
						init_tinp_c: for (int tc = 0; tc < Tc; tc++) {
#pragma HLS pipeline II = 1
							init_tinp_n: for (int tn = 0; tn < Tn; tn++) {
#pragma HLS unroll
							tinp[tr][tc][tn] = OUT6[(tr+row)*N*C + (tc+col)*N + (tn+chi)];
							}
					}}
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
//start 2X2 Maxpooling
					p_tm: for (int pm = 0; pm < Tm; pm++) {
						p_tr: for (int pr = 0; pr < Tr/2; pr++) {
							p_tc: for (int pc = 0; pc < Tc/2; pc++) {
								
								P: maxx[pr][pc][pm] = Max(tout[2*pr][2*pc][pm], tout[2*pr][2*pc+1][pm], tout[2*pr+1][2*pc][pm], tout[2*pr+1][2*pc+1][pm]);							
					}}}
//end
				}
				// Writeback tile of output (Loop promotion)
				wb_tout_r: for (int tr = 0; tr < Tr/2; tr++) {
					wb_tout_c: for (int tc = 0; tc < Tc/2; tc++) {
#pragma HLS pipeline II = 1
						wb_tout_m: for (int tm = 0; tm < Tm; tm++) {
#pragma HLS unroll
							if (maxx[tr][tc][tm] < 0) maxx[tr][tc][tm] = 0;
							temp_out[tm] = maxx[tr][tc][tm];
						}
						out_stream.write(temp_out);
				}}
			
	}}}

}

// Tiled_cnn8 without 2X2 Maxpooling
static void Tiled_cnn8(
		hls::stream<hls::vector<short, BUSWIDTH>> & ker_stream,
		short OUT7[131072],
		hls::stream<hls::vector<short, BUSWIDTH>> & out_stream
		) {
	
	const int R = 14;
	const int C = 14;
	const int M = 512;
	const int N = 512;
	const int K = 3;
	const int Tr = 14;
	const int Tc = 14;
	const int Tm = 32;
	const int Tn = 32;

	static short tinp[Tr][Tc][Tn];
	static short tker[K][K][Tm][Tn];
	static short tout[Tr-(K-1)][Tc-(K-1)][Tm];
	static short maxx[(Tr-(K-1))/2][(Tc-(K-1))/2][Tm];

	hls::vector<short, BUSWIDTH> temp_inp;
	hls::vector<short, BUSWIDTH> temp_ker;
	hls::vector<short, BUSWIDTH> temp_out;

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
					init_tinp_r: for (int tr = 0; tr < Tr; tr++) {
						init_tinp_c: for (int tc = 0; tc < Tc; tc++) {
#pragma HLS pipeline II = 1
							init_tinp_n: for (int tn = 0; tn < Tn; tn++) {
#pragma HLS unroll
							tinp[tr][tc][tn] = OUT7[(tr+row)*N*C + (tc+col)*N + (tn+chi)];
							}
					}}
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
				}
				// Writeback tile of output (Loop promotion)
				wb_tout_r: for (int tr = 0; tr < Tr; tr++) {
					wb_tout_c: for (int tc = 0; tc < Tc; tc++) {
#pragma HLS pipeline II = 1
						wb_tout_m: for (int tm = 0; tm < Tm; tm++) {
#pragma HLS unroll
							if (tout[tr][tc][tm] < 0) tout[tr][tc][tm] = 0;
							temp_out[tm] = tout[tr][tc][tm];
						}
						out_stream.write(temp_out);
				}}
			
	}}}

}

// Store output of Tiled_cnn1 after zero padding
static void store_result1(short OUT1[831744], hls::stream<hls::vector<short, BUSWIDTH>> & out_stream) {
		

	const int R = 224;
	const int C = 224;
	const int M = 64;
	const int N = 64;
	const int K = 3;
	const int Tr = 56;
	const int Tc = 56;
	const int Tm = 32;
	const int Tn = 32;

	static short TEMP_OUT1[R/2*C/2*M];
	hls::vector<short, BUSWIDTH> tout;
	hls::vector<short, BUSWIDTH> temp_out;
	
	//arrange to array for zero padding
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
								TEMP_OUT1[r*M*C/2+c*M+m+b] = temp_out[b];
							}
					}
				}
			}
		}
	}}

	//zero padding
	for(int row = 0; row < (R/2 + (K-1)); row++) {
		for(int col = 0; col < (C/2 + (K-1)); col++) {
			for(int chi = 0; chi < M; chi++) {
				if(row > 0 && row < (R/2 + (K-1) - 1) && col > 0 && col < (C/2 + (K-1) - 1))
				{
					OUT1[row*(C/2 + (K-1))*M + col*M + chi] = TEMP_OUT1[(row-1)*C/2*M + (col-1)*M + chi];
				}
				else OUT1[row*(C/2 + (K-1))*M + col*M + chi] = 0;
			}
		}
	}
}

// Store output of Tiled_cnn2 after zero padding
static void store_result2(short OUT2[1663488], hls::stream<hls::vector<short, BUSWIDTH>> & out_stream) {
		

	const int R = 112;
	const int C = 112;
	const int M = 128;
	const int N = 64;
	const int K = 3;
	const int Tr = 56;
	const int Tc = 56;
	const int Tm = 32;
	const int Tn = 32;

	static short TEMP_OUT2[R*C*M];
	hls::vector<short, BUSWIDTH> tout;
	hls::vector<short, BUSWIDTH> temp_out;

	//arrange to array for zero padding
	r_loop: for(int row = 0; row < R; row+=Tr/2) {
		c_loop: for(int col = 0; col < C; col+=Tc/2) {
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
								TEMP_OUT2[r*M*C+c*M+m+b] = temp_out[b];
							}
					}
				}
			}
		}
	}}

	//zero padding
	for(int row = 0; row < (R + (K-1)); row++) {
		for(int col = 0; col < (C + (K-1)); col++) {
			for(int chi = 0; chi < M; chi++) {
				if(row > 0 && row < (R + (K-1) - 1) && col > 0 && col < (C + (K-1) - 1))
				{
					OUT2[row*(C + (K-1))*M + col*M + chi] = TEMP_OUT2[(row-1)*C*M + (col-1)*M + chi];
				}
				else OUT2[row*(C + (K-1))*M + col*M + chi] = 0;
			}
		}
	}
}

// Store output of Tiled_cnn3 after zero padding
static void store_result3(short OUT3[430592], hls::stream<hls::vector<short, BUSWIDTH>> & out_stream) {
		

	const int R = 112;
	const int C = 112;
	const int M = 128;
	const int N = 128;
	const int K = 3;
	const int Tr = 56;
	const int Tc = 56;
	const int Tm = 32;
	const int Tn = 32;

	static short TEMP_OUT3[R/2*C/2*M];
	hls::vector<short, BUSWIDTH> tout;
	hls::vector<short, BUSWIDTH> temp_out;
	
	//arrange to array for zero padding
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
								TEMP_OUT3[r*M*C/2+c*M+m+b] = temp_out[b];
							}
					}
				}
			}
		}
	}}

	//zero padding
	for(int row = 0; row < (R/2 + (K-1)); row++) {
		for(int col = 0; col < (C/2 + (K-1)); col++) {
			for(int chi = 0; chi < M; chi++) {
				if(row > 0 && row < (R/2 + (K-1) - 1) && col > 0 && col < (C/2 + (K-1) - 1))
				{
					OUT3[row*(C/2 + (K-1))*M + col*M + chi] = TEMP_OUT3[(row-1)*C/2*M + (col-1)*M + chi];
				}
				else OUT3[row*(C/2 + (K-1))*M + col*M + chi] = 0;
			}
		}
	}
}

// Store output of Tiled_cnn4 after zero padding
static void store_result4(short OUT4[861184], hls::stream<hls::vector<short, BUSWIDTH>> & out_stream) {
		

	const int R = 56;
	const int C = 56;
	const int M = 256;
	const int N = 128;
	const int K = 3;
	const int Tr = 56;
	const int Tc = 56;
	const int Tm = 32;
	const int Tn = 32;

	static short TEMP_OUT4[R*C*M];
	hls::vector<short, BUSWIDTH> tout;
	hls::vector<short, BUSWIDTH> temp_out;
	
	//arrange to array for zero padding
	r_loop: for(int row = 0; row < R; row+=Tr/2) {
		c_loop: for(int col = 0; col < C; col+=Tc/2) {
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
								TEMP_OUT4[r*M*C+c*M+m+b] = temp_out[b];
							}
					}
				}
			}
		}
	}}

	//zero padding
	for(int row = 0; row < (R + (K-1)); row++) {
		for(int col = 0; col < (C + (K-1)); col++) {
			for(int chi = 0; chi < M; chi++) {
				if(row > 0 && row < (R + (K-1) - 1) && col > 0 && col < (C + (K-1) - 1))
				{
					OUT4[row*(C + (K-1))*M + col*M + chi] = TEMP_OUT4[(row-1)*C*M + (col-1)*M + chi];
				}
				else OUT4[row*(C + (K-1))*M + col*M + chi] = 0;
			}
		}
	}
}

// Store output of Tiled_cnn5 after zero padding
static void store_result5(short OUT5[230400], hls::stream<hls::vector<short, BUSWIDTH>> & out_stream) {
		

	const int R = 56;
	const int C = 56;
	const int M = 256;
	const int N = 256;
	const int K = 3;
	const int Tr = 56;
	const int Tc = 56;
	const int Tm = 32;
	const int Tn = 32;

	static short TEMP_OUT5[R/2*C/2*M];
	hls::vector<short, BUSWIDTH> tout;
	hls::vector<short, BUSWIDTH> temp_out;
	
	//arrange to array for zero padding
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
								TEMP_OUT5[r*M*C/2+c*M+m+b] = temp_out[b];
							}
					}
				}
			}
		}
	}}

	//zero padding
	for(int row = 0; row < (R/2 + (K-1)); row++) {
		for(int col = 0; col < (C/2 + (K-1)); col++) {
			for(int chi = 0; chi < M; chi++) {
				if(row > 0 && row < (R/2 + (K-1) - 1) && col > 0 && col < (C/2 + (K-1) - 1))
				{
					OUT5[row*(C/2 + (K-1))*M + col*M + chi] = TEMP_OUT5[(row-1)*C/2*M + (col-1)*M + chi];
				}
				else OUT5[row*(C/2 + (K-1))*M + col*M + chi] = 0;
			}
		}
	}
}

// Store output of Tiled_cnn6 after zero padding
static void store_result6(short OUT6[460800], hls::stream<hls::vector<short, BUSWIDTH>> & out_stream) {
		

	const int R = 28;
	const int C = 28;
	const int M = 512;
	const int N = 256;
	const int K = 3;
	const int Tr = 28;
	const int Tc = 28;
	const int Tm = 32;
	const int Tn = 32;

	static short TEMP_OUT6[R*C*M];
	hls::vector<short, BUSWIDTH> tout;
	hls::vector<short, BUSWIDTH> temp_out;
	
	//arrange to array for zero padding
	r_loop: for(int row = 0; row < R; row+=Tr) {
		c_loop: for(int col = 0; col < C; col+=Tc) {
			m_loop: for(int cho = 0; cho < M; cho+=Tm) {
				
				wb_tout_r: for (int tr = 0; tr < Tr; tr++) {
					int r = row + tr;
					wb_tout_c: for (int tc = 0; tc < Tc; tc++) {
#pragma HLS pipeline II = 1
						int c = col + tc;
						tout = out_stream.read();
						wb_tout_m: for (int tm = 0; tm < Tm; tm+=BUSWIDTH) {
#pragma HLS unroll
							for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
								int m = cho + tm;
								temp_out[b] = tout[tm+b];
								TEMP_OUT6[r*M*C+c*M+m+b] = temp_out[b];
							}
					}
				}
			}
		}
	}}

	//zero padding
	for(int row = 0; row < (R + (K-1)); row++) {
		for(int col = 0; col < (C + (K-1)); col++) {
			for(int chi = 0; chi < M; chi++) {
				if(row > 0 && row < (R + (K-1) - 1) && col > 0 && col < (C + (K-1) - 1))
				{
					OUT6[row*(C + (K-1))*M + col*M + chi] = TEMP_OUT6[(row-1)*C*M + (col-1)*M + chi];
				}
				else OUT6[row*(C + (K-1))*M + col*M + chi] = 0;
			}
		}
	}
}

// Store output of Tiled_cnn7 after zero padding
static void store_result7(short OUT7[131072], hls::stream<hls::vector<short, BUSWIDTH>> & out_stream) {
		

	const int R = 28;
	const int C = 28;
	const int M = 512;
	const int N = 512;
	const int K = 3;
	const int Tr = 56;
	const int Tc = 56;
	const int Tm = 32;
	const int Tn = 32;

	static short TEMP_OUT7[R/2*C/2*M];
	hls::vector<short, BUSWIDTH> tout;
	hls::vector<short, BUSWIDTH> temp_out;
	
	//arrange to array for zero padding
	r_loop: for(int row = 0; row < R/2; row+=Tr/4) {
		c_loop: for(int col = 0; col < C/2; col+=Tc/4) {
			m_loop: for(int cho = 0; cho < M; cho+=Tm) {
				
				wb_tout_r: for (int tr = 0; tr < Tr/4; tr++) {
					int r = row + tr;
					wb_tout_c: for (int tc = 0; tc < Tc/4; tc++) {
#pragma HLS pipeline II = 1
						int c = col + tc;
						tout = out_stream.read();
						wb_tout_m: for (int tm = 0; tm < Tm; tm+=BUSWIDTH) {
#pragma HLS unroll
							for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
								int m = cho + tm;
								temp_out[b] = tout[tm+b];
								TEMP_OUT7[r*M*C/2+c*M+m+b] = temp_out[b];
							}
					}
				}
			}
		}
	}}

	//zero padding
	for(int row = 0; row < (R/2 + (K-1)); row++) {
		for(int col = 0; col < (C/2 + (K-1)); col++) {
			for(int chi = 0; chi < M; chi++) {
				if(row > 0 && row < (R/2 + (K-1) - 1) && col > 0 && col < (C/2 + (K-1) - 1))
				{
					OUT7[row*(C/2 + (K-1))*M + col*M + chi] = TEMP_OUT7[(row-1)*C/2*M + (col-1)*M + chi];
				}
				else OUT7[row*(C/2 + (K-1))*M + col*M + chi] = 0;
			}
		}
	}
}

// Store output of Tiled_cnn8
static void store_result(hls::vector<short, BUSWIDTH>* out, hls::stream<hls::vector<short, BUSWIDTH>> & out_stream) {
		

	const int R = 14;
	const int C = 14;
	const int M = 512;
	const int N = 512;
	const int K = 3;
	const int Tr = 56;
	const int Tc = 56;
	const int Tm = 32;
	const int Tn = 32;


	hls::vector<short, BUSWIDTH> tout;
	hls::vector<short, BUSWIDTH> temp_out;
	
	r_loop: for(int row = 0; row < R; row+=Tr/4) {
		c_loop: for(int col = 0; col < C; col+=Tc/4) {
			m_loop: for(int cho = 0; cho < M; cho+=Tm) {
				
				wb_tout_r: for (int tr = 0; tr < Tr/4; tr++) {
					int r = row + tr;
					wb_tout_c: for (int tc = 0; tc < Tc/4; tc++) {
#pragma HLS pipeline II = 1
						int c = col + tc;
						tout = out_stream.read();
						wb_tout_m: for (int tm = 0; tm < Tm; tm+=BUSWIDTH) {
#pragma HLS unroll
							for(int b = 0; b < BUSWIDTH; b++) {
#pragma HLS unroll
								temp_out[b] = tout[tm+b];
							}
							int m = cho + tm*BUSWIDTH;
							out[( r*M*C + c*M + m )/BUSWIDTH] = temp_out;

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
		hls::vector<short, BUSWIDTH>* ker1,
		hls::vector<short, BUSWIDTH>* ker2,
		hls::vector<short, BUSWIDTH>* ker3,
		hls::vector<short, BUSWIDTH>* ker4,
		hls::vector<short, BUSWIDTH>* ker5,
		hls::vector<short, BUSWIDTH>* ker6,
		hls::vector<short, BUSWIDTH>* ker7,
		hls::vector<short, BUSWIDTH>* ker8,
		hls::vector<short, BUSWIDTH>* out) {

#pragma HLS INTERFACE m_axi port = inp bundle = gmem0
#pragma HLS INTERFACE m_axi port = ker1 bundle = gmem1
#pragma HLS INTERFACE m_axi port = ker2 bundle = gmem2
#pragma HLS INTERFACE m_axi port = ker3 bundle = gmem1
#pragma HLS INTERFACE m_axi port = ker4 bundle = gmem2
#pragma HLS INTERFACE m_axi port = ker5 bundle = gmem1
#pragma HLS INTERFACE m_axi port = ker6 bundle = gmem2
#pragma HLS INTERFACE m_axi port = ker7 bundle = gmem1
#pragma HLS INTERFACE m_axi port = ker8 bundle = gmem2
#pragma HLS INTERFACE m_axi port = out bundle = gmem3

	static hls::stream<hls::vector<short, BUSWIDTH> > inp_stream("input_stream1");
	static hls::stream<hls::vector<short, BUSWIDTH> > ker_stream1("weight_stream1");
	static hls::stream<hls::vector<short, BUSWIDTH> > ker_stream2("weight_stream2");
	static hls::stream<hls::vector<short, BUSWIDTH> > ker_stream3("weight_stream3");
	static hls::stream<hls::vector<short, BUSWIDTH> > ker_stream4("weight_stream4");
	static hls::stream<hls::vector<short, BUSWIDTH> > ker_stream5("weight_stream5");
	static hls::stream<hls::vector<short, BUSWIDTH> > ker_stream6("weight_stream6");
	static hls::stream<hls::vector<short, BUSWIDTH> > ker_stream7("weight_stream7");
	static hls::stream<hls::vector<short, BUSWIDTH> > ker_stream8("weight_stream8");
	static hls::stream<hls::vector<short, BUSWIDTH> > out_stream1("output_stream1");
	static hls::stream<hls::vector<short, BUSWIDTH> > out_stream2("output_stream2");
	static hls::stream<hls::vector<short, BUSWIDTH> > out_stream3("output_stream3");
	static hls::stream<hls::vector<short, BUSWIDTH> > out_stream4("output_stream4");
	static hls::stream<hls::vector<short, BUSWIDTH> > out_stream5("output_stream5");
	static hls::stream<hls::vector<short, BUSWIDTH> > out_stream6("output_stream6");
	static hls::stream<hls::vector<short, BUSWIDTH> > out_stream7("output_stream7");
	static hls::stream<hls::vector<short, BUSWIDTH> > out_stream8("output_stream8");

	static short OUT1[831744];				//	114 x 114 x 64
	static short OUT2[1663488];				//	114 x 114 x 128
	static short OUT3[430592];				//	58 x 58 x 128
	static short OUT4[861184];				//	58 x 58 x 256
	static short OUT5[230400];				//	30 x 30 x 256
	static short OUT6[460800];				//	30 x 30 x 512
	static short OUT7[131072];				//	16 x 16 x 512

#pragma HLS dataflow

	load_input1(inp, inp_stream);

printf("1\n");
	load_weight1(ker1, ker_stream1);
	Tiled_cnn1(ker_stream1, inp_stream, out_stream1);
	store_result1(OUT1, out_stream1);

printf("2\n");
	load_weight2(ker2, ker_stream2);
	Tiled_cnn2(ker_stream2, OUT1, out_stream2);
	store_result2(OUT2, out_stream2);

printf("3\n");
	load_weight3(ker3, ker_stream3);
	Tiled_cnn3(ker_stream3, OUT2, out_stream3);
	store_result3(OUT3, out_stream3);

printf("4\n");
	load_weight4(ker4, ker_stream4);
	Tiled_cnn4(ker_stream4, OUT3, out_stream4);
	store_result4(OUT4, out_stream4);

printf("5\n");
	load_weight5(ker5, ker_stream5);
	Tiled_cnn5(ker_stream5, OUT4, out_stream5);
	store_result5(OUT5, out_stream5);

printf("6\n");
	load_weight6(ker6, ker_stream6);
	Tiled_cnn6(ker_stream6, OUT5, out_stream6);
	store_result6(OUT6, out_stream6);

printf("7\n");
	load_weight7(ker7, ker_stream7);
	Tiled_cnn7(ker_stream7, OUT6, out_stream7);
	store_result7(OUT7, out_stream7);

printf("8\n");
	load_weight8(ker8, ker_stream8);
	Tiled_cnn8(ker_stream8, OUT7, out_stream8);
	store_result(out, out_stream8);

}}
