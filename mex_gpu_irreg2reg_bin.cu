#include <cuda.h>
#include "mex.h"
#include <math.h>
#include <assert.h>

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>
#include <cudpp.h>
#include <jacket.h>

// Input Arguments 
#define data_IN_xi prhs[0]
#define data_IN_yi prhs[1]
#define data_IN_GI prhs[2]
#define data_IN_KBlut prhs[3]
#define data_IN_nj prhs[4]
#define data_IN_siz prhs[5]
#define data_IN_binidx prhs[6]
#define data_IN_binstartidx prhs[7]
#define data_IN_binlength prhs[8]

typedef unsigned int uint;

extern "C"
void gridding_irreg2reg_bin( float* xi, float* yi, 
                             float* GI, size_t NGI,
                             float* KBlut, uint KBwidth, 
                             float* G, size_t Gwidth, size_t Gheight,
                             uint* binidx, uint* binstartidx, 
                             uint* binlength, uint tiledim );

err_t jktFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{ 
    float *xi, *yi, *G, *Gsiz, *KBlut, *GI, *fKBwidth;
    uint hostKBwidth, Gwidth, Gheight;
    uint *binidx, *binstartidx, *binlength;

    TRY( jkt_mem((void **)&xi, data_IN_xi ) );
    TRY( jkt_mem((void **)&yi, data_IN_yi ) );
    TRY( jkt_mem((void **)&GI, data_IN_GI ) );
    TRY( jkt_mem((void **)&KBlut, data_IN_KBlut ) );
    TRY( jkt_mem((void **)&binidx, data_IN_binidx ) );
    TRY( jkt_mem((void **)&binstartidx, data_IN_binstartidx ) );
    TRY( jkt_mem((void **)&binlength, data_IN_binlength ) );

    TRY( jkt_mem_host((void **)&fKBwidth, data_IN_nj ) );
    hostKBwidth = (uint)(fKBwidth[0]);

    TRY( jkt_mem_host((void **)&Gsiz, data_IN_siz ) );
    Gheight = (uint)(Gsiz[0]);
    Gwidth = (uint)(Gsiz[1]);

    mwSize NGI, len;

    NGI = jkt_numel(data_IN_GI);

    // Compute tile length
    len = jkt_numel(data_IN_binstartidx);
    len = Gwidth * Gheight / len;
    len = sqrt(len);
    
    // Output 
    mxArray *data_OUT_G = plhs[0] = 
        jkt_new( Gheight, Gwidth, mxSINGLE_CLASS, 0 );
    TRY( jkt_mem((void **)&G,  data_OUT_G) );

    // Error checking
    if( jkt_complex( data_IN_xi ) ||
        jkt_complex( data_IN_yi ) || 
        jkt_complex( data_IN_GI ) ||
        jkt_complex( data_IN_KBlut) ) {
        return err( "Input data must be real, non-complex\n" );
    }
    if( (Gheight & (Gheight-1) != 0) ||
        (Gwidth & (Gwidth-1) != 0) ) {
        return err( "G must have dimensions of power of two\n" );
    }
    if( jkt_numel(data_IN_KBlut) != 
        ((hostKBwidth+1)*20+1)*((hostKBwidth+1)*20+1) ) {
        return err( "KBlut size does not match KBwidth\n" );
    }
    if( jkt_numel(data_IN_xi) != jkt_numel(data_IN_yi) ||
        jkt_numel(data_IN_xi) != jkt_numel(data_IN_GI) ) {
        return err( "Size of xi, yi and/or GI is not the same\n" );
    }  
    if( jkt_numel(data_IN_binstartidx) != jkt_numel(data_IN_binlength) ) {
        return err( "Size of binstartidx and binlength is not the same\n" );
    }

    // Call CUDA
    gridding_irreg2reg_bin( xi, yi, 
                            GI, NGI,
                            KBlut, hostKBwidth, 
                            G, Gwidth, Gheight,
                            binidx, binstartidx, binlength, len );

    return errNone;
}
