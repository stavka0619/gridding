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
#define data_IN_G prhs[2]
#define data_IN_KBlut prhs[3]
#define data_IN_nj prhs[4]

typedef unsigned int uint;

extern "C"
void gridding_reg2ir( float* xi, float* yi, 
                      float* G, size_t Gwidth, size_t Gheight,
                      float* KBlut, uint KBwidth, 
                      float* GI, size_t NGI );

err_t jktFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{ 
    float *xi, *yi, *G, *KBlut, *GI, *fKBwidth;
    uint hostKBwidth;

    TRY( jkt_mem((void **)&xi, data_IN_xi ) );
    TRY( jkt_mem((void **)&yi, data_IN_yi ) );
    TRY( jkt_mem((void **)&G, data_IN_G ) );
    TRY( jkt_mem((void **)&KBlut, data_IN_KBlut ) );

    TRY( jkt_mem_host((void **)&fKBwidth, data_IN_nj ) );
    hostKBwidth = (uint)(fKBwidth[0]);

    // Output size is length of input xi and yi
    const mwSize *dimsxi, *dimsG;
    int in_dimxi, in_dimG;
    mwSize nxi, nG;
    uint Gwidth, Gheight;

    in_dimxi = jkt_dims( data_IN_xi, &dimsxi );
    nxi = jkt_numel(data_IN_xi);

    in_dimG = jkt_dims( data_IN_G, &dimsG );
    Gheight = dimsG[0];
    Gwidth = dimsG[1];
    nG = jkt_numel(data_IN_G);

    if( 0 ) {
        FILE* fp = fopen( "/home/jpien/ndump.txt", "a" );
        if( fp ) {
            for( int iii=0; iii < in_dimxi; iii++ ) {
                fprintf( fp, "Size of xi: [%d]=%d\n", iii, dimsxi[iii] );
            }
            fprintf( fp, "Size of xi: %d\n", nxi );
            for( int iii=0; iii < in_dimG; iii++ ) {
                fprintf( fp, "Size of G: [%d]=%d\n", iii, dimsG[iii] );
            }
            fprintf( fp, "Size of G: %d\n", nG );
            fprintf( fp, "Width of kernel = %d\n", hostKBwidth );
            fflush( fp );
        }
        fclose( fp );
    }

    // Error checking
    if( jkt_complex( data_IN_xi ) ||
        jkt_complex( data_IN_yi ) || 
        jkt_complex( data_IN_G ) ||
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
    if( jkt_numel(data_IN_xi) != jkt_numel(data_IN_yi) ) {
        return err( "Size of xi and yi is not the same\n" );
    }        
    
    // Output is same number of elements as input irregular sample points
    mxArray *data_OUT_GI = plhs[0] = 
        jkt_new_array( in_dimxi, dimsxi, mxSINGLE_CLASS, 0 );
    TRY( jkt_mem((void **)&GI,  data_OUT_GI) );

    // Error checking
    if( false ) {
        return err( "Invalid input parameters\n" );
    }

    // Call CUDA
    gridding_reg2ir( xi, yi, G, Gwidth, Gheight, 
                     KBlut, hostKBwidth, GI, nxi );

    return errNone;
}
