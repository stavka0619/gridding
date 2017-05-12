#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cutil_inline.h>
#include <cuda.h>

#include "utilCuda.h"

__device__ float kSample( mxType* KBlut, size_t KBwidth,
                         float x, float y )
{
    // Linear interpolate
    int offset = ((KBwidth+1)*10)+0; // Kernel radius is 1 larger than width
    int pitch = (((KBwidth+1)*2)*10)+1; 
    int x0 = floorf(x*10.0) + offset;
    int x1 = ceilf(x*10.0) + offset;
    int y0 = floorf(y*10.0) + offset;
    int y1 = ceilf(y*10.0) + offset;
    float xd = x*10.0 - floorf(x*10.0);
    float yd = y*10.0 - floorf(y*10.0);

//     float nw = KBlut[ (x0*pitch + y0) ];
//     float ne = KBlut[ (x1*pitch + y0) ];
//     float sw = KBlut[ (x0*pitch + y1) ];
//     float se = KBlut[ (x1*pitch + y1) ];

    float nw = KBlut[ (y0*pitch + x0) ];
    float ne = KBlut[ (y0*pitch + x1) ];
    float sw = KBlut[ (y1*pitch + x0) ];
    float se = KBlut[ (y1*pitch + x1) ];

    float nw_ne = (nw * (1-xd)) + (ne * xd);
    float sw_se = (sw * (1-xd)) + (se * xd);
    float val = (nw_ne * (1-yd)) + (sw_se * yd);
    
    return val;
}

__global__ void reg2ir_cuda( mxType* xi, mxType* yi, 
                             mxType* G, size_t Gwidth, size_t Gheight,
                             mxType* KBlut, size_t KBwidth, 
                             mxType* GI, size_t NGI )
{
    volatile int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
    float fkbw = (float)KBwidth;
    float invGwidth = 1.0/(float)Gwidth;
    float invGheight = 1.0/(float)Gheight;

    if( tid < NGI ) {
        // Figure out sample pattern
        float xx = xi[tid]-1;
        float yy = yi[tid]-1;
        float nearx = rintf( xx );
        float neary = rintf( yy );
        float distxstart = xx - (nearx - fkbw);
        float distystart = yy - (neary - fkbw);
        float xstart = nearx - fkbw;
        float ystart = neary - fkbw;
        float accum = 0;

        for( int ww=0; ww < KBwidth*2+1; ww++ ) {
            float xnorm = (xstart+(float)ww)*invGwidth;
            int xcoord = (xnorm - floorf(xnorm)) * (float)Gwidth; 

            for( int hh=0; hh < KBwidth*2+1; hh++ ) {
                float ynorm = (ystart+(float)hh)*invGheight;
                int ycoord = (ynorm - floorf(ynorm)) * (float)Gheight;
                float Gsample = G[xcoord * Gheight + ycoord];
                float kernelSample = kSample( KBlut, KBwidth, 
                                              distxstart - ww,
                                              distystart - hh );
                accum += Gsample * kernelSample;
            }
        }
        GI[tid] = accum;
    }
}   

extern "C"
void gridding_reg2ir( mxType* xi, mxType* yi, 
                      mxType* G, size_t Gwidth, size_t Gheight,
                      mxType* KBlut, size_t KBwidth, 
                      mxType* GI, size_t NGI )
{
    uint threadsPerBlock = 512; 
    uint blocksPerGrid = (NGI + threadsPerBlock - 1) / threadsPerBlock;
    
    reg2ir_cuda
        <<<blocksPerGrid, threadsPerBlock>>>
        ( xi, yi, G, Gwidth, Gheight,
          KBlut, KBwidth,
          GI, NGI );
    
    cudaError_t e = cudaGetLastError();
    if( e != cudaSuccess ) {
        assert( e == cudaSuccess );
    }
}

typedef struct {
    float x;
    float y;
    float gi;
} ircache;
#define IRCACHE_SZ 1024
#define GMEMORYREAD_SZ 64 // 64bytes for half warp coleasced read
#define THREADS_PERBLOCK 256
#define TILE_DIM 16

/******************************************************************************/
__device__ float inblockregion( float x, float y,
                                float nwx, float nwy, float sex, float sey ) {
   
    if( (x > nwx && y > nwy &&
         x < sex && y < sey) ) {
        return true;
    } else {
        return false;
    }
}
__device__ int finblockregion( float x, float y,
                               float nwx, float nwy, float sex, float sey ) {
   
    if( (x > nwx && y > nwy &&
         x < sex && y < sey) ) {
        return 1;
    } else {
        return 0;
    }
}

__global__ void irreg2reg_cuda( mxType* xi, mxType* yi,
                                mxType* GI, size_t NGI,
                                mxType* KBlut, size_t KBwidth,  
                                mxType* G, size_t Gwidth, size_t Gheight )
{
    int id = threadIdx.x;
    mxType sample=0;

    __shared__ int nir;
    __shared__ ircache ir[IRCACHE_SZ];
    __shared__ ircache tmpir[ THREADS_PERBLOCK ];
    __shared__ short tmpir_inregion[ THREADS_PERBLOCK ];
    float blkx;
    float blky;

    // First thread initializes some values
    if( id == 0 ) {
        // Number of irregular points i the cache. Inits to zero
        nir = 0;
    }

    // Block coordinates within the grid
    // There are nBlocks in grid width
    float nBlocksHigh = Gheight/TILE_DIM; 
    blkx = floorf( blockIdx.x/nBlocksHigh );
    blky = blockIdx.x - (blkx * nBlocksHigh);

    // The block area we care about within the whole grid
    int nwy = blky * TILE_DIM; 
    int nwx = blkx * TILE_DIM;
    int sey = nwy + TILE_DIM-1;
    int sex = nwx + TILE_DIM-1;
        
    // Add filter window area
    nwy -= (KBwidth+1);
    nwx -= (KBwidth+1);
    sey += (KBwidth+1);
    sex += (KBwidth+1);

    // All threads read irregular points from global to 
    // optimize coalesced reads in "chunks". Then 
    // thread 0 is going to cache all irregular points within
    // region of interest in shared memory.
    for( int iii=0; iii < (int)(NGI/blockDim.x); iii++ ) {
        int idx = (iii*blockDim.x) + id;
        int ntmpir = 0;
        tmpir_inregion[id] = 0.0;

        if( idx < NGI ) {
            tmpir[ id ].x = xi[idx] - 1; // Matlab indices
            tmpir[ id ].y = yi[idx] - 1;
            tmpir[ id ].gi = GI[idx];

            // See if we care about this ir?
            tmpir_inregion[id] = finblockregion( tmpir[id].x, tmpir[id].y,
                                                 nwx, nwy, sex, sey );
        }

        __syncthreads();

        // Count the number of ir's in region
        for( uint s=blockDim.x/2; s > 0; s>>=1 ) {
            if( id < s ) {
                tmpir_inregion[id] += tmpir_inregion[id + s];
            }
            __syncthreads();
        }

        // Thread zero writes away total number of ir's we care about
        if( id == 0 ) {
            ntmpir = tmpir_inregion[id];
        }

        // Retag the irs we care about
        if( idx < NGI ) {
            tmpir_inregion[id] = finblockregion( tmpir[id].x, tmpir[id].y,
                                                 nwx, nwy, sex, sey );
        }            

        __syncthreads();

        // Thread 0 is going to cache all irregular points
        if( id == 0 ) {
            // Too many for cache
            if( ntmpir > IRCACHE_SZ - nir ) {
                // Crap, no more cache room.  Let's stop looking
                // We are now on slow path
                nir = -1;
                iii = NGI;
            } else {
                
                // If the irreg point is in the area we care about,
                // then cache it away
                for( int jj = 0; 
                     jj < blockDim.x && 
                         (jj+(iii*blockDim.x)) < NGI; jj++ ) {
                    if( tmpir_inregion[jj] == 1 ) {
                    
                        // Cache it away
                        ir[nir] = tmpir[jj];
                        nir++;                    
                    }
                }
            }
        }

        __syncthreads();
    } 

    // Coord within tile or block (stored col major)
    int posx = floorf(id/TILE_DIM);
    int posy = id - (posx * TILE_DIM);

    // Coord within whole grid
    int gridy = posy + (blky * TILE_DIM); 
    int gridx = posx + (blkx * TILE_DIM);

    // Add filter window area
    sey = gridy + (KBwidth+1);
    sex = gridx + (KBwidth+1);
    nwy = gridy - (KBwidth+1);
    nwx = gridx - (KBwidth+1);

    // If we were able to cache all the points we care about then have 
    // each thread compute its output
    if( nir >= 0 ) {
        for( int iii=0; iii < nir; iii++ ) {
            if( inblockregion( ir[iii].x, ir[iii].y,
                               (float)nwx, (float)nwy, 
                               (float)sex, (float)sey ) ) {
                sample += ir[iii].gi * kSample( KBlut, KBwidth, 
                                                ir[iii].x - gridx, 
                                                ir[iii].y - gridy );
            }
        }
    } else {
        // Oh crap, we couldn't cache everything so we need to take the
        // REALLY slow path and read each irreg point from global
        // memory
        for( int iii=0; iii < NGI; iii++ ) {
            float irx = xi[iii] - 1;
            float iry = yi[iii] - 1;
            if( inblockregion( irx, iry,
                               (float)nwx, (float)nwy, 
                               (float)sex, (float)sey ) ) {
                sample += GI[iii] * kSample( KBlut, KBwidth, 
                                             irx - gridx, iry - gridy );
            }
        } 
    }

    // Write out the output
    G[gridx * Gheight + gridy ] = sample;
}   

extern "C"
void gridding_irreg2reg( mxType* xi, mxType* yi, 
                         mxType* GI, size_t NGI,
                         mxType* KBlut, uint KBwidth, 
                         mxType* G, size_t Gwidth, size_t Gheight )
{
    // Want to divide work so each thread works on an output
    // coord.  Each block is a 16x16 tile.
    uint threadsPerBlock = THREADS_PERBLOCK; 
    uint blocksPerGrid = 
        ((Gwidth*Gheight) + threadsPerBlock - 1) / threadsPerBlock;

    irreg2reg_cuda
        <<<blocksPerGrid, threadsPerBlock>>>
        ( xi, yi, GI, NGI, KBlut, KBwidth, 
          G, Gwidth, Gheight );
    
    cudaError_t e = cudaGetLastError();
    if( e != cudaSuccess ) {
        fprintf( stderr, "Error running kernel <%d>\n", e );
        assert( e == cudaSuccess );
    }
}

/*****************************************************************************/
__global__ void irreg2reg_cuda_bin( mxType* xi, mxType* yi,
                                    mxType* GI, size_t NGI,
                                    mxType* KBlut, size_t KBwidth,  
                                    mxType* G, size_t Gwidth, size_t Gheight,
                                    uint* binidx, uint* binstartidx, 
                                    uint* binlength )
{
    int id = threadIdx.x;
    mxType sample=0;

    int nir;
    __shared__ ircache ir[IRCACHE_SZ];
    float blkx;
    float blky;

    // First thread initializes some values
    if( id == 0 ) {
        // Number of irregular points i the cache. Inits to zero
        nir = 0;
    }

    // Load irreg into cache
    uint numbin = binlength[blockIdx.x];
    uint startidx = binstartidx[blockIdx.x]-1;
    if( numbin <= IRCACHE_SZ ) {
        for( int iii=0; iii <= (numbin/blockDim.x); iii++ ) {
            int idx = (iii*blockDim.x) + id;
            if( idx < numbin ) {
                uint bidx = (binidx[startidx + idx]) - 1;
                ir[idx].x = xi[bidx]-1;
                ir[idx].y = yi[bidx]-1;
                ir[idx].gi = GI[bidx];
            }
        }
        nir = numbin;
    } else {
        // Can't load it into cache
        nir = -1;
    }

    __syncthreads();

    // Block coordinates within the grid
    // There are nBlocks in grid width
    float nBlocksHigh = Gheight/TILE_DIM; 
    blkx = floorf( blockIdx.x/nBlocksHigh );
    blky = blockIdx.x - (blkx * nBlocksHigh);

    // The block area we care about within the whole grid
    int nwy = blky * TILE_DIM; 
    int nwx = blkx * TILE_DIM;
    int sey = nwy + TILE_DIM-1;
    int sex = nwx + TILE_DIM-1;
        
    // Add filter window area
    nwy -= (KBwidth+1);
    nwx -= (KBwidth+1);
    sey += (KBwidth+1);
    sex += (KBwidth+1);

    // Coord within tile or block (stored col major)
    int posx = floorf(id/TILE_DIM);
    int posy = id - (posx * TILE_DIM);

    // Coord within whole grid
    int gridy = posy + (blky * TILE_DIM); 
    int gridx = posx + (blkx * TILE_DIM);

    // Add filter window area
    sey = gridy + (KBwidth+1);
    sex = gridx + (KBwidth+1);
    nwy = gridy - (KBwidth+1);
    nwx = gridx - (KBwidth+1);

    // If we were able to cache all the points we care about then have 
    // each thread compute its output
    if( nir >= 0 ) {
        for( int iii=0; iii < nir; iii++ ) {
            if( inblockregion( ir[iii].x, ir[iii].y,
                               (float)nwx, (float)nwy, 
                               (float)sex, (float)sey ) ) {
                sample += ir[iii].gi * kSample( KBlut, KBwidth, 
                                                ir[iii].x - gridx, 
                                                ir[iii].y - gridy ); 
            }
        }
    } else {
        // Oh crap, we couldn't cache everything so we need to take the
        // REALLY slow path and read each irreg point from global
        // memory
        for( int iii=0; iii < numbin; iii++ ) {
            uint bidx = (binidx[startidx + iii]) - 1;
            float irx = xi[bidx] - 1;
            float iry = yi[bidx] - 1;
            if( inblockregion( irx, iry,
                               (float)nwx, (float)nwy, 
                               (float)sex, (float)sey ) ) {
                sample += GI[bidx] * kSample( KBlut, KBwidth, 
                                              irx - gridx, iry - gridy );
            }
        } 
    }

    // Write out the output
    G[gridx * Gheight + gridy ] = sample;
}     

extern "C"
void gridding_irreg2reg_bin( mxType* xi, mxType* yi, 
                             mxType* GI, size_t NGI,
                             mxType* KBlut, uint KBwidth, 
                             mxType* G, size_t Gwidth, size_t Gheight,
                             uint* binidx, uint* binstartidx, 
                             uint* binlength, uint tiledim )
{
    // Want to divide work so each thread works on an output
    // coord.  Each block is a 16x16 tile.
    uint threadsPerBlock = THREADS_PERBLOCK; 
    uint blocksPerGrid = 
        ((Gwidth*Gheight) + threadsPerBlock - 1) / threadsPerBlock;

    assert( tiledim == TILE_DIM );
    
    irreg2reg_cuda_bin
        <<<blocksPerGrid, threadsPerBlock>>>
        ( xi, yi, GI, NGI, KBlut, KBwidth, 
          G, Gwidth, Gheight, binidx, binstartidx, binlength );
    
    cudaError_t e = cudaGetLastError();
    if( e != cudaSuccess ) {
        assert( e == cudaSuccess );
    }

}
