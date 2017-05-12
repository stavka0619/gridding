#ifndef __UTILCUDA_H
#define __UTILCUDA_H


typedef float mxType;
//typedef mxType mxArray;
typedef uint mxIndexType;
typedef size_t mxFlagType;
typedef unsigned int uint;

typedef unsigned char mxMaskType;


/** Segmented Scan data type */
#define DATATYPE_REAL 0
#define DATATYPE_COMPLEX_INTLV 1
#define DATATYPE_COMPLEX_SEP 2

/** Macros to write to vector */
#define get( _din, _iIdx, _real, _imag, _dataType ) \
    switch( _dataType ) { \
    case DATATYPE_REAL: \
        _real = _din[_iIdx]; \
        break; \
    case DATATYPE_COMPLEX_INTLV: \
        _real = _din[(_iIdx*2)+0]; \
        _imag = _din[(_iIdx*2)+1]; \
        break; \
    case DATATYPE_COMPLEX_SEP: \
        _real = _din[_iIdx]; \
        _imag = _din[_iIdx+(M_H*N_W)]; \
        break; \
    default: \
        break; \
    }

#define setVal( _dout, _oIdx, _real, _imag, _dataType ) \
    switch( _dataType ) { \
    case DATATYPE_REAL: \
        _dout[_oIdx] = _real; \
        break; \
    case DATATYPE_COMPLEX_INTLV: \
        _dout[(_oIdx*2)+0] = _real; \
        _dout[(_oIdx*2)+1] = _imag; \
        break; \
    case DATATYPE_COMPLEX_SEP: \
        _dout[_oIdx] = _real; \
        _dout[_oIdx+(M_H*N_W)] = _imag; \
        break; \
    default: \
        break; \
    }

#define set( _dout, _din, _oIdx, _iIdx, _dataType ) \
    switch( _dataType ) { \
    case DATATYPE_REAL: \
        _dout[_oIdx] = _din[_iIdx]; \
        break; \
    case DATATYPE_COMPLEX_INTLV: \
        _dout[(_oIdx*2)+0] = _din[(_iIdx*2)+0]; \
        _dout[(_oIdx*2)+1] = _din[(_iIdx*2)+1]; \
        break; \
    case DATATYPE_COMPLEX_SEP: \
        _dout[_oIdx] = _din[_iIdx]; \
        _dout[_oIdx+(M_H*N_W)] = _din[_iIdx+(M_H*N_W)]; \
        break; \
    default: \
        break; \
    }

#define CE(call) cutilSafeCall( call )

#ifdef __cplusplus
extern "C" {
#endif
    void GetCudaDevice();
    void initCuda();
    void myMalloc( void** ptr, size_t sz );
    void myFree( void* ptr );
    void myCopyH2D( void *dst, const void *src, size_t sz );
    void myCopyD2H( void *dst, const void *src, size_t sz );

#ifdef __cplusplus
}
#endif
    

#endif // __UTILCUDA_H
