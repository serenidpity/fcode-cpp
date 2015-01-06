// KL1p - A portable C++ compressed sensing library.
// Copyright (c) 2011-2012 René Gebel
// 
// This file is part of the KL1p C++ library.
// This library is distributed in the hope that it will be useful, 
// but WITHOUT ANY WARRANTY of fitness for any purpose. 
//
// This library is free software; You can redistribute it and/or modify it 
// under the terms of the GNU Lesser General Public License (LGPL) 
// as published by the Free Software Foundation, either version 3 of the License,
// or (at your option) any later version.
// See http://www.opensource.org/licenses for more info.

#ifndef KL1P_FCODENORMALRANDOMMATRIXOPERATOR_H
#define KL1P_FCODENORMALRANDOMMATRIXOPERATOR_H

#include "MatrixOperator.h"




namespace kl1p
{

// ---------------------------------------------------------------------------------------------------- //

template<class T>
class TFCODEGaussianMatrixOperator : public TMatrixOperator<T>
{
public:
    
    TFCODEGaussianMatrixOperator(klab::UInt32 m, klab::UInt32 n, const T& mean=klab::TTypeInfo<T>::ZERO, const T& deviation=klab::TTypeInfo<T>::UNIT, bool normalize=false, klab::UInt32 f_m = 4, klab::DoubleReal s = 1.0);
    TFCODEGaussianMatrixOperator(const TFCODEGaussianMatrixOperator<T>& op);
    virtual ~TFCODEGaussianMatrixOperator(); 

    virtual bool    isZero();
    virtual T       sum();
    virtual T       normFrobenius();
    virtual T       squaredNormFrobenius();
    virtual T       mean();
    virtual T       variance();


private:

    TFCODEGaussianMatrixOperator();
    TFCODEGaussianMatrixOperator<T>&     operator=(const TFCODEGaussianMatrixOperator<T>& op);


private:

    T               _mean;
    T               _deviation;
};

// ---------------------------------------------------------------------------------------------------- //
/*
template<class T>
class TFCODEGaussianMatrixOperatorSpecialisation
{
public:

    static T                GenerateNormalRandomNumber(klab::KRandom& random, const T& mean, const T& deviation);


private:

    TFCODEGaussianMatrixOperatorSpecialisation();
    TFCODEGaussianMatrixOperatorSpecialisation(const TFCODEGaussianMatrixOperatorSpecialisation<T>& spec);
    TFCODEGaussianMatrixOperatorSpecialisation<T>&   operator=(const TFCODEGaussianMatrixOperatorSpecialisation<T>& spec);
};

// ---------------------------------------------------------------------------------------------------- //

template<class T>
class TFCODEGaussianMatrixOperatorSpecialisation<std::complex<T> >
{
public:

    static std::complex<T>  GenerateNormalRandomNumber(klab::KRandom& random, const std::complex<T>& mean, const std::complex<T>& deviation);


private:

    TFCODEGaussianMatrixOperatorSpecialisation();
    TFCODEGaussianMatrixOperatorSpecialisation(const TFCODEGaussianMatrixOperatorSpecialisation<std::complex<T> >& spec);
    TFCODEGaussianMatrixOperatorSpecialisation<std::complex<T> >&    operator=(const TFCODEGaussianMatrixOperatorSpecialisation<std::complex<T> >& spec);
};

// ---------------------------------------------------------------------------------------------------- //
*/
template<class T>
inline TFCODEGaussianMatrixOperator<T>::TFCODEGaussianMatrixOperator(klab::UInt32 m, klab::UInt32 n, const T& mean, const T& deviation, bool normalize, klab::UInt32 f_m, klab::DoubleReal s) : TMatrixOperator<T>(),
_mean(mean), _deviation(deviation)
{
    arma::Mat<T>& mat = this->matrixReference();

    mat.set_size(m, n);
    for(klab::UInt32 i=0; i<m-f_m; ++i)
    {
        for(klab::UInt32 j=0; j<n; ++j)
            mat(i, j) = TNormalRandomMatrixOperatorSpecialisation<T>::GenerateNormalRandomNumber(klab::KRandom::Instance(), mean, deviation); 
    }

    for(int i = m-f_m; i<m; i++){
        for(klab::UInt32 j=0; j<n; ++j)
            mat(i,j) = 0.0;
    }

    //Hybrid Matrix Design
    double index[f_m+1];
    index[0] = 1.0;
    double sum = 0;
    int realIndex[f_m+1];

    for(int i = 1; i<f_m+1; i++){
        index[i] = s * index[i-1];
        sum += index[i];
    }

    double scale = (double)(n - f_m + 1) / sum;

    for(int i = 1; i<f_m+1; i++){
        index[i] = scale * index[i];
        sum += index[i];
    }

    realIndex[0] = 0;

    for(int i = 1; i<f_m+1; i++){
        realIndex[i] = (int)ceil(index[i]) + realIndex[i-1];
    }

    for(int j = 0; j<f_m; j++) {
        for(int i = realIndex[j]; i <realIndex[j+1]; i++){
            mat((m-f_m)+j,i) = 1.0;
        }
    }

    for( int i = realIndex[f_m]; i< n; i++){
        mat(m-1,i) = 1.0;
    }

    /*
    for(int i = 0; i< m; i++){
        for(int j = 0; j<n; j++){
            std::cout << mat(i,j)  << ",";
        }
        std::cout << " " << std::endl;
    }
    */


    this->resize(m, n);
	
	if(normalize)
		this->normalize();
}

// ---------------------------------------------------------------------------------------------------- //

template<class T>
inline TFCODEGaussianMatrixOperator<T>::TFCODEGaussianMatrixOperator(const TFCODEGaussianMatrixOperator<T>& op) : TMatrixOperator<T>(op)
{}

// ---------------------------------------------------------------------------------------------------- //

template<class T>
inline TFCODEGaussianMatrixOperator<T>::~TFCODEGaussianMatrixOperator()
{}

// ---------------------------------------------------------------------------------------------------- //

template<class T>
inline bool TFCODEGaussianMatrixOperator<T>::isZero()
{
    return (this->m()==0 || this->n()==0 || (klab::Equals(_mean, klab::TTypeInfo<T>::ZERO) && klab::Equals(_deviation, klab::TTypeInfo<T>::ZERO)));
}

// ---------------------------------------------------------------------------------------------------- //

template<class T>
inline T TFCODEGaussianMatrixOperator<T>::sum()
{
	if(this->isNormalized())
	{
		return TMatrixOperator<T>::sum();
	}
	else
	{
		T ret = klab::TTypeInfo<T>::ZERO;

		klab::UInt32 mn = this->m() * this->n();
		if(mn > 0)
			ret = _mean * static_cast<T>(mn);

		return ret;
	}
}

// ---------------------------------------------------------------------------------------------------- //

template<class T>
inline T TFCODEGaussianMatrixOperator<T>::normFrobenius()
{
    return klab::Sqrt(this->squaredNormFrobenius());
}

// ---------------------------------------------------------------------------------------------------- //

template<class T>
inline T TFCODEGaussianMatrixOperator<T>::squaredNormFrobenius()
{
	if(this->isNormalized())
	{
		return TMatrixOperator<T>::squaredNormFrobenius();
	}
	else
	{
		T ret = klab::TTypeInfo<T>::ZERO;

		klab::UInt32 mn = this->m() * this->n();
		if(mn > 0)
			ret = (this->variance() + (klab::Conj(_mean)*_mean)) * static_cast<T>(mn);

		return ret;
	}
}

// ---------------------------------------------------------------------------------------------------- //

template<class T>
inline T TFCODEGaussianMatrixOperator<T>::mean()
{
	if(this->isNormalized())	return TMatrixOperator<T>::mean();
	else						return _mean;
}

// ---------------------------------------------------------------------------------------------------- //

template<class T>
inline T TFCODEGaussianMatrixOperator<T>::variance()
{
	if(this->isNormalized())	return TMatrixOperator<T>::variance();
	else						return (klab::Conj(_deviation)*_deviation);
}

// ---------------------------------------------------------------------------------------------------- //
/*
template<class T>
inline T TFCODEGaussianMatrixOperatorSpecialisation<T>::GenerateNormalRandomNumber(klab::KRandom& random, const T& mean, const T& deviation)
{
    T u1 = random.generate<T>(klab::TTypeInfo<T>::ZERO, klab::TTypeInfo<T>::UNIT);
    T u2 = random.generate<T>(klab::TTypeInfo<T>::ZERO, klab::TTypeInfo<T>::UNIT);

    return (mean + (deviation * (klab::Sqrt(-2.0*klab::Log(u1))*klab::Cos(2.0*klab::PI*u2))));
}

// ---------------------------------------------------------------------------------------------------- //

template<class T>
inline std::complex<T> TFCODEGaussianMatrixOperatorSpecialisation<std::complex<T> >::GenerateNormalRandomNumber(klab::KRandom& random, const std::complex<T>& mean, const std::complex<T>& deviation)
{
    T real = TFCODEGaussianMatrixOperatorSpecialisation<T>::GenerateNormalRandomNumber(random, mean.real(), deviation.real());
    T imag = TFCODEGaussianMatrixOperatorSpecialisation<T>::GenerateNormalRandomNumber(random, mean.imag(), deviation.imag());

    return std::complex<T>(real, imag); 
}

// ---------------------------------------------------------------------------------------------------- //

*/
}

#endif
