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

#include "CompressedSensingExample.h"
#include <ctime>
#include <cmath>
#include <string>

using namespace kl1p;



// ---------------------------------------------------------------------------------------------------- //

void	kl1p::CreateGaussianSignal(klab::UInt32 size, klab::UInt32 sparsity, klab::DoubleReal mean, klab::DoubleReal sigma, arma::Col<klab::DoubleReal>& out)
{
	out.set_size(size);
	out.fill(0.0);

	std::vector<klab::TArrayElement<klab::DoubleReal> > indices;
    for(klab::UInt32 i=0; i<size; ++i)
        indices.push_back(klab::TArrayElement<klab::DoubleReal>(i, klab::KRandom::Instance().generateDoubleReal(0.0, 1.0)));

    std::partial_sort(indices.begin(), indices.begin()+klab::Min(size, sparsity), indices.end(), std::greater<klab::TArrayElement<klab::DoubleReal> >());  

	for(klab::UInt32 i=0; i<sparsity; ++i)
	{
		klab::DoubleReal u1 = klab::KRandom::Instance().generateDoubleReal(0.0, 1.0);
		klab::DoubleReal u2 = klab::KRandom::Instance().generateDoubleReal(0.0, 1.0);

		klab::DoubleReal sign = klab::KRandom::Instance().generateBool() ? -1.0 : 1.0;
		out[indices[i].i()] = sign * ((klab::Sqrt(-2.0*klab::Log(u1)) * klab::Cos(2.0*klab::PI*u2))*sigma + mean);
	}
}

// ---------------------------------------------------------------------------------------------------- //

void	kl1p::WriteToCSVFile(const arma::Col<klab::DoubleReal>& signal, const std::string& filePath)
{
	std::ofstream of(filePath.c_str());
	if(of.is_open())
	{
		for(klab::UInt32 i=0; i<signal.n_rows; ++i)
		//	of<<i<<";"<<signal[i]<<std::endl;
			of<<signal[i]<<std::endl;
		of.close();
	}
	else
	{
		std::cout<<"ERROR! Unable to open file \""<<filePath<<"\" !"<<std::endl;
	}
}

int kl1p::ValidateReconstruction(const arma::Col<klab::DoubleReal>& original, const arma::Col<klab::DoubleReal>& compare, klab::DoubleReal tol)
{
    for(klab::UInt32 i=0; i<original.n_rows; ++i)
        if (std::abs(original[i] - compare[i]) > tol) return 0;
    
    return 1;
}

//void kl1p::WriteToLog(const std::string& type, klab::UInt32 N, klab::UInt32 M, klab::UInt32 k, klab::UInt32 sig, klab::UInt32 F, klab::DoubleReal result){


// ---------------------------------------------------------------------------------------------------- //

/*
 *  N
 *  RHO
 *  sigma
 * */
void	kl1p::RunExample(klab::UInt32 N, klab::DoubleReal RHO, klab::DoubleReal SIGMA)
{

   	klab::DoubleReal alpha = 0.0;			// Ratio of the cs-measurements.
    time_t now = time(0);
    std::stringstream t_ss;
    //t_ss << now;
    t_ss << N << "_" << RHO << "_" << SIGMA;
   
    std::ofstream of( (t_ss.str()+ ".csv").c_str());

   	std::cout<<"Start of KL1p compressed-sensing example."<<std::endl;
   	std::cout<<"Try to determine a sparse vector x "<<std::endl;
   	std::cout<<"from an underdetermined set of linear measurements y=A*x, "<<std::endl;
   	std::cout<<"where A is a random gaussian i.i.d sensing matrix."<<std::endl;
        
   	klab::UInt32 n = N;					// Size of the original signal x0.
   	klab::DoubleReal rho = RHO;				// Ratio of the sparsity of the signal x0.
   	klab::UInt32 k = klab::UInt32(rho*n);	// Sparsity of the signal x0 (number of non-zero elements).
   	klab::UInt64 seed = 0;					// Seed used for random number generation (0 if regenerate random numbers on each launch).
    int num_of_ex = 100; // number of interation
   	bool bWrite = false;					// Write signals to files ?
    int count_of_succeed = 0;
        
        		// Initialize random seed if needed.
   	if(seed > 0)
   		klab::KRandom::Instance().setSeed(seed);
        
        		// Display signal informations.
    std::cout<<"=============================="<<std::endl;
   	std::cout<<"N="<<n<<" (signal size)"<<std::endl;
   	//std::cout<<"M="<<m<<"="<<std::setprecision(5)<<(alpha*100.0)<<"% (number of measurements)"<<std::endl;
   	std::cout<<"K="<<k<<"="<<std::setprecision(5)<<(rho*100.0)<<"% (signal sparsity)"<<std::endl;
   	std::cout<<"Random Seed="<<klab::KRandom::Instance().seed()<<std::endl;
   	std::cout<<"=============================="<<std::endl;
        				
   	   
   	if(bWrite)
   		kl1p::WriteToCSVFile(x0, "OriginalSignal.csv");	// Write x0 to a file.

    try
     	{
        for (alpha = 0.1; alpha <= 0.8; alpha += 0.1){
#ifdef FCODE
         for(klab::UInt32 f=4; f<33; f=f*2){
            arma::Col<klab::DoubleReal> x0;					// Original signal x0 of size n.
   	        kl1p::CreateGaussianSignal(n, k, 0.0, 1.0, x0);	// Create randomly the original signal x0.
     
            klab::UInt32 m = klab::UInt32(alpha*n);	// Number of cs-measurements.
            if(f>m) continue;
            klab::TSmartPointer<kl1p::TOperator<klab::DoubleReal> > A = new kl1p::TFCODEGaussianMatrixOperator<klab::DoubleReal>(m, n, 0.0, 1.0, false, f, SIGMA);
#else
            klab::UInt32 m = klab::UInt32(alpha*n);	// Number of cs-measurements.
            klab::TSmartPointer<kl1p::TOperator<klab::DoubleReal> > A = new kl1p::TNormalRandomMatrixOperator<klab::DoubleReal>(m, n, 0.0, 1.0);
#endif
            count_of_succeed = 0;

            std::cout <<  "n= " << n << " m= " << m << std::endl;

            for(int i=0; i<num_of_ex; i++){
        		// Perform cs-measurements of size m.
                arma::Col<klab::DoubleReal> x0;					// Original signal x0 of size n.
   	            kl1p::CreateGaussianSignal(n, k, 0.0, 1.0, x0);	// Create randomly the original signal x0.
     
        		arma::Col<klab::DoubleReal> y;
        		A->apply(x0, y);
        		
        		klab::DoubleReal tolerance = 1e-3;	// Tolerance of the solution.
        		arma::Col<klab::DoubleReal> x;		// Will contain the solution of the reconstruction.

        
        		klab::KTimer timer;
        
        		// Compute CoSaMP.
    //    		std::cout<<"------------------------------"<<std::endl;
     //   		std::cout<<"[CoSaMP] Start."<<std::endl;
        		timer.start();
        		kl1p::TCoSaMPSolver<klab::DoubleReal> cosamp(tolerance);
        		cosamp.solve(y, A, k, x);
        		timer.stop();
      //  		std::cout<<"[CoSaMP] Done - SNR="<<std::setprecision(5)<<klab::SNR(x, x0)<<" - "
       // 			      <<"Time="<<klab::UInt32(timer.durationInMilliseconds())<<"ms"<<" - "
        //				  <<"Iterations="<<cosamp.iterations()<<std::endl;

                count_of_succeed += ValidateReconstruction(x0, x, tolerance);
        //		std::cout<<std::endl;
        //		std::cout<<"End of example."<<std::endl;
        	}
#ifdef FCODE
            of << "FCODE," <<  n << "," << m << "," << k << "," << "SIGMA" << "," << f << "," << count_of_succeed << std::endl;
#else
            of << "GAUSSIAN," <<  n << "," << m << "," << k << "," << "0.0," << "0" << "," << count_of_succeed << std::endl;
#endif

            of.flush();
#ifdef FCODE    
        }
#endif
        }
        }
        catch(klab::KException& e)
        {
        	std::cout<<"ERROR! KLab exception : "<<klab::FormatExceptionToString(e)<<std::endl;
        }
        catch(std::exception& e)
        {
        	std::cout<<"ERROR! Standard exception : "<<klab::FormatExceptionToString(e)<<std::endl;
        }
        catch(...)
        {
        	std::cout<<"ERROR! Unknown exception !"<<std::endl;
        }

    of.close();
    

}

// ---------------------------------------------------------------------------------------------------- //
