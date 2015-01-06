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

#include "../CompressedSensingExample.h" 

#include <boost/asio/io_service.hpp>
#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>

#define NUMOFTHREAD 128

using namespace kl1p;
using namespace boost;

// ---------------------------------------------------------------------------------------------------- //

typedef struct params{
    int N;
    double sigma;
    double rho;
}params_t;


void runExample(params_t params){
    kl1p::RunExample(params.N, params.rho, params.sigma);
}

int main(int argc, char* argv[])
{

    boost::asio::io_service ioService;
    boost::thread_group threadpool;

    boost::asio::io_service::work work(ioService);

    for(int i = 0; i<NUMOFTHREAD; i++){
        threadpool.create_thread(
                    boost::bind(&boost::asio::io_service::run, &ioService)
                );
    }

    int Ns[] = {128, 256, 512};
    double Sigmas[] = {0.8, 0.9, 1.0, 1.1, 1.2};

    for(int i=0; i<3; i++){
        for(int j=0; j<5; j++){
            for(double rho = 0.05; rho < 0.5; rho+=0.05){
                params_t param = {Ns[i] , Sigmas[j], rho};
                ioService.post(boost::bind(runExample, param));
            }
        }
    }

    ioService.stop();
    threadpool.join_all();
    
	return 0;
}

// ---------------------------------------------------------------------------------------------------- //
 
