//
//  kernel.cu
//
//  Created by Arya Mazaheri on 01/12/2018.
//

#include <iostream>
#include <algorithm>
#include <cmath>
#include "ppm.h"

using namespace std;

/*********** Gray Scale Filter  *********/

/**
 * Converts a given 24bpp image into 8bpp grayscale using the GPU.
 */
__global__
void cuda_grayscale(int width, int height, BYTE *image, BYTE *image_out){
    /*for (int h = 0; h < height ; h++)
    {
        int offset_out = h * width;      // 1 color per pixel
        int offset  =  offset_out * 3; // 3 colors per pixel
        for (int w = 0; w < width; w++)
        {
            BYTE *pixel = &image[offset + w * 3];
            image_out[offset_out + w] = pixel[0] * 0.0722f + // B 
            pixel[1] * 0.7152f + // G
            pixel[2] * 0.2126f;  // R ;
        }
    }*/

    int threadsPerBlock = blockDim.x * blockDim.y;
    int threadIdInBlock = threadIdx.x + blockDim.x * threadIdx.y;

    int blocksInGrid = gridDim.x * gridDim.y;
    int blockIdInGrid = blockIdx.x + gridDim.x * blockIdx.y;
    int globalThreadId = blockIdInGrid * threadsPerBlock + threadIdInBlock;
    int totalNumThreads = blocksInGrid * threadsPerBlock;

    int i = 0;
    while(totalNumThreads * i  < width * height){ 
        if(totalNumThreads * i + globalThreadId <= width * height){
            int pixelindex = (globalThreadId * 3 + totalNumThreads * 3 * i);
            BYTE *pixel = &image[pixelindex];
            image_out[globalThreadId + totalNumThreads * i] = pixel[0] * 0.0722f + // B 
            pixel[1] * 0.7152f + // G
            pixel[2] * 0.2126f;  // R 
        }           
        i++;
    }
    //TODO (9 pt): implement grayscale filter kernel
}


// 1D Gaussian kernel array values of a fixed size (make sure the number > filter size d)
//TODO: Define the cGaussian array on the constant memory (2 pt)

void cuda_updateGaussian(int r, double sd)
{
	float fGaussian[64];
	for (int i = 0; i < 2*r +1 ; i++)
	{
		float x = i - r;
		fGaussian[i] = expf(-(x*x) / (2 * sd*sd));
	}
	//TODO: Copy computed fGaussian to the cGaussian on device memory (2 pts)
	//cudaMemcpyToSymbol(cGaussian, fGaussian, 64 * sizeof(float). cudaMemcpyDeviceToHost/* TODO */);
}

//TODO: implement cuda_gaussian() kernel (3 pts)


/*********** Bilateral Filter  *********/
// Parallel (GPU) Bilateral filter kernel
__global__ void cuda_bilateral_filter(BYTE* input, BYTE* output,
	int width, int height,
	int r, double sI, double sS)
{
	//TODO: implement bilateral filter kernel (9 pts)
}


void gpu_pipeline(const Image & input, Image & output, int r, double sI, double sS)
{
	// Events to calculate gpu run time
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// GPU related variables
    BYTE *d_input = NULL;
  	BYTE *d_image_out[2] = {0}; //temporary output buffers on gpu device
	int image_size = input.cols*input.rows;
	int suggested_blockSize;   // The launch configurator returned block size 
	int suggested_minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch

	// ******* Grayscale kernel launch *************

	//Creating the block size for grayscaling kernel
	cudaOccupancyMaxPotentialBlockSize( &suggested_minGridSize, &suggested_blockSize, cuda_grayscale);
        
        int block_dim_x, block_dim_y;
        block_dim_x = block_dim_y = (int) sqrt(suggested_blockSize); 

        dim3 gray_block(block_dim_x, block_dim_y/* TODO */); // 2 pts

        //TODO: Calculate grid size to cover the whole image - 2 pts
        int grid_dim_x, grid_dim_y;
        grid_dim_x = fmax(input.cols / block_dim_x, suggested_minGridSize);
        grid_dim_y = fmax(input.rows / block_dim_y, suggested_minGridSize);
        dim3 gray_dim(grid_dim_x, grid_dim_y);

        // Allocate the intermediate image buffers for each step
        Image img_out(input.cols, input.rows, 1, "P5");
        cout <<  "\ninit phase"<<endl;
        for (int i = 0; i < 2; i++)
        {  
            cudaError_t malloc_result = cudaMalloc((void**)&d_image_out[i], image_size);
            if(cudaSuccess != malloc_result)
               cout << "malloc " << i << " cuda error " << malloc_result << endl;
            else
                cout << "malloc " << i << " d_image_out successful "   << endl;
            cudaError_t memset_result = cudaMemset(d_image_out[i], 0xff, image_size);
            if(cudaSuccess != memset_result)
                cout << "memset " << i << " cuda error  " << memset_result  << endl;
            else
                cout << "memset " << i << " d_image_out successful "  << endl;
            //TODO: allocate memory on the device (2 pts)
            //TODO: intialize allocated memory on device to zero (2 pts)
        }
        //copy input image to device
        //TODO: Allocate memory on device for input image (2 pts)
        //TODO: Copy input image into the device memory (2 pts)
        cudaError_t mallocInput = cudaMalloc((void**) &d_input, image_size * 3);
        if(cudaSuccess != mallocInput)
            cout << "mallocInput cuda error  " << mallocInput  << endl;
        else
            cout << "malloc d_dinput successful "  << endl;

        BYTE *inputp = input.pixels;
        cudaError_t copyHostToDevice =  cudaMemcpy(d_input, inputp, image_size * 3, cudaMemcpyHostToDevice);
        if(cudaSuccess != copyHostToDevice)
            cout << "copyHostToDevice cuda error  " << cudaGetErrorString(copyHostToDevice)  << endl;
        else
            cout << "copy host to device successful "  << endl;

        cudaEventRecord(start, 0); // start timer
        // Convert input image to grayscale
        //TODO: Launch cuda_grayscale() (2 pts)
        cuda_grayscale<<<gray_dim, gray_block>>>(input.cols, input.rows, d_input, d_image_out[0]);
        cudaEventRecord(stop, 0); // stop timer
        cudaEventSynchronize(stop);

        // Calculate and print kernel run time
        cudaEventElapsedTime(&time, start, stop);
        cout << "GPU Grayscaling time: " << time << " (ms)\n";
        cout << "Launched blocks of size " << gray_block.x * gray_block.y << endl;
    
        //TODO: transfer image from device to the main memory for saving onto the disk (2 pts)
        cudaError_t copyDeviceToHost  = cudaMemcpy(img_out.pixels, d_image_out[0],  image_size, cudaMemcpyDeviceToHost);
        if(cudaSuccess != copyDeviceToHost)
            cout << "copyDeviceToHost cuda error  " << copyDeviceToHost  << endl;
        else   
            cout << "copy device to host successful "  << endl;

        savePPM(img_out, "image_gpu_gray.ppm");
        cudaFree(d_input);
        cudaFree(d_image_out);


	// ******* Bilateral filter kernel launch *************
	
	//Creating the block size for grayscaling kernel
	cudaOccupancyMaxPotentialBlockSize( &suggested_minGridSize, &suggested_blockSize, cuda_bilateral_filter); 
        
        block_dim_x = block_dim_y = (int) sqrt(suggested_blockSize); 

        dim3 bilateral_block(/* TODO */); // 2 pts

        //TODO: Calculate grid size to cover the whole image - 2pts

        // Create gaussain 1d array
        cuda_updateGaussian(r,sS);

        cudaEventRecord(start, 0); // start timer
	//TODO: Launch cuda_bilateral_filter() (2 pts)
        cudaEventRecord(stop, 0); // stop timer
        cudaEventSynchronize(stop);

        // Calculate and print kernel run time
        cudaEventElapsedTime(&time, start, stop);
        cout << "\nGPU Bilateral Filter time: " << time << " (ms)\n";
       // cout << "Launched blocks of size " << bilateral_block.x * bilateral_block.y << endl;

        // Copy output from device to host
	//TODO: transfer image from device to the main memory for saving onto the disk (2 pts)


        // ************** Finalization, cleaning up ************

        // Free GPU variables
	//TODO: Free device allocated memory (3 pts)
}
