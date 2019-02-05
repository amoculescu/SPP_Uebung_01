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
    int threadsPerBlock = blockDim.x * blockDim.y;
    int threadIdInBlock = threadIdx.x + blockDim.x * threadIdx.y;

    int blocksInGrid = gridDim.x * gridDim.y;
    int blockIdInGrid = blockIdx.x + gridDim.x * blockIdx.y;
    int globalThreadId = blockIdInGrid * threadsPerBlock + threadIdInBlock;
    int totalNumThreads = blocksInGrid * threadsPerBlock;

    int i = 0;
    while(totalNumThreads * i  < width * height){ 
        if(totalNumThreads * i + globalThreadId < width * height){
            int pixelIndex = (globalThreadId * 3 + totalNumThreads * 3 * i);
            BYTE *pixel = &image[pixelIndex];
            image_out[globalThreadId + totalNumThreads * i] = pixel[0] * 0.0722f + // B 
            pixel[1] * 0.7152f + // G
            pixel[2] * 0.2126f;  // R 
        }           
        i++;
    }
}


// 1D Gaussian kernel array values of a fixed size (make sure the number > filter size d)
__constant__ float cGaussian[64];
void cuda_updateGaussian(int r, double sd)
{
	float fGaussian[64];
	for (int i = 0; i < 2*r +1 ; i++)
	{
		float x = i - r;
		fGaussian[i] = expf(-(x*x) / (2 * sd*sd));
	}
    cudaError_t copyHostToDeviceSymbol = cudaMemcpyToSymbol(cGaussian, fGaussian, 64 * sizeof(float), 0, cudaMemcpyHostToDevice);
    if(cudaSuccess != copyHostToDeviceSymbol)
        cout << "copy to Symbol on device cuda error  " << copyHostToDeviceSymbol  << endl;
    else   
        cout << "copy to Symbol on device successful "  << endl;
}

//TODO: implement cuda_gaussian() kernel (3 pts)
__device__ double cuda_gaussian(float x, double sigma){
	return expf(-(powf(x, 2)) / (2 * powf(sigma, 2)));
}

/*********** Bilateral Filter  *********/
// Parallel (GPU) Bilateral filter kernel
__global__ void cuda_bilateral_filter(BYTE* input, BYTE* output,
	int width, int height,
	int r, double sI, double sS)
{
    int threadsPerBlock = blockDim.x * blockDim.y;
    int threadIdInBlock = threadIdx.x + blockDim.x * threadIdx.y;

    int blocksInGrid = gridDim.x * gridDim.y;
    int blockIdInGrid = blockIdx.x + gridDim.x * blockIdx.y;
    int globalThreadId = blockIdInGrid * threadsPerBlock + threadIdInBlock;
    int totalNumThreads = blocksInGrid * threadsPerBlock;

    int i = 0;
    int neighborXOffset;
    int neighborYOffset;
    double iFiltered = 0;
    double wP = 0;
    while(totalNumThreads * i  < width * height){
        iFiltered = 0;
        wP = 0;
        if(globalThreadId + totalNumThreads * i < width * height ){
            int pixelIndex = (globalThreadId + totalNumThreads * i);
            unsigned char centrePx = input[pixelIndex];
            for (int dy = -r; dy <= r; dy++){
                int pixelIndexCoordX = pixelIndex % width;
                int pixelIndexCoordY = pixelIndex / width;

                if(pixelIndexCoordY + dy < 0)
                    neighborYOffset = 0;
                else if(pixelIndexCoordY + dy > height - 1)
                    neighborYOffset = height - 1 - pixelIndexCoordY;
                else
                    neighborYOffset = dy;

                for(int dx = -r; dx <= r; dx++){
                    if( ((pixelIndexCoordX + dx) < 0) ||  (((pixelIndexCoordX + dx ) / width)   < (pixelIndexCoordX / width)) ) // neighbor is in row above
                        neighborXOffset = pixelIndexCoordX * (-1);
                    else if( ((pixelIndexCoordX + dx) / width) > (pixelIndexCoordX / width )  ) // neighbor in row bellow
                        neighborXOffset = width - 1 - pixelIndexCoordX;
                    else 
                        neighborXOffset = dx;

                    int neighborIndex = pixelIndex + neighborXOffset  + neighborYOffset * width;
                    unsigned char currPx = input[neighborIndex];

                    double w = (cGaussian[dy + r] * cGaussian[dx + r]) * cuda_gaussian(centrePx - currPx, sI);
                    iFiltered += w * currPx;
                    wP += w;
                }
            }
            output[pixelIndex] = iFiltered / wP;
        }
        i++;
    }
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

        dim3 gray_block(block_dim_x, block_dim_y); // 2 pts

        int grid_dim_x, grid_dim_y;
        grid_dim_x = fmax(input.cols / block_dim_x, suggested_minGridSize);
        grid_dim_y = fmax(input.rows / block_dim_y, suggested_minGridSize);
        dim3 gray_grid(grid_dim_x, grid_dim_y);

        // Allocate the intermediate image buffers for each step
        Image img_out(input.cols, input.rows, 1, "P5");
        cout <<  "\ninit phase"<<endl;
        for (int i = 0; i < 2; i++)
        {  
            cudaError_t malloc_result = cudaMalloc((void**)&d_image_out[i], image_size);
            if(cudaSuccess != malloc_result)
               cout << "malloc " << i << " cuda error " << malloc_result << endl;
            else
                cout << "malloc d_image_out[" << i << "] successful "   << endl;
            cudaError_t memset_result = cudaMemset(d_image_out[i], 0xff, image_size);
            if(cudaSuccess != memset_result)
                cout << "memset " << i << " cuda error  " << memset_result  << endl;
            else
                cout << "memset d_image_out[" << i << "] successful "  << endl;
         }
        //copy input image to device
        cudaError_t mallocInput = cudaMalloc((void**) &d_input, image_size * 3);
        if(cudaSuccess != mallocInput)
            cout << "malloc Input cuda error  " << mallocInput  << endl;
        else
            cout << "malloc d_dinput successful "  << endl;

        BYTE *inputp = input.pixels;
        cudaError_t copyHostToDevice =  cudaMemcpy(d_input, inputp, image_size * 3, cudaMemcpyHostToDevice);
        if(cudaSuccess != copyHostToDevice)
            cout << "copyHostToDevice cuda error  " << cudaGetErrorString(copyHostToDevice)  << endl;
        else
            cout << "copy input to device successful "  << endl;

        cudaEventRecord(start, 0); // start timer
        // Convert input image to grayscale
        cuda_grayscale<<<gray_grid, gray_block>>>(input.cols, input.rows, d_input, d_image_out[0]);
        cudaEventRecord(stop, 0); // stop timer
        cudaEventSynchronize(stop);

        // Calculate and print kernel run time
        cudaEventElapsedTime(&time, start, stop);
        cout << "GPU Grayscaling time: " << time << " (ms)\n";
        cout << "Launched blocks of size " << gray_block.x * gray_block.y << endl;
    
        cudaError_t copyDeviceToHost  = cudaMemcpy(img_out.pixels, d_image_out[0],  image_size, cudaMemcpyDeviceToHost);
          if(cudaSuccess != copyDeviceToHost)
            cout << "copyDeviceToHost cuda error  " << copyDeviceToHost  << endl;
        else   
            cout << "copy device to host successful "  << endl;

        savePPM(img_out, "image_gpu_gray.ppm");
        cudaFree(d_input);

	// ******* Bilateral filter kernel launch *************
	
	//Creating the block size for grayscaling kernel
	cudaOccupancyMaxPotentialBlockSize( &suggested_minGridSize, &suggested_blockSize, cuda_bilateral_filter); 
        
        block_dim_x = block_dim_y = (int) sqrt(suggested_blockSize); 

        dim3 bilateral_block(block_dim_x, block_dim_y); // 2 pts

        //TODO: Calculate grid size to cover the whole image - 2pts
        grid_dim_x = fmax(input.cols / block_dim_x, suggested_minGridSize);
        grid_dim_y = fmax(input.rows / block_dim_y, suggested_minGridSize);
        dim3 bilateral_grid(grid_dim_x, grid_dim_y);
        // Create gaussian 1d array
        cuda_updateGaussian(r,sS);

        cudaEventRecord(start, 0); // start timer
    //TODO: Launch cuda_bilateral_filter() (2 pts)
        cuda_bilateral_filter<<<bilateral_grid, bilateral_block>>>
        (d_image_out[0], d_image_out[1], input.cols, input.rows, r, sI, sS);
        cudaEventRecord(stop, 0); // stop timer
        cudaEventSynchronize(stop);

        // Calculate and print kernel run time
        cudaEventElapsedTime(&time, start, stop);
        cout << "\nGPU Bilateral Filter time: " << time << " (ms)\n";
        cout << "Launched blocks of size " << bilateral_block.x * bilateral_block.y << endl;

        // Copy output from device to host

	//TODO: transfer image from device to the main memory for saving onto the disk (2 pts)

    cudaError_t copyDeviceToHostBilateral  = cudaMemcpy(output.pixels, d_image_out[1],  image_size, cudaMemcpyDeviceToHost);
    if(cudaSuccess != copyDeviceToHostBilateral)
        cout << "copyDeviceToHostBilateral cuda error  " << copyDeviceToHostBilateral  << endl;
    else   
        cout << "copy bilateral from device to host successful "  << endl;

        // ************** Finalization, cleaning up ************

        // Free GPU variables
	//TODO: Free device allocated memory (3 pts)
}
