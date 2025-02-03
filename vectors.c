#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define N 1024

const char* kernelVectorAdd =
        "__kernel void vector_add(__global const float* A, __global const float* B, __global float* C) {"
        "    int id = get_global_id(0);"
        "    C[id] = A[id] + B[id];"
        "}";

int main() {
    float *A, *B, *C;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem bufA, bufB, bufC;
    size_t dataSize = N * sizeof(float);

    A = (float*)malloc(dataSize);
    B = (float*)malloc(dataSize);
    C = (float*)malloc(dataSize);

    for (int i = 0; i < N; i++) {
        A[i] = i * 1.0f;
        B[i] = i * 2.0f;
    }
    // init OpenCl
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);

    program = clCreateProgramWithSource(context, 1, &kernelVectorAdd, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "vector_add", NULL);

    bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dataSize, A, NULL);
    bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dataSize, B, NULL);
    bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize, NULL, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    // launching an OpenCL kernel
    size_t globalSize = N;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, dataSize, C, 0, NULL, NULL);

    for (int i = 0; i < 10; i++) {
        printf("A[%d] + B[%d] = %f\n", i, i, C[i]);
    }

    printf("...\n");

    for (int i = N - 10; i < N; i++) {
        printf("A[%d] + B[%d] = %f\n", i, i, C[i]);
    }

    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(A);
    free(B);
    free(C);

    return 0;
}
