#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <CL\cl.h>

class OpenCLKernelLoader
{
public:
	static OpenCLKernelLoader * getInstance();
	cl_program getProgram(const char * program_path);
	cl_kernel getKernel(const char * kernel_name, cl_program program);
	cl_context getContext();
	cl_command_queue getQueue();
	void finishQueue();
	void checkError(cl_int ret, int line);


protected:
	OpenCLKernelLoader();
	~OpenCLKernelLoader();
	OpenCLKernelLoader(const OpenCLKernelLoader &);
	OpenCLKernelLoader & operator= (const OpenCLKernelLoader &);

private:
	cl_int           m_nRet;
	cl_uint          m_nPlatformNum;
	cl_uint          m_nDeviceNum;

	cl_platform_id   m_pCLPlatformID;
	cl_device_id     m_pCLDeviceID;
	cl_context       m_pCLContext;
	cl_command_queue m_pCLCommandQueue;
};

