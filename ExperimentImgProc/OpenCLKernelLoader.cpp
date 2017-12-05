#include "stdafx.h"
#include "OpenCLKernelLoader.h"



OpenCLKernelLoader * OpenCLKernelLoader::getInstance()
{
	static OpenCLKernelLoader instance;
	return &instance;
}

cl_program OpenCLKernelLoader::getProgram(const char * program_path)
{
	FILE * fp = fopen(program_path, "rb");
	size_t sourceLength;
	char * sourceCode;

	if (!fp)
	{
		CString errMsg;
		errMsg.Format(_T("Failed to load kernel.\n"));
		AfxMessageBox(errMsg);
	}
	fseek(fp, 0, SEEK_END);
	sourceLength = ftell(fp);
	rewind(fp);
	sourceCode = (char *) malloc(sourceLength+1);
	sourceCode[sourceLength] = '\0';
	fread(sourceCode, sizeof(char), sourceLength, fp);
	fclose(fp);
	cl_program program = clCreateProgramWithSource(m_pCLContext, 1, (const char **) &sourceCode, (const size_t *) &sourceLength, &m_nRet);
	m_nRet = clBuildProgram(program, 1, &m_pCLDeviceID, NULL, NULL, NULL);
	checkError(m_nRet, __LINE__);

	size_t logSize;
	clGetProgramBuildInfo(program, m_pCLDeviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
	char* buildLog = new char[logSize];
	clGetProgramBuildInfo(program, m_pCLDeviceID, CL_PROGRAM_BUILD_LOG, sizeof(char)*logSize, buildLog, NULL);


	delete[] buildLog;

	return program;
}

cl_kernel OpenCLKernelLoader::getKernel(const char * kernel_name, cl_program program)
{
	cl_kernel kernel = clCreateKernel(program, kernel_name, &m_nRet);
	checkError(m_nRet, __LINE__);
	return kernel;
}

cl_context OpenCLKernelLoader::getContext()
{
	return m_pCLContext;
}

cl_command_queue OpenCLKernelLoader::getQueue()
{
	return m_pCLCommandQueue;
}

void OpenCLKernelLoader::finishQueue()
{
	m_nRet = clFlush(m_pCLCommandQueue);
	m_nRet = clFinish(m_pCLCommandQueue);
}

OpenCLKernelLoader::OpenCLKernelLoader()
{
	// get platform
	m_nRet = clGetPlatformIDs(0, NULL, &m_nPlatformNum);
	checkError(m_nRet, __LINE__);
	m_nRet = clGetPlatformIDs(m_nPlatformNum, &m_pCLPlatformID, NULL);
	checkError(m_nRet, __LINE__);

	// get device
	m_nRet = clGetDeviceIDs(m_pCLPlatformID, CL_DEVICE_TYPE_ALL, 0, NULL, &m_nDeviceNum);
	checkError(m_nRet, __LINE__);
	cl_device_id *deviceIDs = (cl_device_id*) malloc(m_nDeviceNum * sizeof(cl_device_id));
	m_nRet = clGetDeviceIDs(m_pCLPlatformID, CL_DEVICE_TYPE_ALL, 2, deviceIDs, NULL);
	checkError(m_nRet, __LINE__);
	m_pCLDeviceID = deviceIDs[m_nDeviceNum-1];
	free(deviceIDs);

	//// debug
	//size_t param;
	//m_nRet = clGetDeviceInfo(m_pCLDeviceID, CL_DEVICE_EXTENSIONS, 0, NULL, &param);
	//checkError(m_nRet, __LINE__);
	//char* info = (char*) malloc(sizeof(char)*param);
	//m_nRet = clGetDeviceInfo(m_pCLDeviceID, CL_DEVICE_EXTENSIONS, param, info, NULL);
	//checkError(m_nRet, __LINE__);
	//free(info);

	// create context and queue
	cl_context_properties props[] =
	{
		CL_CONTEXT_PLATFORM, (cl_context_properties) m_pCLPlatformID, 0
	};
	m_pCLContext = clCreateContext(props, 1, &m_pCLDeviceID, NULL, NULL, &m_nRet);
	checkError(m_nRet, __LINE__);

	m_pCLCommandQueue = clCreateCommandQueue(m_pCLContext, m_pCLDeviceID, 0, &m_nRet);
	checkError(m_nRet, __LINE__);
}


OpenCLKernelLoader::~OpenCLKernelLoader()
{
	clReleaseCommandQueue(m_pCLCommandQueue);
	clReleaseContext(m_pCLContext);
	clReleaseDevice(m_pCLDeviceID);
}

void OpenCLKernelLoader::checkError(cl_int ret, int line)
{
	if (CL_SUCCESS != ret)
	{
		CString errMsg;
		errMsg.Format(_T("Failed at %d line in \"OpenCLKernelLoader.cpp\".\n"), line);
		AfxMessageBox(errMsg);
	}
}
