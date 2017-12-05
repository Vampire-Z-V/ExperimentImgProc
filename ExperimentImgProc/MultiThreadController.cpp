#include "stdafx.h"
#include "MultiThreadController.h"


MultiThreadController * MultiThreadController::getInstance()
{
	static MultiThreadController instance;
	return &instance;
}

void MultiThreadController::syncInit(int thread_num)
{
	m_nThreadNum = thread_num;
	m_nThreadOrder = 0;
	m_nThreadFinishedCount = 0;
	m_pbThreadFinished = new bool[thread_num];
	for (int i = 0; i < thread_num; i++)
	{
		m_pbThreadFinished[i] = false;
	}
}

void MultiThreadController::syncEnd()
{
	delete[] m_pbThreadFinished;
	m_pbThreadFinished = NULL;
}

void MultiThreadController::waitFor(int target_thread_id)
{
	while (!m_pbThreadFinished[target_thread_id]);
}

void MultiThreadController::waitInLine(int curr_thread_id)
{
	while (curr_thread_id != m_nThreadOrder);
}

void MultiThreadController::waitAllFinished()
{
	while (m_nThreadFinishedCount != m_nThreadNum);
}

void MultiThreadController::quitFromLine()
{
	m_nThreadOrder++;
}

void MultiThreadController::finish()
{
	m_nThreadFinishedCount++;
}

void MultiThreadController::finish(int curr_thread_id)
{
	m_pbThreadFinished[curr_thread_id] = true;
}

MultiThreadController::MultiThreadController()
{
}

MultiThreadController::~MultiThreadController()
{
}
