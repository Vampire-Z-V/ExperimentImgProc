#pragma once
#define MAX_THREAD_NUM 5

class MultiThreadController
{
public:
	static MultiThreadController * getInstance();

	void syncInit(int thread_num);
	void syncEnd();
	void waitFor(int target_thread_id);
	void waitInLine(int curr_thread_id);
	void waitAllFinished();
	void quitFromLine();
	void finish();
	void finish(int curr_thread_id);

protected:
	MultiThreadController();
	~MultiThreadController();
	MultiThreadController(const MultiThreadController &);
	MultiThreadController & operator= (const MultiThreadController &);

private:
	int m_nThreadNum;
	int m_nThreadOrder;
	int m_nThreadFinishedCount;
	bool *m_pbThreadFinished;
};

