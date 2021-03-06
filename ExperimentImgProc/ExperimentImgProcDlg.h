// ExperimentImgProcDlg.h : header file
//

#pragma once
#include "afxcmn.h"
#include "afxwin.h"
#include "ImageProcess.h"
#include "MultiThreadController.h"
#include "OpenCLKernelLoader.h"

#define OPERATION_NUM 7
#define TECHNIQUE_NUM 5

#define ADDNOISE_INDEX        0
#define MEDIANFILTER_INDEX    1
#define ROTATE_INDEX          2
#define SCALE_INDEX           3
#define AUTOLEVELS_INDEX      4
#define AUTOWHITE_INDEX       5
#define BILATERALFILTER_INDEX 6

#define WIN_INDEX             0
#define OPENMP_INDEX          1
#define BOOST_INDEX           2
#define CUDA_INDEX            3
#define OPENCL_INDEX          4

struct Thread
{
	CWinThread * thread;
	bool isSuspend;
};
using ProcessesFn = UINT(*)(LPVOID);


//struct DistributionParam
//{
//	int thread_num;
//	ProcessesFn proc_fn;
//};

// CExperimentImgProcDlg dialog
class CExperimentImgProcDlg : public CDialog
{
	// Construction
public:
	CExperimentImgProcDlg(CWnd* pParent = NULL);	// standard constructor
	virtual ~CExperimentImgProcDlg();
	virtual BOOL PreTranslateMessage(MSG* pMsg);

	// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_EXPERIMENTIMGPROC_DIALOG };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	CImage* getImage(int num = 0) { return m_pImgSrc[num]; }
	static UINT PictureRender(LPVOID p);
	static UINT LogPrinter(LPVOID p);

	void PrintPicture(CImage *pImgSrc, CStatic &cPicPanel);
	void ClearPicture();
	void PrintLog(CString &info);
	void PrintPictureInfo(CString & path, int width=0, int height=0);
	bool CheckLoop();
	bool CheckPicture();
	void EnableItems(bool enable = true);
	void UseOneLoop();
	bool CanUseMedianFilter();
	void AfterChangeRadio();
	void CreateTempImg(bool isRotate);

	void StartProcess();
	void FinishProcess(UINT msg=0);
	void ResumeThread(Thread & thread);
	void SuspendThread(Thread & thread);

	void DepartParam(ThreadParam & dst, int index);
	static UINT WIN_threads_distribute(LPVOID p);
	static UINT OPENMP_threads_distribute(LPVOID p);
	static UINT BOOST_threads_distribute(LPVOID p);

	void AddNoise_WIN();
	void MedianFilter_WIN();
	void Rotate_WIN();
	void Scale_WIN();
	void AutoLevels_WIN();
	void AutoWhite_WIN();
	void BilateralFilter_WIN();

	void AddNoise_OPENMP();
	void MedianFilter_OPENMP();
	void Rotate_OPENMP();
	void Scale_OPENMP();
	void AutoLevels_OPENMP();
	void AutoWhite_OPENMP();
	void BilateralFilter_OPENMP();

	void AddNoise_BOOST();
	void MedianFilter_BOOST();
	void Rotate_BOOST();
	void Scale_BOOST();
	void AutoLevels_BOOST();
	void AutoWhite_BOOST();
	void BilateralFilter_BOOST();

	void MedianFilter_CUDA();
	void Rotate_CUDA();
	void Scale_CUDA();
	void AutoWhite_CUDA();

	void MedianFilter_OPENCL();
	void Rotate_OPENCL();
	void Scale_OPENCL();
	void AutoWhite_OPENCL();

	static UINT Rotate_Scale_GPU(LPVOID p);
	static UINT AutoWhite_GPU(LPVOID p);

	// Implementation
protected:
	HICON m_hIcon;
	CImage * m_pImgSrc[2];
	CImage * m_pImgCpy[2];
	CImage * m_pImgTemp[2];
	CString m_strImgPath[2];
	bool m_bAddedNoise[2];
	bool m_bUsedImgTemp;
	int m_nDegree;
	float m_fScaleRatio;
	CString m_strLogs;
	bool m_bIsRotate;
	bool m_bUsedCUDA;

	using OperationsFn = void(CExperimentImgProcDlg::*)();
	OperationsFn m_arrOpsFns[OPERATION_NUM][TECHNIQUE_NUM];
	// bool m_arrbCanProcess[OPERATION_NUM][TECHNIQUE_NUM];

	int m_nThreadNum;
	LARGE_INTEGER freq;
	LARGE_INTEGER startTime;
	LARGE_INTEGER recTime;
	BasicThreadParam* m_pBasicThreadParam;
	RotateThreadParam* m_pRotateThreadParam;
	ScaleThreadParam* m_pScaleThreadParam;

	ThreadParam * m_pThreadParam;
	ProcessesFn m_procFn;

	Thread m_CPictureRender;
	Thread m_CLogPrinter;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()

	afx_msg LRESULT OnThreadMsgReceived(WPARAM wParam, LPARAM lParam);
	
public:
	afx_msg void OnBnClickedButtonOpen();
	afx_msg void OnBnClickedRadioPicture1();
	afx_msg void OnBnClickedRadioPicture2();
	afx_msg void OnBnClickedButtonMarge();
	afx_msg void OnCbnSelchangeComboOperation();
	afx_msg void OnCbnSelchangeComboTech();
	afx_msg void OnNMCustomdrawSliderThread(NMHDR *pNMHDR, LRESULT *pResult);
	afx_msg void OnEnChangeEditLoop();
	afx_msg void OnBnClickedButtonProc();
	afx_msg void OnNMCustomdrawSliderRotate(NMHDR *pNMHDR, LRESULT *pResult);
	afx_msg void OnNMCustomdrawSliderScale(NMHDR *pNMHDR, LRESULT *pResult);
protected:
	CSliderCtrl m_SliderThreads;
	CSliderCtrl m_SliderRotate;
	CString m_strRotateAngle;
	CString m_strScale;
	CSliderCtrl m_SliderScale;
	CStatic m_Picture[2];
	CComboBox m_ComboOperation;
	CComboBox m_ComboTech;
	int m_nLoop;
	CEdit m_EditLoop;
	int m_nRadioPicture;
	CEdit m_EditLog;
	CButton m_ButtonProc;
	CButton m_ButtonOpen;
	CButton m_ButtonMarge;
};
