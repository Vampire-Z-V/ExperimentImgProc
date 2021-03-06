#pragma once
#include "afxwin.h"
#include "afxcmn.h"
#include "ImageProcess.h"


// MargePictureDlg header file

class MargePictureDlg : public CDialogEx
{
	DECLARE_DYNAMIC(MargePictureDlg)

public:
	MargePictureDlg(CImage *src_1,CImage *src_2,int tech,int threads,CWnd* pParent = NULL);   // standard constructor
	virtual ~MargePictureDlg();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_MARGE };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	virtual BOOL OnInitDialog();
	DECLARE_MESSAGE_MAP()

	static UINT Update(void* p);

	void PrintPicture(CImage *pImgSrc, CStatic &cPicPanel);

	void Marge_WIN();

public:
	afx_msg void OnBnClickedButtonMargeProcess();
	afx_msg void OnNMCustomdrawSliderAlpha(NMHDR *pNMHDR, LRESULT *pResult);
	afx_msg LRESULT OnMargeThreadMsgReceived(WPARAM wParam, LPARAM lParam);
	CComboBox m_ComboTech;
	CSliderCtrl m_SliderThreads;
	CButton m_ButtonProc;

protected:
	CImage * m_pImgSrc[2];
	CImage * m_pMargedImg;
	int m_nTech;

	MargeThreadParam* m_pMargeThreadParam;
	CWinThread* m_pPictureRenderThread;

public:
	CStatic m_Picture;
	int m_nThreadNum;
	CString m_strAlpha;
	CSliderCtrl m_SliderAlpha;
	afx_msg void OnNMCustomdrawSliderMargethread(NMHDR *pNMHDR, LRESULT *pResult);
};
