// MargePictureDlg.cpp : implementation file
//

#include "stdafx.h"
#include "ExperimentImgProc.h"
#include "MargePictureDlg.h"
#include "afxdialogex.h"



IMPLEMENT_DYNAMIC(MargePictureDlg, CDialogEx)

MargePictureDlg::MargePictureDlg(CImage *src_1, CImage *src_2, int tech, int threads, CWnd* pParent)
	: CDialogEx(IDD_MARGE, pParent)
	, m_nThreadNum(threads)
	, m_strAlpha(_T("0.50"))
{
	m_pImgSrc[0] = src_1;
	m_pImgSrc[1] = src_2;
	m_pMargedImg =	NULL;
	m_nTech = tech;
	m_pMargeThreadParam = new MargeThreadParam[MAX_THREAD];

}

MargePictureDlg::~MargePictureDlg()
{
	m_pPictureRenderThread->SuspendThread();
	delete[] m_pMargeThreadParam;
	m_pMargedImg->Destroy();
}

void MargePictureDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_COMBO_MARGETECH, m_ComboTech);
	DDX_Control(pDX, IDC_SLIDER_MARGETHREAD, m_SliderThreads);
	DDX_Control(pDX, IDC_BUTTON_MARGE_PROCESS, m_ButtonProc);
	DDX_Control(pDX, IDC_STATIC_MARGEPICTURE, m_Picture);
	DDX_Text(pDX, IDC_STATIC_MARGETHREAD_NUM, m_nThreadNum);
	DDX_Text(pDX, IDC_STATIC_ALPHA, m_strAlpha);
	DDX_Control(pDX, IDC_SLIDER_ALPHA, m_SliderAlpha);
}

BOOL MargePictureDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}


	// TODO: Add extra initialization here

	m_ComboTech.InsertString(0, _T("WIN"));
	m_ComboTech.InsertString(1, _T("OpenMP"));
	m_ComboTech.SetCurSel(m_nTech);

	m_SliderThreads.SetRange(MIN_THREAD, MAX_THREAD, TRUE);
	m_SliderThreads.SetPos(m_nThreadNum);

	m_SliderAlpha.SetRange(0, 100, TRUE);
	m_SliderAlpha.SetPos(50);

	m_ButtonProc.EnableWindow(FALSE);

	m_pPictureRenderThread=AfxBeginThread((AFX_THREADPROC) &MargePictureDlg::Update, this, 0, 0, CREATE_SUSPENDED);
	Marge_WIN();
	return FALSE;  // return TRUE  unless you set the focus to a control
}

UINT MargePictureDlg::Update(void * p)
{
	while (1)
	{
		Sleep(20);
		MargePictureDlg* dlg = (MargePictureDlg*) p;
		if(dlg->m_pMargedImg != NULL)
			dlg->PrintPicture(dlg->m_pMargedImg, dlg->m_Picture);
	}
	return 0;
}

void MargePictureDlg::PrintPicture(CImage * pImgSrc, CStatic & cPicPanel)
{
	if (pImgSrc != NULL)
	{
		int height;
		int width;
		CRect rect;
		CRect rect1;
		height = pImgSrc->GetHeight();
		width = pImgSrc->GetWidth();

		cPicPanel.GetClientRect(&rect);
		int rect_width = rect.Width() - 4;
		int rect_height = rect.Height() - 4;

		CDC *pDC = cPicPanel.GetDC();
		SetStretchBltMode(pDC->m_hDC, STRETCH_HALFTONE);

		if (width > rect_width || height > rect_height)
		{
			float xScale = (float) rect_width / (float) width;
			float yScale = (float) rect_height / (float) height;
			float ScaleIndex = (xScale <= yScale ? xScale : yScale);
			width *= ScaleIndex;
			height *= ScaleIndex;
		}

		rect1 = CRect(
			CPoint((int) ((rect.Width() - width) / 2), (int) ((rect.Height() - height) / 2)),
			CSize((int) width, (int) height)
		);
		pImgSrc->StretchBlt(pDC->m_hDC, rect1, SRCCOPY);
		ReleaseDC(pDC);
	}
}

void MargePictureDlg::Marge_WIN()
{
	float alpha = (float) m_SliderAlpha.GetPos() / 100;
	int width = m_pImgSrc[0]->GetWidth();
	int height = m_pImgSrc[0]->GetHeight();
	int bpp = m_pImgSrc[0]->GetBPP();

	if (m_pMargedImg != NULL)
	{
		m_pMargedImg->Destroy();
	}
	m_pMargedImg = new CImage();
	m_pMargedImg->Create(width, height, bpp);
	m_pPictureRenderThread->ResumeThread();

	int subLength = width * height / m_nThreadNum;
	for (int i = 0; i < m_nThreadNum; ++i)
	{
		m_pMargeThreadParam[i].startIndex = i * subLength;
		m_pMargeThreadParam[i].endIndex = i != m_nThreadNum - 1 ?
			(i + 1) * subLength - 1 : width * height - 1;
		m_pMargeThreadParam[i].src_1 = m_pImgSrc[0];
		m_pMargeThreadParam[i].src_2 = m_pImgSrc[1];
		m_pMargeThreadParam[i].target = m_pMargedImg;
		m_pMargeThreadParam[i].alpha = alpha;
		m_pMargeThreadParam[i].handle = this->m_hWnd;

		AfxBeginThread((AFX_THREADPROC) &ImageProcess::marge, &m_pMargeThreadParam[i]);
	}
}


BEGIN_MESSAGE_MAP(MargePictureDlg, CDialogEx)
	ON_BN_CLICKED(IDC_BUTTON_MARGE_PROCESS, &MargePictureDlg::OnBnClickedButtonMargeProcess)
	ON_NOTIFY(NM_CUSTOMDRAW, IDC_SLIDER_ALPHA, &MargePictureDlg::OnNMCustomdrawSliderAlpha)
	ON_MESSAGE(WM_MARGE, &MargePictureDlg::OnMargeThreadMsgReceived)
	ON_NOTIFY(NM_CUSTOMDRAW, IDC_SLIDER_MARGETHREAD, &MargePictureDlg::OnNMCustomdrawSliderMargethread)
END_MESSAGE_MAP()


// MargePictureDlg dialog


void MargePictureDlg::OnBnClickedButtonMargeProcess()
{
	// TODO: Add your control notification handler code here

	m_ButtonProc.EnableWindow(FALSE);

	switch (m_ComboTech.GetCurSel())
	{
	case 0:
		Marge_WIN();
		break;
	case 1:
		
	case 2:
		
	default:
		m_ButtonProc.EnableWindow();
		break;
	}

}

void MargePictureDlg::OnNMCustomdrawSliderAlpha(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMCUSTOMDRAW pNMCD = reinterpret_cast<LPNMCUSTOMDRAW>(pNMHDR);
	// TODO: Add your control notification handler code here
	float alpha = (float) m_SliderAlpha.GetPos();
	m_strAlpha.Format(_T("%.2f"), alpha /= 100);
	UpdateData(FALSE);
	*pResult = 0;
}

LRESULT MargePictureDlg::OnMargeThreadMsgReceived(WPARAM wParam, LPARAM lParam)
{
	static int tempCount = 0;
	static LARGE_INTEGER currTime;
	static CString logInfo("test");

	if ((int) wParam == 1)
		tempCount++;

	if (m_nThreadNum == tempCount)
	{
		tempCount = 0;
		m_ButtonProc.EnableWindow();
		m_pPictureRenderThread->SuspendThread();
		PrintPicture(m_pMargedImg, m_Picture);
	}
	return 0;
}


void MargePictureDlg::OnNMCustomdrawSliderMargethread(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMCUSTOMDRAW pNMCD = reinterpret_cast<LPNMCUSTOMDRAW>(pNMHDR);
	// TODO: Add your control notification handler code here
	m_nThreadNum = m_SliderThreads.GetPos();
	UpdateData(FALSE);
	*pResult = 0;
}
