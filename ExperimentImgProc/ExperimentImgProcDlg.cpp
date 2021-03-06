// ExperimentImgProcDlg.cpp : implementation file
//

#include "stdafx.h"
#include "ExperimentImgProc.h"
#include "ExperimentImgProcDlg.h"
#include "MargePictureDlg.h"
#include "afxdialogex.h"
#include <omp.h>
#include <boost\thread\thread.hpp>
#include <boost\bind.hpp>

extern "C"
void Rotate_Scale(
	int srcWidth, int srcHeight,
	int targetWidth, int targetHeight,
	bool isRotate,
	double radian,
	bool isRGB,
	int * srcR, int * srcG, int * srcB,
	int * resultR, int * resultG, int * resultB,
	float * weight
);

extern "C"
void collectRGB(
	int srcWidth, int srcHeight,
	bool isRGB,
	int * srcR, int * srcG, int * srcB,
	double * totalR, double * totalG, double * totalB
);

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#define GET_MS(start_t, end_t) \
	((end_t.QuadPart - start_t.QuadPart) * 1000 / freq.QuadPart)

#define ROUNDUP(groupSize, globalSize) \
	(globalSize % groupSize == 0 ? globalSize : (globalSize + groupSize - globalSize % groupSize))

// CAboutDlg dialog used for App About

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

	// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()

// CExperimentImgProcDlg dialog
CExperimentImgProcDlg::CExperimentImgProcDlg(CWnd* pParent /*=NULL*/)
	: CDialog(IDD_EXPERIMENTIMGPROC_DIALOG, pParent)
	, m_nThreadNum(MAX_THREAD)
	, m_strRotateAngle(_T("0°"))
	, m_strScale(_T("x1"))
	, m_nLoop(1)
	, m_nRadioPicture(0)
	, m_nDegree(0)
	, m_fScaleRatio(0.0f)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);

	m_pBasicThreadParam = new BasicThreadParam[MAX_THREAD];
	m_pRotateThreadParam = new RotateThreadParam[MAX_THREAD];
	m_pScaleThreadParam = new ScaleThreadParam[MAX_THREAD];

	m_pThreadParam = new ThreadParam[MAX_THREAD];

	QueryPerformanceFrequency(&freq);

	for (int i = 0; i < 2; i++)
	{
		m_pImgCpy[i] = new CImage();
		m_pImgTemp[i] = new CImage();
		m_bAddedNoise[i] = false;
	}
	m_bUsedImgTemp = false;

	m_arrOpsFns[ADDNOISE_INDEX       ][WIN_INDEX] = &CExperimentImgProcDlg::AddNoise_WIN;
	m_arrOpsFns[MEDIANFILTER_INDEX   ][WIN_INDEX] = &CExperimentImgProcDlg::MedianFilter_WIN;
	m_arrOpsFns[ROTATE_INDEX         ][WIN_INDEX] = &CExperimentImgProcDlg::Rotate_WIN;
	m_arrOpsFns[SCALE_INDEX          ][WIN_INDEX] = &CExperimentImgProcDlg::Scale_WIN;
	m_arrOpsFns[AUTOLEVELS_INDEX     ][WIN_INDEX] = &CExperimentImgProcDlg::AutoLevels_WIN;
	m_arrOpsFns[AUTOWHITE_INDEX      ][WIN_INDEX] = &CExperimentImgProcDlg::AutoWhite_WIN;
	m_arrOpsFns[BILATERALFILTER_INDEX][WIN_INDEX] = &CExperimentImgProcDlg::BilateralFilter_WIN;

	m_arrOpsFns[ADDNOISE_INDEX       ][OPENMP_INDEX] = &CExperimentImgProcDlg::AddNoise_OPENMP;
	m_arrOpsFns[MEDIANFILTER_INDEX   ][OPENMP_INDEX] = &CExperimentImgProcDlg::MedianFilter_OPENMP;
	m_arrOpsFns[ROTATE_INDEX         ][OPENMP_INDEX] = &CExperimentImgProcDlg::Rotate_OPENMP;
	m_arrOpsFns[SCALE_INDEX          ][OPENMP_INDEX] = &CExperimentImgProcDlg::Scale_OPENMP;
	m_arrOpsFns[AUTOLEVELS_INDEX     ][OPENMP_INDEX] = &CExperimentImgProcDlg::AutoLevels_OPENMP;
	m_arrOpsFns[AUTOWHITE_INDEX      ][OPENMP_INDEX] = &CExperimentImgProcDlg::AutoWhite_OPENMP;
	m_arrOpsFns[BILATERALFILTER_INDEX][OPENMP_INDEX] = &CExperimentImgProcDlg::BilateralFilter_OPENMP;

	m_arrOpsFns[ADDNOISE_INDEX       ][BOOST_INDEX] = &CExperimentImgProcDlg::AddNoise_BOOST;
	m_arrOpsFns[MEDIANFILTER_INDEX   ][BOOST_INDEX] = &CExperimentImgProcDlg::MedianFilter_BOOST;
	m_arrOpsFns[ROTATE_INDEX         ][BOOST_INDEX] = &CExperimentImgProcDlg::Rotate_BOOST;
	m_arrOpsFns[SCALE_INDEX          ][BOOST_INDEX] = &CExperimentImgProcDlg::Scale_BOOST;
	m_arrOpsFns[AUTOLEVELS_INDEX     ][BOOST_INDEX] = &CExperimentImgProcDlg::AutoLevels_BOOST;
	m_arrOpsFns[AUTOWHITE_INDEX      ][BOOST_INDEX] = &CExperimentImgProcDlg::AutoWhite_BOOST;
	m_arrOpsFns[BILATERALFILTER_INDEX][BOOST_INDEX] = &CExperimentImgProcDlg::BilateralFilter_BOOST;

	m_arrOpsFns[ADDNOISE_INDEX       ][CUDA_INDEX] = NULL;
	m_arrOpsFns[MEDIANFILTER_INDEX   ][CUDA_INDEX] = NULL;
	m_arrOpsFns[ROTATE_INDEX         ][CUDA_INDEX] = &CExperimentImgProcDlg::Rotate_CUDA;
	m_arrOpsFns[SCALE_INDEX          ][CUDA_INDEX] = &CExperimentImgProcDlg::Scale_CUDA;
	m_arrOpsFns[AUTOLEVELS_INDEX     ][CUDA_INDEX] = NULL;
	m_arrOpsFns[AUTOWHITE_INDEX      ][CUDA_INDEX] = &CExperimentImgProcDlg::AutoWhite_CUDA;
	m_arrOpsFns[BILATERALFILTER_INDEX][CUDA_INDEX] = NULL;

	m_arrOpsFns[ADDNOISE_INDEX       ][OPENCL_INDEX] = NULL;
	m_arrOpsFns[MEDIANFILTER_INDEX   ][OPENCL_INDEX] = NULL;
	m_arrOpsFns[ROTATE_INDEX         ][OPENCL_INDEX] = &CExperimentImgProcDlg::Rotate_OPENCL;
	m_arrOpsFns[SCALE_INDEX          ][OPENCL_INDEX] = &CExperimentImgProcDlg::Scale_OPENCL;
	m_arrOpsFns[AUTOLEVELS_INDEX     ][OPENCL_INDEX] = NULL;
	m_arrOpsFns[AUTOWHITE_INDEX      ][OPENCL_INDEX] = &CExperimentImgProcDlg::AutoWhite_OPENCL;
	m_arrOpsFns[BILATERALFILTER_INDEX][OPENCL_INDEX] = NULL;
}

CExperimentImgProcDlg::~CExperimentImgProcDlg()
{
	delete[] m_pBasicThreadParam;
	delete[] m_pRotateThreadParam;
	delete[] m_pScaleThreadParam;

	delete[] m_pThreadParam;

	for (int i = 0; i < 2; i++)
	{
		if (m_pImgSrc[i] != NULL)
			m_pImgSrc[i]->Destroy();
		if (m_pImgCpy[i] != NULL)
			m_pImgCpy[i]->Destroy();
		if (m_pImgTemp[i] != NULL)
			m_pImgTemp[i]->Destroy();
	}
}

void CExperimentImgProcDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_STATIC_THREADNUM, m_nThreadNum);
	DDX_Control(pDX, IDC_SLIDER_THREAD, m_SliderThreads);
	DDX_Control(pDX, IDC_SLIDER_ROTATE, m_SliderRotate);
	DDX_Text(pDX, IDC_STATIC_ROTATEANGLE, m_strRotateAngle);
	DDX_Text(pDX, IDC_STATIC_SCALE, m_strScale);
	DDX_Control(pDX, IDC_SLIDER_SCALE, m_SliderScale);
	DDX_Control(pDX, IDC_PICTURE_1, m_Picture[0]);
	DDX_Control(pDX, IDC_PICTURE_2, m_Picture[1]);
	DDX_Control(pDX, IDC_COMBO_OPERATION, m_ComboOperation);
	DDX_Control(pDX, IDC_COMBO_TECH, m_ComboTech);
	DDX_Text(pDX, IDC_EDIT_LOOP, m_nLoop);
	DDX_Control(pDX, IDC_EDIT_LOOP, m_EditLoop);
	DDX_Radio(pDX, IDC_RADIO_PICTURE1, m_nRadioPicture);
	DDX_Control(pDX, IDC_EDIT_LOG, m_EditLog);
	DDX_Control(pDX, IDC_BUTTON_PROC, m_ButtonProc);
	DDX_Control(pDX, IDC_BUTTON_OPEN, m_ButtonOpen);
	DDX_Control(pDX, IDC_BUTTON_MARGE, m_ButtonMarge);
}

BEGIN_MESSAGE_MAP(CExperimentImgProcDlg, CDialog)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON_OPEN, &CExperimentImgProcDlg::OnBnClickedButtonOpen)
	ON_CBN_SELCHANGE(IDC_COMBO_TECH, &CExperimentImgProcDlg::OnCbnSelchangeComboTech)
	ON_NOTIFY(NM_CUSTOMDRAW, IDC_SLIDER_THREAD, &CExperimentImgProcDlg::OnNMCustomdrawSliderThread)
	ON_EN_CHANGE(IDC_EDIT_LOOP, &CExperimentImgProcDlg::OnEnChangeEditLoop)
	ON_BN_CLICKED(IDC_BUTTON_PROC, &CExperimentImgProcDlg::OnBnClickedButtonProc)
	ON_BN_CLICKED(IDC_BUTTON_MARGE, &CExperimentImgProcDlg::OnBnClickedButtonMarge)
	ON_NOTIFY(NM_CUSTOMDRAW, IDC_SLIDER_ROTATE, &CExperimentImgProcDlg::OnNMCustomdrawSliderRotate)
	ON_CBN_SELCHANGE(IDC_COMBO_OPERATION, &CExperimentImgProcDlg::OnCbnSelchangeComboOperation)
	ON_BN_CLICKED(IDC_RADIO_PICTURE1, &CExperimentImgProcDlg::OnBnClickedRadioPicture1)
	ON_BN_CLICKED(IDC_RADIO_PICTURE2, &CExperimentImgProcDlg::OnBnClickedRadioPicture2)
	ON_NOTIFY(NM_CUSTOMDRAW, IDC_SLIDER_SCALE, &CExperimentImgProcDlg::OnNMCustomdrawSliderScale)

	ON_MESSAGE(WM_NOISE,            &CExperimentImgProcDlg::OnThreadMsgReceived)
	ON_MESSAGE(WM_MEDIAN_FILTER,    &CExperimentImgProcDlg::OnThreadMsgReceived)
	ON_MESSAGE(WM_ROTATE,           &CExperimentImgProcDlg::OnThreadMsgReceived)
	ON_MESSAGE(WM_SCALE,            &CExperimentImgProcDlg::OnThreadMsgReceived)
	ON_MESSAGE(WM_AUTO_LEVELS,      &CExperimentImgProcDlg::OnThreadMsgReceived)
	ON_MESSAGE(WM_AUTO_WHITE,       &CExperimentImgProcDlg::OnThreadMsgReceived)
	ON_MESSAGE(WM_BILATERAL_FILTER, &CExperimentImgProcDlg::OnThreadMsgReceived)
END_MESSAGE_MAP()

// CExperimentImgProcDlg message handlers

BOOL CExperimentImgProcDlg::OnInitDialog()
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

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here
	m_ComboOperation.InsertString(ADDNOISE_INDEX       , _T("椒盐噪声"));
	m_ComboOperation.InsertString(MEDIANFILTER_INDEX   , _T("中值滤波"));
	m_ComboOperation.InsertString(ROTATE_INDEX         , _T("图片旋转"));
	m_ComboOperation.InsertString(SCALE_INDEX          , _T("图片缩放"));
	m_ComboOperation.InsertString(AUTOLEVELS_INDEX     , _T("自动色阶"));
	m_ComboOperation.InsertString(AUTOWHITE_INDEX      , _T("自动白平衡"));
	m_ComboOperation.InsertString(BILATERALFILTER_INDEX, _T("双边滤波"));
	m_ComboOperation.SetCurSel(0);

	m_ComboTech.InsertString(WIN_INDEX   , _T("WIN"));
	m_ComboTech.InsertString(OPENMP_INDEX, _T("OpenMP"));
	m_ComboTech.InsertString(BOOST_INDEX , _T("Boost"));
	m_ComboTech.InsertString(CUDA_INDEX  , _T("CUDA"));
	m_ComboTech.InsertString(OPENCL_INDEX, _T("OpenCL"));
	m_ComboTech.SetCurSel(0);

	m_SliderThreads.SetRange(MIN_THREAD, MAX_THREAD, TRUE);
	m_SliderThreads.SetPos(MAX_THREAD);

	m_SliderRotate.SetRange(MIN_ROTATE, MAX_ROTATE, TRUE);
	m_SliderRotate.SetPos(0);

	m_SliderScale.SetRange(1, MAX_SCALE * 2 - 1, TRUE);
	m_SliderScale.SetPos(MAX_SCALE);

	m_ButtonProc.EnableWindow(FALSE);

	//m_pPictureRenderThread = AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::Update, this, 0, 0, CREATE_SUSPENDED);
	m_CPictureRender.thread = AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::PictureRender, this, 0, 0, CREATE_SUSPENDED);
	m_CPictureRender.isSuspend = true;
	m_CLogPrinter.thread = AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::LogPrinter, this, 0, 0, CREATE_SUSPENDED);
	m_CLogPrinter.isSuspend = true;
	return FALSE;  // return TRUE  unless you set the focus to a control
}

void CExperimentImgProcDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialog::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CExperimentImgProcDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialog::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CExperimentImgProcDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

/*********************************** Operations Thread Distribution ***********************************/
#pragma region Operations Thread Distribution

void CExperimentImgProcDlg::DepartParam(ThreadParam & dst, int index)
{
	static int subLength = 0;
	static int width = 0;
	static int height = 0;
	if (index == 0)
	{
		if (m_bUsedImgTemp)
		{
			width = m_pImgTemp[m_nRadioPicture]->GetWidth();
			height = m_pImgTemp[m_nRadioPicture]->GetHeight();
		}
		else
		{
			width = m_pImgSrc[m_nRadioPicture]->GetWidth();
			height = m_pImgSrc[m_nRadioPicture]->GetHeight();
		}
		subLength = width * height / m_nThreadNum;
	}

	dst.imgSrc_1 = m_pImgSrc[m_nRadioPicture];
	dst.startIndex = index * subLength;
	dst.endIndex = index != m_nThreadNum - 1 ? (index + 1) * subLength - 1 : width * height - 1;
	if (m_bUsedImgTemp)
	{
		dst.imgTemp = m_pImgTemp[m_nRadioPicture];
	}
	
	dst.radian = m_nDegree * PI / 180;
	dst.scale_ratio = m_fScaleRatio;

	dst.thread_id = index;
	dst.handle = this->m_hWnd;
}

UINT CExperimentImgProcDlg::WIN_threads_distribute(LPVOID p)
{
	CExperimentImgProcDlg * dlg = (CExperimentImgProcDlg*) p;
	for (int i = 0; i < dlg->m_nThreadNum; ++i)
	{
		dlg->DepartParam(dlg->m_pThreadParam[i], i);
		AfxBeginThread((AFX_THREADPROC) dlg->m_procFn, &dlg->m_pThreadParam[i]);
	}
	return 0;
}

UINT CExperimentImgProcDlg::OPENMP_threads_distribute(LPVOID p)
{
	CExperimentImgProcDlg * dlg = (CExperimentImgProcDlg*) p;
#pragma omp parallel for num_threads(dlg->m_nThreadNum)
	for (int i = 0; i < dlg->m_nThreadNum; ++i)
	{
		dlg->DepartParam(dlg->m_pThreadParam[i], i);
		dlg->m_procFn(&dlg->m_pThreadParam[i]);
	}
	return 0;
}

UINT CExperimentImgProcDlg::BOOST_threads_distribute(LPVOID p)
{
	CExperimentImgProcDlg * dlg = (CExperimentImgProcDlg*) p;
	boost::thread_group td_gp;
	for (int i = 0; i < dlg->m_nThreadNum; ++i)
	{
		dlg->DepartParam(dlg->m_pThreadParam[i], i);
		boost::thread *td = new boost::thread(boost::bind(dlg->m_procFn, &dlg->m_pThreadParam[i]));
		td_gp.add_thread(td);
	}
	td_gp.join_all();
	return 0;
}

#pragma region AddNoise

void CExperimentImgProcDlg::AddNoise_WIN()
{
	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::WIN_threads_distribute, this);
}

void CExperimentImgProcDlg::AddNoise_OPENMP()
{
	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::OPENMP_threads_distribute, this);
}

void CExperimentImgProcDlg::AddNoise_BOOST()
{
	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::BOOST_threads_distribute, this);

}
#pragma endregion

#pragma region MedianFilter

void CExperimentImgProcDlg::MedianFilter_WIN()
{
	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::WIN_threads_distribute, this);
}

void CExperimentImgProcDlg::MedianFilter_OPENMP()
{
	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::OPENMP_threads_distribute, this);
}

void CExperimentImgProcDlg::MedianFilter_BOOST()
{
	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::BOOST_threads_distribute, this);
}
#pragma endregion

#pragma region Rotate

void CExperimentImgProcDlg::Rotate_WIN()
{
	CreateTempImg(true);
	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::WIN_threads_distribute, this);
}

void CExperimentImgProcDlg::Rotate_OPENMP()
{
	CreateTempImg(true);
	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::OPENMP_threads_distribute, this);
}

void CExperimentImgProcDlg::Rotate_BOOST()
{
	CreateTempImg(true);
	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::BOOST_threads_distribute, this);
}
#pragma endregion

#pragma region Scale

void CExperimentImgProcDlg::Scale_WIN()
{
	CreateTempImg(false);
	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::WIN_threads_distribute, this);
}

void CExperimentImgProcDlg::Scale_OPENMP()
{
	CreateTempImg(false);
	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::OPENMP_threads_distribute, this);
}

void CExperimentImgProcDlg::Scale_BOOST()
{
	CreateTempImg(false);
	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::BOOST_threads_distribute, this);
}
#pragma endregion

#pragma region AutoLevels

void CExperimentImgProcDlg::AutoLevels_WIN()
{
	MultiThreadController::getInstance()->syncInit(m_nThreadNum);
	AutoLevel::init();

	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::WIN_threads_distribute, this);
}

void CExperimentImgProcDlg::AutoLevels_OPENMP()
{
	MultiThreadController::getInstance()->syncInit(m_nThreadNum);
	AutoLevel::init();

	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::OPENMP_threads_distribute, this);
}

void CExperimentImgProcDlg::AutoLevels_BOOST()
{
	MultiThreadController::getInstance()->syncInit(m_nThreadNum);
	AutoLevel::init();

	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::BOOST_threads_distribute, this);
}
#pragma endregion

#pragma region AutoWhite

void CExperimentImgProcDlg::AutoWhite_WIN()
{
	MultiThreadController::getInstance()->syncInit(m_nThreadNum);
	AutoWhite::init();

	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::WIN_threads_distribute, this);
}

void CExperimentImgProcDlg::AutoWhite_OPENMP()
{
	MultiThreadController::getInstance()->syncInit(m_nThreadNum);
	AutoWhite::init();

	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::OPENMP_threads_distribute, this);
}

void CExperimentImgProcDlg::AutoWhite_BOOST()
{
	MultiThreadController::getInstance()->syncInit(m_nThreadNum);
	AutoWhite::init();

	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::BOOST_threads_distribute, this);
}
#pragma endregion

#pragma region BilateralFilter
void CExperimentImgProcDlg::BilateralFilter_WIN()
{
	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::WIN_threads_distribute, this);
}

void CExperimentImgProcDlg::BilateralFilter_OPENMP()
{
	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::OPENMP_threads_distribute, this);
}

void CExperimentImgProcDlg::BilateralFilter_BOOST()
{
	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::BOOST_threads_distribute, this);
}
#pragma endregion

#pragma endregion
/******************************************************************************************************/

/******************************************* GPU Accelerate *******************************************/
#pragma region GPU Accelerate

#pragma region MedianFilter
void CExperimentImgProcDlg::MedianFilter_CUDA()
{

}
#pragma endregion

#pragma region Rotate
void CExperimentImgProcDlg::Rotate_CUDA()
{
	m_bIsRotate = true;
	m_bUsedCUDA = true;
	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::Rotate_Scale_GPU, this);
}
void CExperimentImgProcDlg::Rotate_OPENCL()
{

	m_bIsRotate = true;
	m_bUsedCUDA = false;
	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::Rotate_Scale_GPU, this);

}
#pragma endregion

#pragma region Scale
void CExperimentImgProcDlg::Scale_CUDA()
{
	m_bIsRotate = false;
	m_bUsedCUDA = true;
	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::Rotate_Scale_GPU, this);
}

void CExperimentImgProcDlg::Scale_OPENCL()
{
	m_bIsRotate = false;
	m_bUsedCUDA = false;
	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::Rotate_Scale_GPU, this);
}
#pragma endregion

#pragma region AutoWhite
void CExperimentImgProcDlg::AutoWhite_CUDA()
{
	AutoWhite::init();
	m_bUsedCUDA = true;
	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::AutoWhite_GPU, this);
}
void CExperimentImgProcDlg::AutoWhite_OPENCL()
{
	AutoWhite::init();
	m_bUsedCUDA = false;
	AfxBeginThread((AFX_THREADPROC) &CExperimentImgProcDlg::AutoWhite_GPU, this);
}
#pragma endregion

UINT CExperimentImgProcDlg::Rotate_Scale_GPU(LPVOID p)
{
#pragma region Common Init
	CExperimentImgProcDlg * dlg = (CExperimentImgProcDlg*) p;
	dlg->CreateTempImg(dlg->m_bIsRotate);
	int targetWidth = dlg->m_pImgTemp[dlg->m_nRadioPicture]->GetWidth();
	int targetHeight = dlg->m_pImgTemp[dlg->m_nRadioPicture]->GetHeight();
	int srcWidth = dlg->m_pImgSrc[dlg->m_nRadioPicture]->GetWidth();
	int srcHeight = dlg->m_pImgSrc[dlg->m_nRadioPicture]->GetHeight();

	int srcPit = dlg->m_pImgSrc[dlg->m_nRadioPicture]->GetPitch();
	int targetPit = dlg->m_pImgTemp[dlg->m_nRadioPicture]->GetPitch();
	int bitCount = dlg->m_pImgTemp[dlg->m_nRadioPicture]->GetBPP() / 8;
	bool isRGB = bitCount != 1;
	double r = dlg->m_bIsRotate ? dlg->m_nDegree * PI / 180 : dlg->m_fScaleRatio;

	byte* pSrcData = (byte*) dlg->m_pImgSrc[dlg->m_nRadioPicture]->GetBits();
	byte* pTargetData = (byte*) dlg->m_pImgTemp[dlg->m_nRadioPicture]->GetBits();

	int srcSize = srcWidth * srcHeight;
	int targetSize = targetWidth * targetHeight;

	COLORREF backColor = GetSysColor(COLOR_3DFACE);

	int *srcR = NULL;
	int *srcG = NULL;
	int *srcB = NULL;
	int *R = NULL;
	int *G = NULL;
	int *B = NULL;
	float *weight = NULL;

	// init
	ImageProcess::getSourceColors(*dlg->m_pImgSrc[dlg->m_nRadioPicture], &srcR, &srcG, &srcB);
	if (isRGB)
	{
		R = new int[targetSize];
		G = new int[targetSize];
		memset(R, 0, sizeof(int) * targetSize);
		memset(G, 0, sizeof(int) * targetSize);
	}
	B = new int[targetSize];
	weight = new float[targetSize];
	memset(B, 0, sizeof(int) * targetSize);
	memset(weight, 0, sizeof(float) * targetSize);
#pragma endregion

	if(dlg->m_bUsedCUDA)
		Rotate_Scale(srcWidth, srcHeight, targetWidth, targetHeight, dlg->m_bIsRotate, r, isRGB, srcR, srcG, srcB, R, G, B, weight);
	else
	{
		int is_rotate = dlg->m_bIsRotate ? 1 : 0;
		int is_rgb = isRGB ? 1 : 0;
		cl_int ret;
		OpenCLKernelLoader *loader = OpenCLKernelLoader::getInstance();
		cl_command_queue command_queue = loader->getQueue();
		cl_context context = loader->getContext();
		cl_program program = loader->getProgram("Rotate_Scale.cl");
		cl_kernel kernel = loader->getKernel("Rotate_Scale", program);

		cl_mem srcR_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, srcSize * sizeof(int), NULL, &ret);
		loader->checkError(ret, __LINE__);
		cl_mem srcG_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, srcSize * sizeof(int), NULL, &ret);
		loader->checkError(ret, __LINE__);
		cl_mem srcB_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, srcSize * sizeof(int), NULL, &ret);
		loader->checkError(ret, __LINE__);
		cl_mem R_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, targetSize * sizeof(int), NULL, &ret);
		loader->checkError(ret, __LINE__);
		cl_mem G_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, targetSize * sizeof(int), NULL, &ret);
		loader->checkError(ret, __LINE__);
		cl_mem B_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, targetSize * sizeof(int), NULL, &ret);
		loader->checkError(ret, __LINE__);
		cl_mem weight_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, targetSize * sizeof(float), NULL, &ret);
		loader->checkError(ret, __LINE__);

		clEnqueueWriteBuffer(command_queue, srcR_mem_obj, CL_TRUE, 0, srcSize * sizeof(int), srcR, 0, NULL, NULL);
		clEnqueueWriteBuffer(command_queue, srcG_mem_obj, CL_TRUE, 0, srcSize * sizeof(int), srcG, 0, NULL, NULL);
		clEnqueueWriteBuffer(command_queue, srcB_mem_obj, CL_TRUE, 0, srcSize * sizeof(int), srcB, 0, NULL, NULL);

		ret = clSetKernelArg(kernel, 0, sizeof(int), (void *) &srcWidth);
		ret = clSetKernelArg(kernel, 1, sizeof(int), (void *) &srcHeight);
		ret = clSetKernelArg(kernel, 2, sizeof(int), (void *) &targetWidth);
		loader->checkError(ret, __LINE__);
		ret = clSetKernelArg(kernel, 3, sizeof(int), (void *) &targetHeight);
		loader->checkError(ret, __LINE__);
		ret = clSetKernelArg(kernel, 4, sizeof(int), (void *) &is_rotate);
		loader->checkError(ret, __LINE__);
		ret = clSetKernelArg(kernel, 5, sizeof(double), (void *) &r);
		loader->checkError(ret, __LINE__);
		ret = clSetKernelArg(kernel, 6, sizeof(int), (void *) &is_rgb);
		loader->checkError(ret, __LINE__);
		ret = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *) &srcR_mem_obj);
		loader->checkError(ret, __LINE__);
		ret = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *) &srcG_mem_obj);
		loader->checkError(ret, __LINE__);
		ret = clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *) &srcB_mem_obj);
		loader->checkError(ret, __LINE__);
		ret = clSetKernelArg(kernel, 10, sizeof(cl_mem), (void *) &R_mem_obj);
		loader->checkError(ret, __LINE__);
		ret = clSetKernelArg(kernel, 11, sizeof(cl_mem), (void *) &G_mem_obj);
		loader->checkError(ret, __LINE__);
		ret = clSetKernelArg(kernel, 12, sizeof(cl_mem), (void *) &B_mem_obj);
		loader->checkError(ret, __LINE__);
		ret = clSetKernelArg(kernel, 13, sizeof(cl_mem), (void *) &weight_mem_obj);
		loader->checkError(ret, __LINE__);

		cl_uint work_dim = 2;
		size_t local_work_size[2] = { 16, 16 };
		size_t global_work_size[2] = {
			ROUNDUP(local_work_size[0], targetWidth),
			ROUNDUP(local_work_size[1], targetHeight)
		};

		ret = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
		loader->checkError(ret, __LINE__);
		if (isRGB)
		{
			ret = clEnqueueReadBuffer(command_queue, R_mem_obj, CL_TRUE, 0, targetSize * sizeof(int), R, 0, NULL, NULL);
			loader->checkError(ret, __LINE__);
			ret = clEnqueueReadBuffer(command_queue, G_mem_obj, CL_TRUE, 0, targetSize * sizeof(int), G, 0, NULL, NULL);
			loader->checkError(ret, __LINE__);
		}
		ret = clEnqueueReadBuffer(command_queue, B_mem_obj, CL_TRUE, 0, targetSize * sizeof(int), B, 0, NULL, NULL);
		loader->checkError(ret, __LINE__);
		clEnqueueReadBuffer(command_queue, weight_mem_obj, CL_TRUE, 0, targetSize * sizeof(float), weight, 0, NULL, NULL);
		loader->checkError(ret, __LINE__);

		loader->finishQueue();
		ret = clReleaseKernel(kernel);
		ret = clReleaseProgram(program);
		ret = clReleaseMemObject(srcR_mem_obj);
		ret = clReleaseMemObject(srcG_mem_obj);
		ret = clReleaseMemObject(srcB_mem_obj);
		ret = clReleaseMemObject(R_mem_obj);
		ret = clReleaseMemObject(G_mem_obj);
		ret = clReleaseMemObject(B_mem_obj);
		ret = clReleaseMemObject(weight_mem_obj);
	}

	// use the results return from gpu process to change the target pixels' color
	for (int index = 0; index < targetSize; index++)
	{
		int x = index % targetWidth;
		int y = index / targetWidth;

		if (isRGB)
		{
			*(pTargetData + targetPit * y + x * bitCount + 2) = R[index] == -1 ? GetBValue(backColor) : R[index] / weight[index];
			*(pTargetData + targetPit * y + x * bitCount + 1) = G[index] == -1 ? GetBValue(backColor) : G[index] / weight[index];
		}
		*(pTargetData + targetPit * y + x * bitCount + 0) = B[index] == -1 ? GetBValue(backColor) : B[index] / weight[index];
	}

	delete[] srcB;
	delete[] srcG;
	delete[] srcR;
	delete[] B;
	delete[] G;
	delete[] R;
	delete[] weight;

	LPARAM lParam = dlg->m_bIsRotate ? WM_ROTATE : WM_SCALE;
	::PostMessage(dlg->m_hWnd, lParam, -1, lParam);
	return 0;
}
UINT CExperimentImgProcDlg::AutoWhite_GPU(LPVOID p)
{
	CExperimentImgProcDlg * dlg = (CExperimentImgProcDlg*) p;
	int srcWidth = dlg->m_pImgSrc[dlg->m_nRadioPicture]->GetWidth();
	int srcHeight = dlg->m_pImgSrc[dlg->m_nRadioPicture]->GetHeight();

	int srcPit = dlg->m_pImgSrc[dlg->m_nRadioPicture]->GetPitch();
	int bitCount = dlg->m_pImgSrc[dlg->m_nRadioPicture]->GetBPP() / 8;
	bool isRGB = bitCount != 1;

	byte* pSrcData = (byte*) dlg->m_pImgSrc[dlg->m_nRadioPicture]->GetBits();

	int srcSize = srcWidth * srcHeight;

	int *srcR = NULL;
	int *srcG = NULL;
	int *srcB = NULL;
	double R = 0.0;
	double G = 0.0;
	double B = 0.0;

	// init
	ImageProcess::getSourceColors(*dlg->m_pImgSrc[dlg->m_nRadioPicture], &srcR, &srcG, &srcB);

	if (dlg->m_bUsedCUDA)
		collectRGB(srcWidth, srcHeight, isRGB, srcR, srcG, srcB, &AutoWhite::R, &AutoWhite::G, &AutoWhite::B);
	else
	{
		int is_rgb = isRGB ? 1 : 0;
		cl_int ret;
		OpenCLKernelLoader *loader = OpenCLKernelLoader::getInstance();
		cl_command_queue command_queue = loader->getQueue();
		cl_context context = loader->getContext();
		cl_program program = loader->getProgram("AutoWhite.cl");
		cl_kernel kernel = loader->getKernel("collection", program);

		cl_mem srcR_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, srcSize * sizeof(int), NULL, &ret);
		loader->checkError(ret, __LINE__);
		cl_mem srcG_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, srcSize * sizeof(int), NULL, &ret);
		loader->checkError(ret, __LINE__);
		cl_mem srcB_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, srcSize * sizeof(int), NULL, &ret);
		loader->checkError(ret, __LINE__);
		cl_mem R_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double), NULL, &ret);
		loader->checkError(ret, __LINE__);
		cl_mem G_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double), NULL, &ret);
		loader->checkError(ret, __LINE__);
		cl_mem B_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double), NULL, &ret);
		loader->checkError(ret, __LINE__);

		clEnqueueWriteBuffer(command_queue, srcR_mem_obj, CL_TRUE, 0, srcSize * sizeof(int), srcR, 0, NULL, NULL);
		clEnqueueWriteBuffer(command_queue, srcG_mem_obj, CL_TRUE, 0, srcSize * sizeof(int), srcG, 0, NULL, NULL);
		clEnqueueWriteBuffer(command_queue, srcB_mem_obj, CL_TRUE, 0, srcSize * sizeof(int), srcB, 0, NULL, NULL);

		ret = clSetKernelArg(kernel, 0, sizeof(int), (void *) &srcWidth);
		ret = clSetKernelArg(kernel, 1, sizeof(int), (void *) &srcHeight);
		ret = clSetKernelArg(kernel, 2, sizeof(int), (void *) &is_rgb);
		loader->checkError(ret, __LINE__);
		ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &srcR_mem_obj);
		loader->checkError(ret, __LINE__);
		ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &srcG_mem_obj);
		loader->checkError(ret, __LINE__);
		ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *) &srcB_mem_obj);
		loader->checkError(ret, __LINE__);
		ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *) &R_mem_obj);
		loader->checkError(ret, __LINE__);
		ret = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *) &G_mem_obj);
		loader->checkError(ret, __LINE__);
		ret = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *) &B_mem_obj);
		loader->checkError(ret, __LINE__);

		cl_uint work_dim = 2;
		size_t local_work_size[2] = { 16, 16 };
		size_t global_work_size[2] = {
			ROUNDUP(local_work_size[0], srcWidth),
			ROUNDUP(local_work_size[1], srcHeight)
		};

		ret = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
		loader->checkError(ret, __LINE__);
		if (isRGB)
		{
			ret = clEnqueueReadBuffer(command_queue, R_mem_obj, CL_TRUE, 0, sizeof(double), &AutoWhite::R, 0, NULL, NULL);
			loader->checkError(ret, __LINE__);
			ret = clEnqueueReadBuffer(command_queue, G_mem_obj, CL_TRUE, 0, sizeof(double), &AutoWhite::G, 0, NULL, NULL);
			loader->checkError(ret, __LINE__);
		}
		ret = clEnqueueReadBuffer(command_queue, B_mem_obj, CL_TRUE, 0, sizeof(double), &AutoWhite::B, 0, NULL, NULL);
		loader->checkError(ret, __LINE__);

		loader->finishQueue();
		ret = clReleaseKernel(kernel);
		ret = clReleaseProgram(program);
		ret = clReleaseMemObject(srcR_mem_obj);
		ret = clReleaseMemObject(srcG_mem_obj);
		ret = clReleaseMemObject(srcB_mem_obj);
		ret = clReleaseMemObject(R_mem_obj);
		ret = clReleaseMemObject(G_mem_obj);
		ret = clReleaseMemObject(B_mem_obj);
	}

	double K;
	double KR;
	double KG;
	double KB;
	if (isRGB)
	{
		K = (AutoWhite::R + AutoWhite::G + AutoWhite::B) / srcSize / 3;
		KR = K / AutoWhite::R * srcSize;
		KG = K / AutoWhite::G * srcSize;
	}
	else
	{
		K = AutoWhite::B / srcSize;
	}
	KB = K / AutoWhite::B * srcSize;
	
	for (int y = 0; y < srcHeight; y++)
		for (int x = 0; x < srcWidth; x++)
		{
			if (isRGB)
			{
				int srcGray_R = *(pSrcData + srcPit * y + x * bitCount + 2) * KR;
				int srcGray_G = *(pSrcData + srcPit * y + x * bitCount + 1) * KG;

				srcGray_R = srcGray_R >= GRAYLEVEL ? GRAYLEVEL - 1 : srcGray_R;
				srcGray_G = srcGray_G >= GRAYLEVEL ? GRAYLEVEL - 1 : srcGray_G;

				*(pSrcData + srcPit * y + x * bitCount + 2) = srcGray_R;
				*(pSrcData + srcPit * y + x * bitCount + 1) = srcGray_G;
			}
			int srcGray_B = *(pSrcData + srcPit * y + x * bitCount + 2) * KB;
			srcGray_B = srcGray_B >= GRAYLEVEL ? GRAYLEVEL - 1 : srcGray_B;
			*(pSrcData + srcPit * y + x * bitCount + 2) = srcGray_B;
		}
		
	::PostMessage(dlg->m_hWnd, WM_AUTO_WHITE, -1, WM_AUTO_WHITE);
	return 0;
}
#pragma endregion
/******************************************************************************************************/

/******************************************* MessageReceive *******************************************/
#pragma region MessageReceive
LRESULT CExperimentImgProcDlg::OnThreadMsgReceived(WPARAM wParam, LPARAM lParam)
{
	static int tempCount = 0;
	static int tempProcessCount = 0;
	static LARGE_INTEGER currTime;
	static CString logInfo;

	if ((int) wParam == 1)
		tempCount++;

	if ((int) wParam == -1)
	{
		logInfo.Format(_T("     GPU加速完成\r\n"));
		m_strLogs += logInfo;
		tempCount = m_nThreadNum;
	}
	else if (m_nLoop == 1)
	{
		logInfo.Format(_T("     %d 个线程完成\r\n"), tempCount);
		m_strLogs += logInfo;
	}

	if (m_nThreadNum == tempCount)
	{
		tempCount = 0;
		tempProcessCount++;

		if (m_nLoop != 1)
		{
			QueryPerformanceCounter(&currTime);
			logInfo.Format(_T("    第 %d 次处理，耗时：%dms\r\n"), tempProcessCount, GET_MS(recTime, currTime));
			m_strLogs += logInfo;
			recTime = currTime;
		}

		if (tempProcessCount < m_nLoop)
		{
			(this->*m_arrOpsFns[m_ComboOperation.GetCurSel()][m_ComboTech.GetCurSel()])();
		}
		else
		{
			tempProcessCount = 0;
			QueryPerformanceCounter(&currTime);
			logInfo.Format(_T("完毕，线程数：%d，总耗时：%dms\r\n\r\n"), (int) wParam == -1? 1 : m_nThreadNum, GET_MS(startTime, currTime));

			CString procName;
			switch (lParam)
			{
			case WM_NOISE:
				procName.Format(_T("添加椒盐噪声"));
				break;
			case WM_MEDIAN_FILTER:
				procName.Format(_T("中值滤波"));
				break;
			case WM_ROTATE:
				procName.Format(_T("图片旋转"));
				break;
			case WM_SCALE:
				procName.Format(_T("图片缩放"));
				break;
			case WM_AUTO_LEVELS:
				procName.Format(_T("自动色阶"));
				break;
			case WM_AUTO_WHITE:
				procName.Format(_T("自动白平衡"));
				break;
			case WM_BILATERAL_FILTER:
				procName.Format(_T("双边滤波"));
				break;
			}
			m_strLogs += procName + logInfo;
			FinishProcess(lParam);
		}
	}

	return 0;
}
#pragma endregion
/******************************************************************************************************/

/********************************************** Controls **********************************************/
#pragma region Controls

void CExperimentImgProcDlg::OnBnClickedButtonOpen()
{
	// TODO: Add your control notification handler code here
	TCHAR szFilter[] = _T("JPEG (*.jpg)|*.jpg|BMP (*.bmp)|*.bmp|PNG (*.png)|*.png|TIFF (*.tif)|*.tif|All Files（*.*）|*.*||");
	CString filePath("");

	CFileDialog fileOpenDialog(TRUE, NULL, NULL, OFN_HIDEREADONLY, szFilter);
	if (fileOpenDialog.DoModal() == IDOK)
	{
		VERIFY(filePath = fileOpenDialog.GetPathName());
		m_strImgPath[m_nRadioPicture] = filePath;

		if (m_pImgSrc[m_nRadioPicture] != NULL)
		{
			m_pImgSrc[m_nRadioPicture]->Destroy();
			delete m_pImgSrc[m_nRadioPicture];
		}
		m_pImgSrc[m_nRadioPicture] = new CImage();
		m_pImgSrc[m_nRadioPicture]->Load(filePath);

		PrintPictureInfo(m_strImgPath[m_nRadioPicture], m_pImgSrc[m_nRadioPicture]->GetWidth(), m_pImgSrc[m_nRadioPicture]->GetHeight());

		if (m_ComboOperation.GetCurSel() == 1)
			m_ButtonProc.EnableWindow(CanUseMedianFilter());
		else
			m_ButtonProc.EnableWindow();

		ClearPicture();
		PrintPicture(m_pImgSrc[m_nRadioPicture], m_Picture[m_nRadioPicture]);
		m_bAddedNoise[m_nRadioPicture] = false;
		if(m_ComboOperation.GetCurSel() == MEDIANFILTER_INDEX)
			CanUseMedianFilter();
	}
}

void CExperimentImgProcDlg::OnBnClickedRadioPicture1()
{
	// TODO: Add your control notification handler code here
	m_nRadioPicture = 0;
	AfterChangeRadio();
}

void CExperimentImgProcDlg::OnBnClickedRadioPicture2()
{
	// TODO: Add your control notification handler code here
	m_nRadioPicture = 1;
	AfterChangeRadio();
}

void CExperimentImgProcDlg::OnBnClickedButtonMarge()
{
	// TODO: Add your control notification handler code here
	CString log;
	bool startMarge = false;
	if (m_pImgSrc[0] == NULL)
	{
		log = "左边图片不能为空";
	}
	else if (m_pImgSrc[1] == NULL)
	{
		log = "右边图片不能为空";
	}
	else if (m_pImgSrc[0]->GetWidth() != m_pImgSrc[1]->GetWidth()
		|| m_pImgSrc[0]->GetHeight() != m_pImgSrc[1]->GetHeight())
	{
		log = "两张图片大小不一致";
	}
	else
	{
		log = "开始进行图片融合";
		startMarge = true;
	}
	PrintLog(log);

	if (startMarge)
	{
		MargePictureDlg dlg(m_pImgSrc[0], m_pImgSrc[1], m_ComboTech.GetCurSel(), m_nThreadNum);
		dlg.DoModal();

	}
}

void CExperimentImgProcDlg::OnCbnSelchangeComboOperation()
{
	// TODO: Add your control notification handler code here
	//CString logInfo;

	switch (m_ComboOperation.GetCurSel())
	{
	case 1:
		CanUseMedianFilter();
		break;
	case 2:
	case 3:
		EnableItems();
		UseOneLoop();
		break;
	default:
		EnableItems();
		break;
	}
}

void CExperimentImgProcDlg::OnCbnSelchangeComboTech()
{
	// TODO: Add your control notification handler code here
}

void CExperimentImgProcDlg::OnNMCustomdrawSliderThread(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMCUSTOMDRAW pNMCD = reinterpret_cast<LPNMCUSTOMDRAW>(pNMHDR);
	// TODO: Add your control notification handler code here
	m_nThreadNum = m_SliderThreads.GetPos();
	UpdateData(FALSE);
	*pResult = 0;
}

void CExperimentImgProcDlg::OnEnChangeEditLoop()
{
	// TODO:  If this is a RICHEDIT control, the control will not
	// send this notification unless you override the CDialog::OnInitDialog()
	// function and call CRichEditCtrl().SetEventMask()
	// with the ENM_CHANGE flag ORed into the mask.

	// TODO:  Add your control notification handler code here
	UpdateData(TRUE);
}

void CExperimentImgProcDlg::OnBnClickedButtonProc()
{
	// TODO: Add your control notification handler code here
	if (CheckLoop() && CheckPicture())
	{
		/*EnableItems(false);
		m_bPictureRenderThreadSuspend = false;
		m_pPictureRenderThread->ResumeThread();
		ClearPicture();*/

		StartProcess();

		int ops_selected = m_ComboOperation.GetCurSel();
		int tech_selected = m_ComboTech.GetCurSel();
		CString logInfo;

		switch (ops_selected)
		{
		case ADDNOISE_INDEX:
			m_bUsedImgTemp = false;
			m_procFn = ImageProcess::addNoise;
			logInfo.Format(_T("添加椒盐噪声:\r\n"));
			break;
		case MEDIANFILTER_INDEX:
			m_bUsedImgTemp = false;
			m_procFn = ImageProcess::medianFilter;
			logInfo.Format(_T("中值滤波:\r\n"));
			break;
		case ROTATE_INDEX:
			m_bUsedImgTemp = true;
			m_procFn = ImageProcess::rotate;
			logInfo.Format(_T("图片旋转%d°:\r\n"), m_nDegree);
			break;
		case SCALE_INDEX:
			m_bUsedImgTemp = true;
			m_procFn = ImageProcess::scale;
			logInfo.Format(_T("图片缩放%.1f倍:\r\n"), m_fScaleRatio);
			break;
		case AUTOLEVELS_INDEX:
			m_bUsedImgTemp = false;
			m_procFn = ImageProcess::autoLevels;
			logInfo.Format(_T("进行自动色阶:\r\n"));
			break;
		case AUTOWHITE_INDEX:
			m_bUsedImgTemp = false;
			m_procFn = ImageProcess::autoWhite;
			logInfo.Format(_T("进行自动白平衡:\r\n"));
			break;
		case BILATERALFILTER_INDEX:
			m_bUsedImgTemp = false;
			m_procFn = ImageProcess::bilateralFilter;
			logInfo.Format(_T("进行双边滤波:\r\n"));
			break;
		}
		
		if (m_arrOpsFns[ops_selected][tech_selected] != NULL)
		{
			m_strLogs += logInfo;
			QueryPerformanceCounter(&startTime);
			recTime = startTime;
			ClearPicture();
			(this->*m_arrOpsFns[ops_selected][tech_selected])();
		}
		else
		{
			logInfo.Format(_T("暂不支持此操作\r\n"));
			m_strLogs += logInfo;
			/*EnableItems();
			m_bPictureRenderThreadSuspend = true;*/
			FinishProcess();
		}
	}
}

void CExperimentImgProcDlg::OnNMCustomdrawSliderRotate(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMCUSTOMDRAW pNMCD = reinterpret_cast<LPNMCUSTOMDRAW>(pNMHDR);
	// TODO: Add your control notification handler code here
	m_nDegree = m_SliderRotate.GetPos();
	m_strRotateAngle.Format(_T("%d°"), m_nDegree);
	UpdateData(FALSE);
	*pResult = 0;
}

void CExperimentImgProcDlg::OnNMCustomdrawSliderScale(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMCUSTOMDRAW pNMCD = reinterpret_cast<LPNMCUSTOMDRAW>(pNMHDR);
	// TODO: Add your control notification handler code here
	float scale = (float) m_SliderScale.GetPos();
	m_fScaleRatio = scale <= MAX_SCALE ? scale / (float) MAX_SCALE : scale - MAX_SCALE + 1;
	m_strScale.Format(_T("x%.1f"), m_fScaleRatio);
	UpdateData(FALSE);
	*pResult = 0;
}

BOOL CExperimentImgProcDlg::PreTranslateMessage(MSG* pMsg)
{
	// TODO: Add your specialized code here and/or call the base class
	if (WM_KEYDOWN == pMsg->message && VK_RETURN == pMsg->wParam)
	{
		if (GetFocus() == GetDlgItem(IDC_EDIT_LOOP))
		{
			UpdateData(TRUE);
			GetDlgItem(IDC_BUTTON_PROC)->SetFocus();
			UpdateData(FALSE);
		}

		return true;
	}
	return CDialog::PreTranslateMessage(pMsg);
}
#pragma endregion
/******************************************************************************************************/

/*********************************************** Utility **********************************************/
#pragma region Utility
UINT CExperimentImgProcDlg::PictureRender(LPVOID p)
{
	CExperimentImgProcDlg* dlg = (CExperimentImgProcDlg*) p;
	while (1)
	{
		Sleep(200);
		if (dlg->m_bUsedImgTemp)
		{
			dlg->PrintPicture(dlg->m_pImgTemp[dlg->m_nRadioPicture], dlg->m_Picture[dlg->m_nRadioPicture]);
		}
		else
		{
			dlg->PrintPicture(dlg->m_pImgSrc[dlg->m_nRadioPicture], dlg->m_Picture[dlg->m_nRadioPicture]);
		}

		if (dlg->m_CPictureRender.isSuspend)
		{
			dlg->m_CPictureRender.thread->SuspendThread();
		}
	}
	return 0;
}

UINT CExperimentImgProcDlg::LogPrinter(LPVOID p)
{
	CExperimentImgProcDlg* dlg = (CExperimentImgProcDlg*) p;
	CString LogRec;
	while (1)
	{
		if (LogRec != dlg->m_strLogs)
		{
			LogRec = dlg->m_strLogs;
			dlg->PrintLog(dlg->m_strLogs);
		}

		if (LogRec == dlg->m_strLogs && dlg->m_CLogPrinter.isSuspend)
		{
			dlg->m_strLogs.Format(_T(""));
			dlg->m_CLogPrinter.thread->SuspendThread();
		}
	}
	return 0;
}

void CExperimentImgProcDlg::PrintPicture(CImage * pImgSrc, CStatic & cPicPanel)
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

void CExperimentImgProcDlg::ClearPicture()
{
	CRect rect;
	m_Picture[m_nRadioPicture].GetClientRect(&rect);
	m_Picture[m_nRadioPicture].GetDC()->FillSolidRect(rect.left + 1, rect.top + 1, rect.Width() - 2, rect.Height() - 2, RGB(240, 240, 240));
}

void CExperimentImgProcDlg::PrintLog(CString & info)
{
	int count = info.GetLength();
	m_EditLog.SetRedraw(FALSE);
	m_EditLog.SetWindowText(info);

	int line = m_EditLog.GetLineCount();
	m_EditLog.LineScroll(line, 0);
	m_EditLog.SetSel(count, count);
	m_EditLog.SetRedraw(TRUE);
}

void CExperimentImgProcDlg::PrintPictureInfo(CString & path, int width, int height)
{
	((CEdit*) GetDlgItem(IDC_EDIT_PATH))->SetWindowTextW(path);
	CString size;
	if (width != 0 && height != 0)
	{
		size.Format(_T("%d x %d"), width, height);
	}
	((CEdit*) GetDlgItem(IDC_EDIT_PICTURE_SIZE))->SetWindowTextW(size);
}

bool CExperimentImgProcDlg::CheckLoop()
{
	if (m_nLoop > 100 || m_nLoop < 1)
	{
		CString alert;
		alert.Format(_T("请输入一个 1 到 100 之间的整数"));
		AfxMessageBox(alert);

		m_nLoop = 1;
		UpdateData(FALSE);
		GetDlgItem(IDC_EDIT_LOOP)->SetFocus();
		return false;
	}
	return true;
}

bool CExperimentImgProcDlg::CheckPicture()
{
	return m_pImgSrc[m_nRadioPicture] != NULL;
}

void CExperimentImgProcDlg::EnableItems(bool enable)
{
	m_ButtonOpen.EnableWindow(enable);
	m_ButtonProc.EnableWindow(enable);
	m_ComboOperation.EnableWindow(enable);
	m_ComboTech.EnableWindow(enable);
	m_SliderThreads.EnableWindow(enable);
	m_EditLoop.EnableWindow(enable);
	GetDlgItem(IDC_RADIO_PICTURE1)->EnableWindow(enable);
	GetDlgItem(IDC_RADIO_PICTURE2)->EnableWindow(enable);
	m_ButtonMarge.EnableWindow(enable);
	m_SliderRotate.EnableWindow(enable);
	m_SliderScale.EnableWindow(enable);
}

void CExperimentImgProcDlg::UseOneLoop()
{
	m_nLoop = 1;
	UpdateData(FALSE);
	m_EditLoop.EnableWindow(FALSE);
}

bool CExperimentImgProcDlg::CanUseMedianFilter()
{
	if (!CheckPicture())
		return false;
	if (!m_bAddedNoise[m_nRadioPicture])
	{
		m_ButtonProc.EnableWindow(FALSE);
		CString logInfo;
		logInfo = m_nRadioPicture == 0 ? "左边图片" : "右边图片";
		logInfo += "还没有加噪声\r\n";
		PrintLog(logInfo);
		m_EditLoop.EnableWindow(FALSE);
		return false;
	}
	else
	{
		UseOneLoop();
		m_ButtonProc.EnableWindow(TRUE);
		return true;
	}
}

void CExperimentImgProcDlg::AfterChangeRadio()
{
	m_ButtonProc.EnableWindow(CheckPicture());
	if (CheckPicture())
		PrintPictureInfo(m_strImgPath[m_nRadioPicture], m_pImgSrc[m_nRadioPicture]->GetWidth(), m_pImgSrc[m_nRadioPicture]->GetHeight());
	else
		PrintPictureInfo(m_strImgPath[m_nRadioPicture]);

	switch (m_ComboOperation.GetCurSel())
	{
	case 1:
		CanUseMedianFilter();
		break;
	case 2:
	case 3:
		UseOneLoop();
		break;
	default:
		break;
	}
}

void CExperimentImgProcDlg::CreateTempImg(bool isRotate)
{
	int width = m_pImgSrc[m_nRadioPicture]->GetWidth();
	int height = m_pImgSrc[m_nRadioPicture]->GetHeight();
	int bitCount = m_pImgSrc[m_nRadioPicture]->GetBPP();

	int w, h;
	if (isRotate)
	{
		double radian = m_nDegree * PI / 180;
		w = abs((int) (width * cos(radian))) + abs((int) (height * sin(radian)));
		h = abs((int) (width * sin(radian))) + abs((int) (height * cos(radian)));
	}
	else
	{
		w = width * m_fScaleRatio;
		h = height * m_fScaleRatio;
	}

	if (m_pImgTemp[m_nRadioPicture] != NULL)
	{
		m_pImgTemp[m_nRadioPicture]->Destroy();
	}
	m_pImgTemp[m_nRadioPicture] = new CImage();
	m_pImgTemp[m_nRadioPicture]->Create(w, h, bitCount);
}

void CExperimentImgProcDlg::StartProcess()
{
	EnableItems(false);
	ResumeThread(m_CPictureRender);
	ResumeThread(m_CLogPrinter);
}

void CExperimentImgProcDlg::FinishProcess(UINT msg)
{
	EnableItems();
	SuspendThread(m_CPictureRender);
	SuspendThread(m_CLogPrinter);

	switch (msg)
	{
	case WM_NOISE:
		m_bAddedNoise[m_nRadioPicture] = true;
		break;
	case WM_MEDIAN_FILTER:
		m_bAddedNoise[m_nRadioPicture] = false;
		m_ButtonProc.EnableWindow(FALSE);
	case WM_ROTATE:
	case WM_SCALE:
		m_EditLoop.EnableWindow(FALSE);
		break;
	case WM_AUTO_LEVELS:
	case WM_AUTO_WHITE:
		MultiThreadController::getInstance()->syncEnd();
		break;
	case WM_BILATERAL_FILTER:
		break;
	}
}

void CExperimentImgProcDlg::ResumeThread(Thread & thread)
{
	thread.isSuspend = false;
	thread.thread->ResumeThread();
}
void CExperimentImgProcDlg::SuspendThread(Thread & thread)
{
	thread.isSuspend = true;
}

#pragma endregion
/******************************************************************************************************/

