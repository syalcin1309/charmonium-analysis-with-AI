#include <TROOT.h>
#include <TCanvas.h>
#include <RooFit.h>
#include <RooRealVar.h>
#include <RooBreitWigner.h>
#include <RooCBShape.h>
#include <RooFFTConvPdf.h>
#include <RooPlot.h>
#include <RooDataHist.h>
#include <TAttLine.h>
#include <RooExponential.h>
//#include "RooChi2Var.h"
#include <RooAddPdf.h>
#include < RooPolynomial.h>
#include < RooCurve.h>

void fit(){
  
  gROOT->SetStyle ("Plain");
  gSystem->Load("libRooFit") ;
  gStyle->SetFitFormat("g"); 
  using namespace RooFit ;
  cout << "loaded" << endl;
  Double_t Mass[]= {};

  TH1F* h1= new TH1F("h1", "h1 title",30,2.7969, 3.3969);
  h1->Sumw2();

  std::cout<<"hello1"<<std::endl;
  
  Double_t *mass_rounded[3681];
  
  for (Int_t i = 0; i < 3681 ; i++) {
     h1->Fill(Mass[i]);
  }
  std::cout<<"hello2"<<std::endl;

  
  double hmin = h1->GetXaxis()->GetXmin();
  double hmax = h1->GetXaxis()->GetXmax();

  std::cout<<"hello3"<<std::endl;

  TH1F *h3 = (TH1F*) h1->Clone();
 
  
  // Declare observable x
  RooRealVar x("x","MM_{(#pi #pi)} GeV/c^{2}",2.7969, 3.3969) ;
  
  RooDataHist dh("dh","dh",x,Import(*h1));

  RooPlot* frame = x.frame(Title("J/$\psi$ mass")) ;
  frame->SetTitleSize(0.001);
  dh.plotOn(frame,MarkerColor(1),MarkerSize(3.0),MarkerStyle(20),LineWidth(1)); //it was 1.85 markersize
  
  // Crystal-Ball
  
  RooRealVar mean( "mean", "mean",3.09, 3.08,3.1);//087 3.0969

  
  RooRealVar sigma( "sigma", "sigma",0.028,0.025,0.03);//2.3// ,60.0, 120.0);
  RooRealVar alpha( "alpha", "alpha",1.35,1.32,1.39);
  RooRealVar n( "n", "n", 1.0,0.9,1.1);//0.81
  RooCBShape cb( "cb", "cb", x, mean, sigma, alpha, n );

  
  //Gaussian 
  RooRealVar mean1("#mu","mean of gaussians",3.08,3.0,3.2);
  RooRealVar sigma1("#sigma","width of gaussians",0.03);
  RooGaussian sig1("sig1", "sig1", x, mean1, sigma1);

  //expo
  RooRealVar lambda("lambda", "slope", 0.95,0.8,1);
  // RooRealVar lambda2("lambda2", "slope2", 0.54,-1,1.5);
  RooExponential expo("expo", "exponential PDF", x, lambda);

  //polynomial
  RooRealVar a0( "a0", "a0", 0.01 );//0.0099 //0.01
  RooRealVar a1( "a1", "a1", 0.0012);//0.0016
  RooPolynomial poly("poly","poly",x,RooArgList(a0,a1));
  

  ///
  // convolution
  RooFFTConvPdf pdf( "pdf", "pdf", x, cb, sig1);

  RooFitResult* filters = cb.fitTo(dh,"qr",Range(2.79,3.48),PrintLevel(-1));

  cb.plotOn(frame,LineColor(4),LineWidth(2.0));//this will show fit overlay on canvas

  cb.paramOn(frame,Format("NELU",AutoPrecision(5)));


  TCanvas* c = new TCanvas("ZmassHisto","ZmassHisto",600,600) ;


  Double_t Mass_hist[]= {};


  Double_t *mass_hist_rounded[];


  ////

  TH1F *h2= new TH1F("h2", "h2 title", 30, 60.0, 120.0);
  h2->Sumw2();



  ///
  
 
  for (Int_t i = 0; i <  ; i++) {

    h2->Fill(Mass_hist[i]);
  }

  double hmin2 = h2->GetXaxis()->GetXmin();
  double hmax2 = h2->GetXaxis()->GetXmax();


  
  
  // Declare observable x
  RooRealVar x2("x2","x2",hmin2,hmax2) ;


  x2.setBins(10000,"cache") ;
  x2.setMin("cache",55.5) ;
  x2.setMax("cache",125.5) ;
  
  
  RooDataHist dh2("dh2","dh2",x2,Import(*h2));

  c->cd();
 
   
  h2->SetFillStyle(1001);//1001
  h2->SetFillColorAlpha(626,0.8);

  h2->SetLineColorAlpha(630,0.8);
  h2->SetLineWidth(1);

  h2->GetXaxis()->SetTitle("m_{#mu^{+}#mu^{-}} (GeV/c^{2})");
  h2->GetXaxis()->SetTitleSize(0.049);
  
  h2->GetYaxis()->SetTitle("Events / ( 2 GeV/c^{2})");
  h2->GetYaxis()->SetTitleSize(0.049);
  
  h2->GetXaxis()->SetTitleOffset(0.89);
  h2->GetYaxis()->SetTitleOffset(0.68);

  h2->GetXaxis()->SetTitleFont(42);
  h2->GetYaxis()->SetTitleFont(42);
  

  h2->GetXaxis()->SetLabelOffset(0.009);
  h2->GetYaxis()->SetLabelOffset(0.006);

  h2->GetXaxis()->SetLabelSize(0.038);
  h2->GetYaxis()->SetLabelSize(0.038);

  h2->GetXaxis()->SetLabelFont(42);
  h2->GetYaxis()->SetLabelFont(42);

  h2->GetYaxis()->SetRangeUser(0,810);


  h2->Draw("hist same");






  
  c->cd() ;
  gPad->SetLeftMargin(0.15);
  gStyle->SetOptStat();
  gStyle->SetOptFit(1111);

  frame->GetXaxis()->SetTitle("m_{#mu^{+}#mu^{-}} (GeV/c^{2})");
  frame->GetXaxis()->SetTitleSize(0.049);
  
  frame->GetYaxis()->SetTitle("Events / ( 0.02 GeV/c^{2})");
  frame->GetYaxis()->SetTitleSize(0.049);
  
  frame->GetXaxis()->SetTitleOffset(0.89);
  frame->GetYaxis()->SetTitleOffset(0.68);

  frame->GetXaxis()->SetTitleFont(42);
  frame->GetYaxis()->SetTitleFont(42);
  

  frame->GetXaxis()->SetLabelOffset(0.009);
  frame->GetYaxis()->SetLabelOffset(0.006);

  frame->GetXaxis()->SetLabelSize(0.038);
  frame->GetYaxis()->SetLabelSize(0.038);

  frame->GetXaxis()->SetLabelFont(42);
  frame->GetYaxis()->SetLabelFont(42);

  
  
  float binsize = h1->GetBinWidth(1);
  char Bsize[30]; 
  

  

  Int_t npar = cb->getParameters(dh)->selectByAttrib("Constant",kFALSE)->getSize();
  Double_t chi2ndf = frame->chiSquare(npar);
  std::cout<<"Chi Square/ndf=:"<<chi2ndf/5<<std::endl;

  std::cout<<"ndf=:"<<npar<<std::endl;

  TPaveText *pt = new TPaveText(0.57,0.7,0.75,0.89,"BRNDC");
  pt->SetName("statBox");
  pt->SetBorderSize(0);
  pt->SetFillColor(0);
  pt->SetTextAlign(12);
  pt->SetTextSize(0.05);
  pt->SetTextFont(42);

  frame->Draw();


  h1->SetMarkerStyle(20);
  h1->SetMarkerSize(3);
 
  
  TH1F *h3 = (TH1F*) h1->Clone();
  TLegend *leg1=new TLegend(0.68,0.37,0.83,0.60) ;
  leg1->SetBorderSize(0);
  leg1->SetTextFont(42);
  leg1->SetTextAlign(12);
  leg1->SetTextSize(0.04);
  h3->SetLineColor(4);
  h3->SetLineWidth(2);
  
}


