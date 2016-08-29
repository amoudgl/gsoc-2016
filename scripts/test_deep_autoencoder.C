#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TLegend.h"
#include "TGraph.h"
#include "TStyle.h"
#include "TH2.h"
#include "TText.h"
#include <TCanvas.h>

#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/DataLoader.h"
#include "TMVA/TMVAGui.h"

void test_dae_mydataset()
{
	TMVA::Tools::Instance();
	TFile *inputFile = TFile::Open( "../datasets/mydataset.root"); 
	TFile* outputFile = TFile::Open( "mydataset_output.root", "RECREATE" );
	TMVA::Factory *factory = new TMVA::Factory("TMVARegression", outputFile, 
	                                           "!V:!Silent:Color:DrawProgressBar:AnalysisType=Regression" );
	TMVA::DataLoader *loader=new TMVA::DataLoader("mydataset");
	loader->AddVariable("var0", 'F');
	loader->AddVariable("var1", 'F');
	loader->AddVariable("var2", 'F');
	loader->AddVariable("var3 := var0-var1", 'F');
	loader->AddVariable("var4 := var0*var2", 'F');
	loader->AddVariable("var5 := var1+var2", 'F');
	TTree *tsignal = (TTree*)inputFile->Get("MyMCSig");
	TTree *tbackground = (TTree*)inputFile->Get("MyMCBkg");
	TCut mycuts = "";
	TCut mycutb = "";
	loader->AddSignalTree( tsignal,     1.0 );
	loader->AddBackgroundTree( tbackground, 1.0 );
	loader->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=3000:nTrain_Background=3000:nTest_Signal=1449:nTest_Background=1449:SplitMode=Random:NormMode=NumEvents:!V");

	TString layoutString ("Layout=TANH|3,LINEAR");
	TString training0 ("LearningRate=1e-1,Momentum=0.0,Repetitions=1,ConvergenceSteps=300,BatchSize=20,TestRepetitions=15,WeightDecay=0.001,Regularization=NONE,DropConfig=0.0+0.5+0.5+0.5,DropRepetitions=1,Multithreading=True");
	TString training1 ("LearningRate=1e-2,Momentum=0.5,Repetitions=1,ConvergenceSteps=300,BatchSize=30,TestRepetitions=7,WeightDecay=0.001,Regularization=L2,Multithreading=True,DropConfig=0.0+0.1+0.1+0.1,DropRepetitions=1");
	TString trainingStrategyString ("TrainingStrategy=");
	trainingStrategyString += training0 + "|" + training1;
	TString nnOptions ("AE(indexLayer=1;pretraining=false;!H:V:ErrorStrategy=SUMOFSQUARES:VarTransform=G:WeightInitialization=XAVIERUNIFORM");
	nnOptions.Append (":");
	nnOptions.Append (layoutString);
	nnOptions.Append (":");
	nnOptions.Append (trainingStrategyString);
	nnOptions.Append (")");
	cout << nnOptions.Data() << endl;
	TMVA::DataLoader* newloader = loader->VarTransform(nnOptions);
}

void test_deep_autoencoder(){
	test_dae_mydataset();
}



