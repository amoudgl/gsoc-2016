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

void plot_variance_histogram(TMVA::DataLoader* loader1, Double_t threshold)
{
    TCanvas *c1 = new TCanvas("c1","different scales hists",600,400);
    std::vector<TMVA::VariableInfo>& vars = loader1->GetDataSetInfo().GetVariableInfos();
    Int_t nvars = loader1->GetDataSetInfo().GetNVariables();
    TH1F *h1 = new TH1F("h1t","Variance Histogram", nvars, 0, nvars);
    TH1F *h2 = new TH1F("h2","Variance Histogram", nvars, 0, nvars);
    leg = new TLegend(.75,.80,1,1);
    leg->AddEntry(h1,"Selected Variables","f");
    leg->AddEntry(h2,"Rejected Variables","f");

    h1->SetFillColor(8);
    h1->SetStats(0);
    h1->SetLineColor(1);
    h2->SetStats(0);
    h2->SetFillColor(2);
    h2->SetLineColor(1);

    TAxis* xAxis = h1->GetXaxis();
    TAxis* yAxis= h1->GetYaxis();

    xAxis->SetTitle("Variables");
    xAxis->SetTitleColor(4);
    xAxis->SetTitleSize(0.05);
    xAxis->SetLabelSize(0.047);
    yAxis->SetTitle("Variance");
    yAxis->SetTitleColor(4);
    yAxis->SetTitleSize(0.05);
    yAxis->SetLabelSize(0.047);

    for (Int_t i = 1; i<=nvars; i++)
    {
        if (vars[i - 1].GetVariance() > threshold)
            h1->SetBinContent(i, vars[i - 1].GetVariance());
        else
            h2->SetBinContent(i, vars[i - 1].GetVariance());
        xAxis->SetBinLabel(i, vars[i - 1].GetExpression());
    }
    h1->Draw();
    h2->Draw("same");
    leg->Draw();
    c1->Draw();
}

void test_vt_higgs()
{
    TMVA::Tools::Instance();
    TFile *input = TFile::Open( "../../datasets/higgs-dataset.root"); 
    TFile* outputFile = TFile::Open( "mydataset_output.root", "RECREATE" );
    TMVA::Factory *factory = new TMVA::Factory("TMVAClassification", outputFile, 
                                           "!V:ROC:!Correlations:!Silent:Color:!DrawProgressBar:AnalysisType=Classification" );
    TMVA::DataLoader *loader1=new TMVA::DataLoader("higgs-dataset");
    TTree *Tsignal     = (TTree*)input->Get("TreeS");
    TTree *Tbackground = (TTree*)input->Get("TreeB");
    TString outfileName( "higgs_output.root" );
    TFile* OutputFile = TFile::Open( outfileName, "RECREATE" );

    loader1->AddVariable("lepton_pT",'F');
    loader1->AddVariable("lepton_eta",'F');
    loader1->AddVariable("lepton_phi",'F');
    loader1->AddVariable("missing_energy_magnitude",'F');
    loader1->AddVariable("missing_energy_phi",'F');
    loader1->AddVariable("jet_1_pt",'F');
    loader1->AddVariable("jet_1_eta",'F');
    loader1->AddVariable("jet_1_phi",'F');
    loader1->AddVariable("jet_1_b_tag",'F');
    loader1->AddVariable("jet_2_pt",'F');
    loader1->AddVariable("jet_2_eta",'F');
    loader1->AddVariable("jet_2_phi",'F');
    loader1->AddVariable("jet_2_b_tag",'F');
    loader1->AddVariable("jet_3_pt",'F');
    loader1->AddVariable("jet_3_eta",'F');
    loader1->AddVariable("jet_3_phi",'F');
    loader1->AddVariable("jet_3_b_tag",'F');
    loader1->AddVariable("jet_4_pt",'F');
    loader1->AddVariable("jet_4_eta",'F');
    loader1->AddVariable("jet_4_phi",'F');
    loader1->AddVariable("jet_4_b_tag",'F');
    loader1->AddVariable("m_jj",'F');
    loader1->AddVariable("m_jjj",'F');
    loader1->AddVariable("m_lv",'F');
    loader1->AddVariable("m_jlv",'F');
    loader1->AddVariable("m_bb",'F');
    loader1->AddVariable("m_wbb",'F');
    loader1->AddVariable("m_wwbb",'F');

    Double_t signalWeight     = 1.0;
    Double_t backgroundWeight = 1.0;
    loader1->AddSignalTree    ( Tsignal,     signalWeight     );
    loader1->AddBackgroundTree( Tbackground, backgroundWeight );
    TCut myCuts = ""; 
    TCut myCutb = "";
    loader1->PrepareTrainingAndTestTree( myCuts, myCutb,
                                             "nTrain_Signal=1829123:nTrain_Background=1170877:nTest_Signal=4000000:nTest_Background=4000000:SplitMode=Random:NormMode=NumEvents:!V" );

    TMVA::DataLoader* loader2 = loader1->VarTransform("VT(1.0)");

    // plot variance histogram
    Double_t threshold = 1.0;
    plot_variance_histogram(loader1, threshold);

    // plot importance histogram
    auto importanceHisto = factory->EvaluateImportance(loader1, TMVA::VIType::kAll, TMVA::Types::kBDT, "BDT",
    "!V:NTrees=5:MinNodeSize=2.5%:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );
    TCanvas *canvas=new TCanvas("Importance");
    importanceHisto->Draw();
    canvas->Draw();

    // Boosted Decision Trees
    factory->BookMethod(loader1,TMVA::Types::kBDT, "BDT","!V:NTrees=200:MinNodeSize=2.5%:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );

    //Multi-Layer Perceptron (Neural Network)
    factory->BookMethod(loader1, TMVA::Types::kMLP, "MLP","!H:!V:NeuronType=tanh:VarTransform=N:NCycles=100:HiddenLayers=N+5:TestRate=5:!UseRegulator" );
    //Support Vector Machine
    factory->BookMethod(loader1, TMVA::Types::kSVM, "SVM", "Gamma=0.25:Tol=0.001" );

    // DNN 
    TString layoutString ("Layout=TANH|50,TANH|10,LINEAR");
    TString training0 ("LearningRate=1e-1,Momentum=0.0,Repetitions=1,ConvergenceSteps=300,BatchSize=20,TestRepetitions=15,WeightDecay=0.001,Regularization=NONE,DropConfig=0.0+0.5+0.5+0.5,DropRepetitions=1,Multithreading=True");
    TString training1 ("LearningRate=1e-2,Momentum=0.5,Repetitions=1,ConvergenceSteps=300,BatchSize=30,TestRepetitions=7,WeightDecay=0.001,Regularization=L2,Multithreading=True,DropConfig=0.0+0.1+0.1+0.1,DropRepetitions=1");
    TString trainingStrategyString ("TrainingStrategy=");
    trainingStrategyString += training0 + "|" + training1;
    TString nnOptions ("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=G:WeightInitialization=XAVIERUNIFORM");
    nnOptions.Append (":");
    nnOptions.Append (layoutString);
    nnOptions.Append (":");
    nnOptions.Append (trainingStrategyString);
    factory->BookMethod(loader1, TMVA::Types::kDNN, "DNN", nnOptions );

    // Book Methods for transformed dataloader
    factory->BookMethod(loader2,TMVA::Types::kBDT, "BDT",
                       "!V:NTrees=200:MinNodeSize=2.5%:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );

    factory->BookMethod(loader2, TMVA::Types::kMLP, "MLP",
                       "!H:!V:NeuronType=tanh:VarTransform=N:NCycles=100:HiddenLayers=N+5:TestRate=5:!UseRegulator" );

    factory->BookMethod(loader2, TMVA::Types::kSVM, "SVM", "Gamma=0.25:Tol=0.001" );
    factory->BookMethod(loader2, TMVA::Types::kDNN, "DNN", nnOptions );
    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();
    auto c1 = factory->VarTransformROCPlot(loader1, loader2);
    c1->Draw();    
}

void test_vt_mydataset( )
{
    TMVA::Tools::Instance();
    TFile *inputFile = TFile::Open( "../datasets/mydataset.root"); 
    TFile* outputFile = TFile::Open( "mydataset_output.root", "RECREATE" );
    TMVA::Factory *factory = new TMVA::Factory("TMVAClassification", outputFile, 
                                           "!V:ROC:!Correlations:!Silent:Color:!DrawProgressBar:AnalysisType=Classification" );
    TMVA::DataLoader *loader1=new TMVA::DataLoader("mydataset");
    loader1->AddVariable("var0", 'F');
    loader1->AddVariable("var1", 'F');
    loader1->AddVariable("var2", 'F');
    loader1->AddVariable("var3 := var0-var1", 'F');
    loader1->AddVariable("var4 := var0*var2", 'F');
    loader1->AddVariable("var5 := var1+var2", 'F');
    TTree *tsignal = (TTree*)inputFile->Get("MyMCSig");
    TTree *tbackground = (TTree*)inputFile->Get("MyMCBkg");
    TCut mycuts = "";
    TCut mycutb = "";
    loader1->AddSignalTree( tsignal,     1.0 );
    loader1->AddBackgroundTree( tbackground, 1.0 );
    loader1->PrepareTrainingAndTestTree( mycuts, mycutb,"nTrain_Signal=3000:nTrain_Background=3000:nTest_Signal=1449:nTest_Background=1449:SplitMode=Random:NormMode=NumEvents:!V");
    TMVA::DataLoader* loader2 = loader1->VarTransform("VT(2.95)");
    // Double_t threshold = 2.95;
    // plot_variance_histogram(loader1, threshold);
    // Boosted Decision Trees
    factory->BookMethod(loader1,TMVA::Types::kBDT, "BDT","!V:NTrees=200:MinNodeSize=2.5%:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );

    //Multi-Layer Perceptron (Neural Network)
    factory->BookMethod(loader1, TMVA::Types::kMLP, "MLP","!H:!V:NeuronType=tanh:VarTransform=N:NCycles=100:HiddenLayers=N+5:TestRate=5:!UseRegulator" );
    //Support Vector Machine
    factory->BookMethod(loader1, TMVA::Types::kSVM, "SVM", "Gamma=0.25:Tol=0.001" );

    // DNN 
    TString layoutString ("Layout=TANH|50,TANH|10,LINEAR");
    TString training0 ("LearningRate=1e-1,Momentum=0.0,Repetitions=1,ConvergenceSteps=300,BatchSize=20,TestRepetitions=15,WeightDecay=0.001,Regularization=NONE,DropConfig=0.0+0.5+0.5+0.5,DropRepetitions=1,Multithreading=True");
    TString training1 ("LearningRate=1e-2,Momentum=0.5,Repetitions=1,ConvergenceSteps=300,BatchSize=30,TestRepetitions=7,WeightDecay=0.001,Regularization=L2,Multithreading=True,DropConfig=0.0+0.1+0.1+0.1,DropRepetitions=1");
    TString trainingStrategyString ("TrainingStrategy=");
    trainingStrategyString += training0 + "|" + training1;
    TString nnOptions ("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=G:WeightInitialization=XAVIERUNIFORM");
    nnOptions.Append (":");
    nnOptions.Append (layoutString);
    nnOptions.Append (":");
    nnOptions.Append (trainingStrategyString);
    factory->BookMethod(loader1, TMVA::Types::kDNN, "DNN", nnOptions );

    // Book Methods for transformed dataloader
    factory->BookMethod(loader2,TMVA::Types::kBDT, "BDT",
                       "!V:NTrees=200:MinNodeSize=2.5%:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );

    factory->BookMethod(loader2, TMVA::Types::kMLP, "MLP",
                       "!H:!V:NeuronType=tanh:VarTransform=N:NCycles=100:HiddenLayers=N+5:TestRate=5:!UseRegulator" );

    factory->BookMethod(loader2, TMVA::Types::kSVM, "SVM", "Gamma=0.25:Tol=0.001" );
    factory->BookMethod(loader2, TMVA::Types::kDNN, "DNN", nnOptions );
    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();
    auto c1 = factory->VarTransformROCPlot(loader1, loader2);
    c1->Draw(); 
}

void test_variance_threshold()
{
    test_vt_higgs();
    // test_vt_mydataset();
}