from rootpy.io import root_open
from ROOT import TMVA

factory = TMVA.Factory("TMVAClassification", outfile, "AnalysisType=Classification")
factory.AddVariable("a", 'F')
factory.AddVariable("b", 'F')
factory.SetInputTrees(infile.sample, Cut('label==1'), Cut('label==0'))
factory.PrepareTrainingAndTestTree(Cut(), Cut(), "SplitMode=Random:NormMode=NumEvents")
factory.BookMethod(TMVA.Types.kBDT, "BDT", "NTrees=850:nEventsMin=150:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=-1")
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
outfile.close()
infile.close()
