
#include <iostream>
#include <tclap/CmdLine.h>

#include "phraseModel.h"
#include "param.h"

using namespace std;
using namespace TCLAP;

void printCommandLine(int argc, char** argv)
{
	// print command line // 
	cout << "Command line: " << endl;
	for (int i = 0; i < argc; i++)
	{
		cout << argv[i] << endl;
	}
	cout << endl;
}

int main(int argc, char** argv)
{

	try
	{
		// program options //
		CmdLine cmd("Command description message ", ' ' , "1.1");

		ValueArg<string> trainingFile("f", "trainingFile", "Training filename -- alignment file" , false, "", "string", cmd);
		ValueArg<string> devFile("d", "devFile", "Development filename -- dev set", false, "", "string", cmd);
		ValueArg<string> testingFile("t", "testingFile", "Testing filename -- testing file", false, "", "string", cmd);
		ValueArg<string> answerFile("a", "answer", "Answer output filename -- answer file", false, "", "string", cmd);

		ValueArg<string> modelOutFilename("","mo","Model output filename", false, "", "string", cmd);
		ValueArg<string> modelInFilename("","mi","Model filename for testing", false, "", "string", cmd);
		
		ValueArg<string> LMInFilename("", "lm", "Filename of language model; file should be in arpa format", false, "", "string", cmd);

		ValueArg<string> WCInFilename("", "wc", "Filename of word list; should contain word and count, separated by tab", false, "", "string", cmd);
		ValueArg<int> nBest("","nBest","n-best size for training (default 10)", false, 10, "int", cmd);
		ValueArg<int> contextSize("","cs","Context size (default 5)", false, 5, "int", cmd);
		ValueArg<int> nGram("","ng","n-gram size (default 11)", false, 11, "int", cmd);

		ValueArg<int> markovOrder("","order","Markov order (default 0)", false, 0, "int", cmd);
		SwitchArg linearChain("","linearChain","Linear chain features (default false)", cmd, false);
		SwitchArg copyFeature("","copy","Copy feature (default false)", cmd, false);
		SwitchArg ignoreNull("", "igNull", "Ignore null in markov features (default false)", cmd, false);
		SwitchArg finnishVH("", "FinVH", "Determine if all vowels are front or back", cmd, false);
		SwitchArg turkishVH("", "TurkVH", "Determine if suffix vowels match frontness of final vowel of stem", cmd, false);
		SwitchArg turkishRH("", "TurkRH", "Determine if high front vowels match rounding of previous vowel", cmd, false);
		
		ValueArg<int> trainAtLeast("","tal", "Train at least n iteration (default 1)", false, 1, "int", cmd);
		ValueArg<int> trainAtMost("","tam", "Train at most n iteration (default 99)" , false, 99, "int", cmd);

		ValueArg<int> nBestTest("","nBestTest","Output n-best answers (default 1)", false ,1 , "int", cmd);
		
		ValueArg<string> inChar("","inChar","Token delimeter string (default null)", false, "", "string", cmd);
		ValueArg<string> outChar("","outChar","Token delimeter output (default null)", false, "", "string", cmd);

		SwitchArg keepModel("","keepModel","Keep all trained model (default false)", cmd, false);

		vector<string> allowAlignLossFn;
		allowAlignLossFn.push_back("minL"); // loss  = min L; x,y' = the instance of min L;  (dynamic depended on Loss)
		allowAlignLossFn.push_back("maxL"); // loss  = max L; x,y' = the instance of max L;  (dynamic depended on Loss)
		allowAlignLossFn.push_back("avgL"); // loss  = avg all Loss; x,y' = avg all instance of x,y' (equally)  (static)
		allowAlignLossFn.push_back("ascL"); // loss  = avg all Loss; x,y' = avg all instance of x,y' (by alignment scores) (static)
		allowAlignLossFn.push_back("rakL"); // loss  = avg all Loss; x,y' = avg all instance of x,y' (by alignment ranks) (static)
		allowAlignLossFn.push_back("minS"); // loss  = Loss of Min score ; x,y' = the instance of min score (dynamic depended on score)
		allowAlignLossFn.push_back("maxS"); // loss  = Loss of Max score ; x,y' = the instance of max score (dynamic depended on score)
		allowAlignLossFn.push_back("mulA"); // put all multiple alignment constraints to train the model *this assumes all y are the same

		ValuesConstraint<string> allowAlignLoss(allowAlignLossFn);

		ValueArg<string> alignLoss("","alignLoss","Multiple-alignments loss computation criteria [minL, maxL, avgL, ascL, rakL, minS, maxS] (default minL)", false, "minL", &allowAlignLoss, cmd);

		ValueArg<double> SVMcPara("","SVMc", "SVM c parameter (default 9999999)", false, 9999999, "double", cmd);
		
		SwitchArg noContextFea("","noContextFea","Do not use context n-gram features (default false)", cmd, false);
		ValueArg<int> jointMgram("","jointMgram","Use joint M-gram features (default M=0)", false, 0, "int", cmd);

		ValueArg<int> beamSize("","beamSize","Beam size (default 20)", false, 20, "int", cmd);
		SwitchArg useBeam("","beam","Use Beam search instead of Viterbi search (default false)", cmd, false);

		ValueArg<int> jointFMgram("","jointFMgram", "Use joint forward M-gram features (default FM=0)", false, 0, "int", cmd);

		ValueArg<string> extFeaTrain("", "extFeaTrain","Extra feature for training file (default null)", false, "", "string", cmd);
		ValueArg<string> extFeaDev("", "extFeaDev","Extra feature for dev file (default null)", false, "", "string", cmd);
		ValueArg<string> extFeaTest("", "extFeaTest","Extra feature for testing file (default null)", false, "", "string", cmd);


		// parse options
		cmd.parse(argc, argv);

		// print command line to cout//
		printCommandLine(argc,argv);

		// print all program options //
		list<Arg*> args = cmd.getArgList();
		for (ArgListIterator it = args.begin(); it != args.end(); it++)
		{
			cout << (*it)->toString() << ": " << (*it)->getDescription() << endl;
		}

		// get option values to myParam //
		param myParam;

		myParam.trainingFile = trainingFile.getValue();
		myParam.devFile = devFile.getValue();
		myParam.testingFile = testingFile.getValue();

		myParam.nBest = nBest.getValue();
		//myParam.maxX = maxX.getValue();
		myParam.maxX = 1; // default size //
		myParam.contextSize = contextSize.getValue();
		myParam.nGram = nGram.getValue();
	
		myParam.copyFeature = copyFeature.getValue();		
		myParam.ignoreNull = ignoreNull.getValue();
		myParam.finnishVH = finnishVH.getValue();
		myParam.turkishVH = turkishVH.getValue();
		myParam.turkishRH = turkishRH.getValue();

		//verify context size and n-gram //
		/*if ( ((myParam.contextSize * 2) + 1) != myParam.nGram )
		{
			cout << "n-gram features has to = (context size * 2) + 1" << endl;
			myParam.nGram = (myParam.contextSize * 2) + 1;
			cout << "re-defined n-gram value : " << myParam.nGram << endl;
		}*/

		myParam.markovOrder = markovOrder.getValue();
		myParam.linearChain = linearChain.getValue();

		myParam.trainAtLeast = trainAtLeast.getValue();
		myParam.trainAtMost = trainAtMost.getValue();
		
		myParam.nBestTest = nBestTest.getValue();

		
		myParam.modelInFilename = modelInFilename.getValue();
		myParam.answerFile = answerFile.getValue();
		
		myParam.LMInFilename = LMInFilename.getValue();
		myParam.WCInFilename = WCInFilename.getValue();
		//myParam.LMSize = 0; //Default no LM; will change when LM is read


		myParam.inChar = inChar.getValue();
		cout << myParam.inChar << endl;
		myParam.outChar = outChar.getValue();

		myParam.keepModel = keepModel.getValue();

		myParam.alignLoss = alignLoss.getValue();

		myParam.SVMcPara = SVMcPara.getValue();

		myParam.noContextFea = noContextFea.getValue();
		myParam.jointMgram = jointMgram.getValue();

		myParam.beamSize = beamSize.getValue();
		myParam.useBeam = useBeam.getValue();

		myParam.jointFMgram = jointFMgram.getValue();

		// if jointMgram/jointFMgram is used, use beam //
		if(myParam.copyFeature)
		{
			cout << "Using copy feature" << endl;
		}
		if(myParam.ignoreNull)
		{
			cout << "Ignoring null on markov features" << endl;
		}
		if (((myParam.jointMgram > 0) || (myParam.jointFMgram > 0)) && ( ! myParam.useBeam))
		{
			cout << "Use joint M-gram features M = " << myParam.jointMgram << endl;
			cout << "Use joint FM-gram features FM = " << myParam.jointFMgram << endl;
			cout << "Forcing to use beam search .... " << endl;
			myParam.useBeam = true;
			cout << "useBeam = " << myParam.useBeam << endl;
		}


		myParam.modelOutFilename = modelOutFilename.getValue();
		if (myParam.modelOutFilename == "")
		{
			myParam.modelOutFilename = trainingFile.getValue();
			myParam.modelOutFilename += "." + stringify(myParam.nBest) + "nBest";
			//myParam.modelOutFilename += "." + stringify(myParam.contextSize) + "cs";
			//myParam.modelOutFilename += "." + stringify(myParam.nGram) + "ng";

			if (myParam.ignoreNull)
			{
				myParam.modelOutFilename += ".ignoreNull";
			}
			if (myParam.LMInFilename != "")
			{
				myParam.modelOutFilename += ".LM";
			}
			if (myParam.WCInFilename != "")
			{
				myParam.modelOutFilename += ".WC";
			}
			if (myParam.finnishVH || myParam.turkishVH)
			{
				myParam.modelOutFilename += ".VH";
			}
			if (myParam.turkishRH)
			{
				myParam.modelOutFilename += ".RH";
			}
			if (myParam.markovOrder > 0)
			{
				myParam.modelOutFilename += "." + stringify(myParam.markovOrder) + "order";
			}

			if (myParam.linearChain)
			{
				myParam.modelOutFilename += ".linearChain";
			}
		}

		myParam.extraFeaTrain = extFeaTrain.getValue();
		myParam.extraFeaDev = extFeaDev.getValue();
		myParam.extraFeaTest = extFeaTest.getValue();

		// main program

		phraseModel myModel;

		// training 
		if (myParam.trainingFile != "")
		{
			myModel.training(myParam);
		}

		// testing 
		//if ( (myParam.testingFile != "") && (myParam.modelInFilename != "") )
		if (myParam.modelInFilename != "")
		{
			myModel.testing(myParam);
		}// testing by getting string input from stdin



	}
	catch (TCLAP::ArgException &e)
	{
		cerr << "error: " << e.error() <<  " for arg " << e.argId() << endl;
	}

	return 0;
}
