#include "phraseModel.h"
#include "utf8.h"
//------- SVM default parameters -------------------//

void set_default_parameters(LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm)
{
	//strcpy (modelfile, "svm_model");
	strcpy (learn_parm->predfile, "trans_predictions");
	strcpy (learn_parm->alphafile, "");
	//strcpy (restartfile, "");

	learn_parm->biased_hyperplane=1;
	learn_parm->sharedslack=0;
	learn_parm->remove_inconsistent=0;
	learn_parm->skip_final_opt_check=0;
	learn_parm->svm_maxqpsize=10;
	learn_parm->svm_newvarsinqp=0;
	learn_parm->svm_iter_to_shrink=-9999;
	learn_parm->maxiter=100000;
	learn_parm->kernel_cache_size=40;
	learn_parm->svm_c=9999999;
	learn_parm->eps=0.1;
	learn_parm->transduction_posratio=-1.0;
	learn_parm->svm_costratio=1.0;
	learn_parm->svm_costratio_unlab=1.0;
	learn_parm->svm_unlabbound=1E-5;
	learn_parm->epsilon_crit=0.001;
	learn_parm->epsilon_a=1E-15;
	learn_parm->compute_loo=0;
	learn_parm->rho=1.0;
	learn_parm->xa_depth=0;
	kernel_parm->kernel_type=0;
	kernel_parm->poly_degree=3;
	kernel_parm->rbf_gamma=1.0;
	kernel_parm->coef_lin=1;
	kernel_parm->coef_const=1;
	strcpy(kernel_parm->custom,"empty");

	if(learn_parm->svm_iter_to_shrink == -9999) {
    if(kernel_parm->kernel_type == LINEAR) 
      learn_parm->svm_iter_to_shrink=2;
    else
      learn_parm->svm_iter_to_shrink=100;
  }
	if((learn_parm->skip_final_opt_check) 
     && (kernel_parm->kernel_type == LINEAR)) {
    printf("\nIt does not make sense to skip the final optimality check for linear kernels.\n\n");
    learn_parm->skip_final_opt_check=0;
  }    
  if((learn_parm->skip_final_opt_check) 
     && (learn_parm->remove_inconsistent)) {
    printf("\nIt is necessary to do the final optimality check when removing inconsistent \nexamples.\n");
  //  wait_any_key();
  //  print_help();
    exit(0);
  }    
  if((learn_parm->svm_maxqpsize<2)) {
    printf("\nMaximum size of QP-subproblems not in valid range: %ld [2..]\n",learn_parm->svm_maxqpsize); 
  //  wait_any_key();
  //  print_help();
    exit(0);
  }
  if((learn_parm->svm_maxqpsize<learn_parm->svm_newvarsinqp)) {
    printf("\nMaximum size of QP-subproblems [%ld] must be larger than the number of\n",learn_parm->svm_maxqpsize); 
    printf("new variables [%ld] entering the working set in each iteration.\n",learn_parm->svm_newvarsinqp); 
  //  wait_any_key();
  //  print_help();
    exit(0);
  }
  if(learn_parm->svm_iter_to_shrink<1) {
    printf("\nMaximum number of iterations for shrinking not in valid range: %ld [1,..]\n",learn_parm->svm_iter_to_shrink);
  //  wait_any_key();
  //  print_help();
    exit(0);
  }
  if(learn_parm->svm_c<0) {
    printf("\nThe C parameter must be greater than zero!\n\n");
 //   wait_any_key();
 //   print_help();
    exit(0);
  }
  if(learn_parm->transduction_posratio>1) {
    printf("\nThe fraction of unlabeled examples to classify as positives must\n");
    printf("be less than 1.0 !!!\n\n");
 //   wait_any_key();
 //   print_help();
    exit(0);
  }
  if(learn_parm->svm_costratio<=0) {
    printf("\nThe COSTRATIO parameter must be greater than zero!\n\n");
 //   wait_any_key();
 //   print_help();
    exit(0);
  }
  if(learn_parm->epsilon_crit<=0) {
    printf("\nThe epsilon parameter must be greater than zero!\n\n");
 //   wait_any_key();
 //   print_help();
    exit(0);
  }
  if(learn_parm->rho<0) {
    printf("\nThe parameter rho for xi/alpha-estimates and leave-one-out pruning must\n");
    printf("be greater than zero (typically 1.0 or 2.0, see T. Joachims, Estimating the\n");
    printf("Generalization Performance of an SVM Efficiently, ICML, 2000.)!\n\n");
 //   wait_any_key();
 //   print_help();
    exit(0);
  }
  if((learn_parm->xa_depth<0) || (learn_parm->xa_depth>100)) {
    printf("\nThe parameter depth for ext. xi/alpha-estimates must be in [0..100] (zero\n");
    printf("for switching to the conventional xa/estimates described in T. Joachims,\n");
    printf("Estimating the Generalization Performance of an SVM Efficiently, ICML, 2000.)\n");
 //   wait_any_key();
 //   print_help();
    exit(0);
  }
}

//------- SVM default parameters -------------------//


// Copy feature added by Garrett Nicolai, February, 2014.
// The copy feature is a single feature that determines whether
// an operation maps x to x, ie. if it simply copies the characters
// on the left-hand side to the right-hand side.
// It was inspired by a similar feature mentioned in 
// "A global model for joint lemmatization and part-of-speech prediction"
// by Kristina Toutanova and Colin Cherry, ACL 2009

// LM feature added by Garrett Nicolai, June, 2017
// Using a language model learned elsewhere, and stored
// in arpa format, extract a number of features
// related to the likelihood of the thus-far-generated
// candidate.  Rather than use normalized log-likelihood,
// we bin the likelihood in order to maintain an all-binary
// feature set.
// Most log-likelihoods of this type seem to fall between
// -0.5 and -1.5, which allows us to create 10 bins.

phraseModel::phraseModel(void)
{
}

phraseModel::~phraseModel(void)
{
}

void phraseModel::readingExtraFeature(param &myParam, string filename, hash_string_vData& output)
{
	cout << "Reading file: " << filename << endl;

	ifstream INPUTFILE;

	INPUTFILE.open(filename.c_str());
	if (! INPUTFILE)
	{
		cerr << endl << "Error: unable to open file " << filename << endl;
		exit(-1);
	}

	while (! INPUTFILE.eof())
	{
	}
	
	INPUTFILE.close();
}

void phraseModel::readWCFile(param &myParam, string filename, hash_string_double& WLCounts)
{
	cout << "Reading word list: " << filename << endl;


	ifstream INPUTFILE;
	double normalizer = 0.0;
	INPUTFILE.open(filename.c_str());
        if (! INPUTFILE)
        {
                cerr << endl << "Error: unable to open file " << filename << endl;
                exit(-1);
        }


	while (! INPUTFILE.eof())
	{

		string line; 
		getline(INPUTFILE, line);
		::setlocale(LC_ALL, "");
   		std::transform(line.begin(), line.end(), line.begin(), ::towlower);
  

		size_t firstIndex = line.find_first_not_of(' '); //trim function from https://stackoverflow.com/questions/25829143/c-trim-whitespace-from-a-string
    		size_t lastIndex = line.find_last_not_of(' ');

		if(firstIndex == std::string::npos || lastIndex == std::string::npos)
		{
			continue;
		}
		line = line.substr(firstIndex, lastIndex - firstIndex + 1);
	        std::vector<std::string> parts;
                std::size_t pos = 0, found;

                while((found = line.find_first_of("\t", pos)) != std::string::npos) {
                        parts.push_back(line.substr(pos, found - pos));
                        pos = found+1;
                }
		parts.push_back(line.substr(pos));

		if(parts.size() < 2)
		{
			continue;
		}



		 //cout << "SUBSEQ: " << subSequence << endl;
                int trueLength = utf8::distance(parts[1].c_str(), parts[1].c_str() + parts[1].length());
                //cout << "LENGTH: " << trueLength << endl;
		double count = ::atof(parts[0].c_str());
		if(normalizer == 0.0)
		{
			normalizer = count / 275000; //Tuning found this value to be best
			cout << "Normalizer: " << normalizer << endl;
		}
		count /= normalizer;
                for(int i = 0; i < trueLength; i++)
                {

			string prefix = "<w>" + utf8_substr(parts[1], 0, i+1);
			//cout << "WORD:" << parts[1] << endl;
		 	if(WLCounts.find(prefix) == WLCounts.end())
			{
				WLCounts[prefix] = 0.0;
			}

			WLCounts[prefix] = WLCounts[prefix] + (count * (i / trueLength));
                        
		}

		if(WLCounts.find("<w>" + parts[1] + "<\\w>") == WLCounts.end())
		{
			WLCounts["<w>" + parts[1] + "<\\w>"] == 0.0;
		}
		
		WLCounts["<w>" + parts[1] + "<\\w>"] = WLCounts["<w>" + parts[1] + "<\\w>"] + count; 



	}
	cout << "Done reading!" << endl;

}	

void phraseModel::readLMFile(param &myParam, string filename, hash_string_double& LMProbs, hash_string_double& LMBackoff, int& maxLM)
{

	cout << "Reading LM File: " << filename << endl;

	ifstream INPUTFILE;
	
	INPUTFILE.open(filename.c_str());
	if (! INPUTFILE)
	{
		cerr << endl << "Error: unable to open file " << filename << endl;
		exit(-1);
	}

	bool dataFound = false;	
	bool readToMap = false; //There is a lot of preamble; ignore it


	while (! INPUTFILE.eof())
	{
		string line; 

		getline(INPUTFILE, line);

		if(line == "\\data\\")
		{
			dataFound = true;
		}
		if(line == "\\end\\")//end of LM
		{
			break;
		}
		if(dataFound && line == "\\1-grams:")
		{
			readToMap = true; //Start of actual LM; start reading
		}

		if(! readToMap || line == "")
		{
			continue;
		}

		//cout << line << endl;
		std::vector<std::string> parts;
		std::size_t pos = 0, found;

		while((found = line.find_first_of("\t ", pos)) != std::string::npos) {
    			parts.push_back(line.substr(pos, found - pos));
    			pos = found+1;
		}
		parts.push_back(line.substr(pos));
		//cout << parts[0] << endl;
		if(parts.size() == 1)//This will only happen for n-gram size
		{
			string nGramSize = parts[0].substr(1, parts[0].find('-') - 1);
			istringstream buffer(nGramSize);
			buffer >> maxLM;//Convert LM size to integer
			//N-grams in ARPA model are in order of increasing size
			//cout << maxLM << endl;
			continue;
		}

		string ngram = "";
		for(int i = 1; i < parts.size() - 1; i++)//parts[0] = probability; parts[n] = backoff probability
		{
			ngram += parts[i];
		}

		//cout << "N-gram: " << ngram << endl; 


		if(parts.size() != maxLM + 2)
		{
			ngram += parts[parts.size() - 1];//This happens when there is no backoff score
		}

		else
		{
			istringstream buffer(parts[parts.size() - 1]);
			double backoff = -1.0;
			buffer >> backoff;
			LMBackoff[ngram] = backoff;
			//cout << "Backoff: " << backoff << endl; 

		}

		istringstream buffer(parts[0]);
		double prob = -9.0;
		buffer >> prob;
		LMProbs[ngram] = prob;
		//cout << "Prob: " << prob << endl; 

	}

	//double probability = getLMProbability("adelantada", LMProbs, LMBackoff, maxLM, true);
		
	return;

}

void phraseModel::processExample(param &myParam, size_t& tot_read, vector<string> example, vector_vData& processedExample, bool genFea)
{


	data lineData;



        //read aligned data //
    	Tokenize(example[0], lineData.alignedX, "|");
		
    	if (example.size() > 1)
    	{
      		Tokenize(example[1], lineData.alignedY, "|");
    	}

	if (example.size() > 2)
	{
		//lineData.alignRank = convertTo<int>(example[2]);
		lineData.alignRank = atoi(example[2].c_str());
	}
	else
	{
		// default // 
		lineData.alignRank = 1;
	}

	if (example.size() > 3)
	{
		//lineData.alignScore = convertTo<double>(example[3]);
		lineData.alignScore = atof(example[3].c_str());
	}
	else
	{
		lineData.alignScore = 1;
	}

	// re-format to unaligned data //
	for (vector<string>::iterator pos = lineData.alignedX.begin() ; pos != lineData.alignedX.end() ; pos++)
	{
		Tokenize(*pos, lineData.unAlignedX, myParam.inChar);
		int nDelete = removeSubString(*pos, myParam.inChar); // remove inChar
		int phraseSize;
		if (myParam.inChar == "")
		{
			phraseSize = (*pos).size();
		}
		else
		{
			phraseSize = nDelete + 1;
		}
		lineData.phraseSizeX.push_back(phraseSize);

		// re-adjust maxX according to |phrase| //
		if (phraseSize > myParam.maxX)
		{
			myParam.maxX = phraseSize;
		}
	}
	removeVectorElem(lineData.unAlignedX, "_"); // remove null

	if (example.size() > 1)
	{
		// re-format to unaligned data //
		for (vector<string>::iterator pos = lineData.alignedY.begin() ; pos != lineData.alignedY.end() ; pos++)
		{
			Tokenize(*pos, lineData.unAlignedY, myParam.inChar);
			removeSubString(*pos, myParam.inChar); // remove inChar
		}
		removeVectorElem(lineData.unAlignedY, "_"); // remove null
	}
	//output.push_back(lineData);

	if (genFea)
	{
		lineData.feaVec = genFeature(myParam, lineData);
	}

	string unAlignedWord = join(lineData.unAlignedX, "", "");

	//processedExample[unAlignedWord].push_back(lineData);
	dataplus dataTMP; 
	dataTMP.mydata.push_back(lineData);
	processedExample.push_back(dataTMP);
	//processedExample.[unAlignedWord].mydata.push_back(lineData);
	tot_read++;


}
        

void phraseModel::readingTestingFile(param &myParam, string filename, vector_vData& output, bool genFea)
{
	size_t totRead = 0;
	cout << "Reading file: " << filename << endl;
	
	ifstream INPUTFILE;

	INPUTFILE.open(filename.c_str());
	if (! INPUTFILE)
	{
		cerr << endl << "Error: unable to open file " << filename << endl;
		exit(-1);
	}

	while (! INPUTFILE.eof())
	{
		string line;
		vector<string> lineList;

		data lineData;

		getline(INPUTFILE, line);

		// ignore empty line
		if (line == "")
		{
			continue;
		}

		// ignore line that indicate no alignment //
		if (line.find("NO ALIGNMENT") != string::npos)
		{
			continue;
		}

		lineList = splitBySpace(line);

		if (lineList.size() > 4)
		{
			cerr << endl << "Warning: wrong expected format" << endl << line << endl;
		}
		else
		{
			// read aligned data //
			Tokenize(lineList[0], lineData.alignedX, "|");
			
			if (lineList.size() > 1)
			{
				Tokenize(lineList[1], lineData.alignedY, "|");
			}

			if (lineList.size() > 2)
			{
				//lineData.alignRank = convertTo<int>(lineList[2]);
				lineData.alignRank = atoi(lineList[2].c_str());
			}
			else
			{
				// default // 
				lineData.alignRank = 1;
			}

			if (lineList.size() > 3)
			{
				//lineData.alignScore = convertTo<double>(lineList[3]);
				lineData.alignScore = atof(lineList[3].c_str());
			}
			else
			{
				lineData.alignScore = 1;
			}

			// re-format to unaligned data //
			for (vector<string>::iterator pos = lineData.alignedX.begin() ; pos != lineData.alignedX.end() ; pos++)
			{
				Tokenize(*pos, lineData.unAlignedX, myParam.inChar);
				int nDelete = removeSubString(*pos, myParam.inChar); // remove inChar
				int phraseSize;
				if (myParam.inChar == "")
				{
					phraseSize = (*pos).size();
				}
				else
				{
					phraseSize = nDelete + 1;
				}
				lineData.phraseSizeX.push_back(phraseSize);

				// re-adjust maxX according to |phrase| //
				if (phraseSize > myParam.maxX)
				{
					myParam.maxX = phraseSize;
				}
			}
			removeVectorElem(lineData.unAlignedX, "_"); // remove null

			if (lineList.size() > 1)
			{
				// re-format to unaligned data //
				for (vector<string>::iterator pos = lineData.alignedY.begin() ; pos != lineData.alignedY.end() ; pos++)
				{
					Tokenize(*pos, lineData.unAlignedY, myParam.inChar);
					removeSubString(*pos, myParam.inChar); // remove inChar
				}
				removeVectorElem(lineData.unAlignedY, "_"); // remove null
			}
			//output.push_back(lineData);

			if (genFea)
			{
				lineData.feaVec = genFeature(myParam, lineData);
			}

			string unAlignedWord = join(lineData.unAlignedX, "", "");

			//output[unAlignedWord].push_back(lineData);
			dataplus dataTMP; 
			dataTMP.mydata.push_back(lineData);
			output.push_back(dataTMP);
			//output[unAlignedWord].mydata.push_back(lineData);
			totRead++;
		}
	}
	INPUTFILE.close();
	cout << "Total read: " << totRead << " instances" << endl;
}

void phraseModel::readingAlignedFile(param &myParam, string filename, hash_string_vData& output, bool genFea)
{
	size_t totRead = 0;
	cout << "Reading file: " << filename << endl;
	
	ifstream INPUTFILE;

	INPUTFILE.open(filename.c_str());
	if (! INPUTFILE)
	{
		cerr << endl << "Error: unable to open file " << filename << endl;
		exit(-1);
	}

	while (! INPUTFILE.eof())
	{
		string line;
		vector<string> lineList;

		data lineData;

		getline(INPUTFILE, line);

		// ignore empty line
		if (line == "")
		{
			continue;
		}

		// ignore line that indicate no alignment //
		if (line.find("NO ALIGNMENT") != string::npos)
		{
			continue;
		}

		lineList = splitBySpace(line);

		// ignore empty line filled with space //
		if (lineList.size() < 1)
		{
			continue;
		}

		if (lineList.size() > 4)
		{
			cerr << endl << "Warning: wrong expected format" << endl << line << endl;
		}
		else
		{
			// read aligned data // 
			Tokenize(lineList[0], lineData.alignedX, "|");
			if (lineList.size() > 1)
			{
				Tokenize(lineList[1], lineData.alignedY, "|");
			}

			if (lineList.size() > 2)
			{
				//lineData.alignRank = convertTo<int>(lineList[2]);
				lineData.alignRank = atoi(lineList[2].c_str());
			}
			else
			{
				// default // 
				lineData.alignRank = 1;
			}

			if (lineList.size() > 3)
			{
				//lineData.alignScore = convertTo<double>(lineList[3]);
				lineData.alignScore = atof(lineList[3].c_str());
			}
			else
			{
				lineData.alignScore = 1;
			}

			// re-format to unaligned data //
			for (vector<string>::iterator pos = lineData.alignedX.begin() ; pos != lineData.alignedX.end() ; pos++)
			{
				Tokenize(*pos, lineData.unAlignedX, myParam.inChar);
				int nDelete = removeSubString(*pos, myParam.inChar); // remove inChar
				int phraseSize;
				if (myParam.inChar == "")
				{
					phraseSize = (*pos).size();
				}
				else
				{
					phraseSize = nDelete + 1;
				}
				lineData.phraseSizeX.push_back(phraseSize);

				// re-adjust maxX according to |phrase| //
				if (phraseSize > myParam.maxX)
				{
					myParam.maxX = phraseSize;
				}
			}
			removeVectorElem(lineData.unAlignedX, "_"); // remove null

			if (lineList.size() > 1)
			{
				// re-format to unaligned data //
				for (vector<string>::iterator pos = lineData.alignedY.begin() ; pos != lineData.alignedY.end() ; pos++)
				{
					Tokenize(*pos, lineData.unAlignedY, myParam.inChar);
					removeSubString(*pos, myParam.inChar); // remove inChar
				}
				removeVectorElem(lineData.unAlignedY, "_"); // remove null
			}
			//output.push_back(lineData);

			if (genFea)
			{
				lineData.feaVec = genFeature(myParam, lineData);
			}

			string unAlignedWord = join(lineData.unAlignedX, "", "");

			//output[unAlignedWord].push_back(lineData);
			output[unAlignedWord].mydata.push_back(lineData);
			totRead++;
		}
	}
	INPUTFILE.close();
	cout << "Total read: " << totRead << " instances" << endl;
}

void phraseModel::initialize(param& myParam, hash_string_vData& trainDataUnique)
{
	for (hash_string_vData::iterator train_pos = trainDataUnique.begin(); train_pos != trainDataUnique.end(); train_pos++)
	{
		//for (unsigned long i = 0; i < train_pos->second.size(); i++)
		for (unsigned long i = 0; i < train_pos->second.mydata.size(); i++)
		{
			//for (unsigned long j = 0; j < train_pos->second[i].alignedX.size() ; j++)
			for (unsigned long j = 0; j < train_pos->second.mydata[i].alignedX.size() ; j++)
			{
				//myAllPhoneme.addPhoneme(train_pos->second[i].alignedY[j], train_pos->second[i].alignedX[j], true);
				myAllPhoneme.addPhoneme(train_pos->second.mydata[i].alignedY[j], train_pos->second.mydata[i].alignedX[j], true);
			}
		}
	}
}

vector_str phraseModel::ngramFeatureGen(param &myParam, vector_str lcontext, vector_str focus, vector_str rcontext)
{
	vector_str output;
	vector_str allSeen;
	string mergeFocus = join(focus,"","");

	if (lcontext.size() < myParam.contextSize)
	{
		lcontext.insert(lcontext.begin(), "{");
	}

	if (rcontext.size() < myParam.contextSize)
	{
		rcontext.push_back("}");
	}

	int posFocus = lcontext.size();

	allSeen.insert(allSeen.end(), lcontext.begin(), lcontext.end()); // left context
	allSeen.push_back(mergeFocus); // focus token
	allSeen.insert(allSeen.end(), rcontext.begin(), rcontext.end()); // right context

	string feaStr;

	for (int i = 0; i < allSeen.size(); i++)
	{
		for (int k = 1; (k <= myParam.nGram) && (k + i <= allSeen.size()); k++)
		{

			feaStr = "L:" + stringify(i - posFocus) + ":" + stringify(i + k - posFocus - 1) + ":";
			feaStr += join(allSeen, i, i + k, "","");
			output.push_back(feaStr);
		} 
	}
	return output;
}

double phraseModel::getLocalFeatureScore(param &myParam, vector_str observation, string classLabel)
{
	double output = 0;
	
	for (vector_str::iterator feaListPos = observation.begin(); feaListPos != observation.end(); feaListPos++)
	{
		output += myWF.getFeature(*feaListPos, classLabel, myParam.atTesting);
	}

	return output;
}

double phraseModel::getCopyScore(param &myParam, bool isCopy, string classLabel)
{
	double output = 0;
	if(isCopy)
	{
		output = myWF.getFeature("COPY", "COPY",  myParam.atTesting);
		//output = myWF.getFeature("COPY", classLabel, myParam.atTesting);
	}

	return output;
}

double phraseModel::getWCFeatureScore(param &myParam, hash_string_double& WLCounts, string word)
{
	double WCFeatureScore = 0.0;
	double count = 0.0;
	std::setlocale(LC_CTYPE, "en_US.UTF-8"); // the locale will be the UTF-8 enabled English


	std::vector<std::string> parts;
        std::size_t pos = 0, found;
	::setlocale(LC_ALL, "");
   	std::transform(word.begin(), word.end(), word.begin(), ::towlower);


        //cout << sequence << endl;
        while((found = word.find_first_of("!@", pos)) != std::string::npos) {

                        if(found - pos != 0 && found != 0)
                        {
                                parts.push_back("<w>" + word.substr(pos, found - pos) + "<\\w>");//It should be okay to split with ASCII, because "!" and "@" are ASCII chars
                        }
			else if(found -pos != 0 && found != 0 && pos == 0)//Start of word already has "<w>"
			{
				parts.push_back(word.substr(pos, found - pos) + "<\\w>");
			}
                        pos = found+1;
                }

	if(pos != 0)
	{
        	parts.push_back("<w>" + word.substr(pos));
	}
	else
	{
		parts.push_back(word.substr(pos));
	}

	for(int i = 0; i < parts.size(); i++)
	{
		if(WLCounts.find(parts[i]) == WLCounts.end())
		{
			count += 0.0;	
		}
		else
		{
			count += WLCounts[parts[i]];
		}

		//cout << "WORD: " << parts[i] << endl;
		//cout << "COUNT: " << count << endl;
		
	}

	count /= parts.size();
        string bin = getWCBin(count);
        for(int j = atoi(bin.c_str()); j < 8; j++)
        {
        	stringstream converter;
                converter << j;
                string jString = converter.str();
                WCFeatureScore += myWF.getFeature("WCBIN:" + jString, "WC", myParam.atTesting);
        }

        //cout << "WORD: " << word << endl;
        //cout << "COUNT: " << count << endl;

	//WCFeatureScore /= parts.size(); //Get average word score

	//cout << "Score: " << count << endl;
	//cout << "Bin: " << bin << endl;
	return WCFeatureScore;

}

string phraseModel::getWCBin(double count)
{
	if(count >= 1000000)
		return "0";
	if(count >= 100000)
		return "1";
	if(count >= 10000)
		return "2";
	if(count >= 1000)
		return "3";
	if(count >= 100)
		return "4";
	if(count >= 10)
		return "5";
	if(count >= 1)
		return "6";
	else
		return "7";

	/*


	if(count < 1)
        {
		return "0";
        }
        if(count < 2)
        {
		return "1";
        }
        if(count < 20)
        {
		return "2";
        }
        if(count < 200)
        {
		return "3";
        }
        if(count < 2000)
        {
		return "4";
        }

	return "5";

	*/
}

double phraseModel::getLMFeatureScore(param &myParam, double LMScore)
{
	double LMFeatureScore = 0.0;
	string bin = getLMBin(LMScore);

        for(int j = atoi(bin.c_str()); j < 8; j++)
        {
             stringstream converter;
             converter << j;
             string jString = converter.str();
             LMFeatureScore += myWF.getFeature("LMBIN:" + jString, "LM", myParam.atTesting);
        }

	//LMFeatureScore = myWF.getFeature("LMBIN:" + bin, "LM", myParam.atTesting);

	//cout << "Score: " << LMScore << endl;
	//cout << "Bin: " << bin << endl;
	return LMFeatureScore;
}

string phraseModel::getLMBin(double LMScore)
{
	if(LMScore > -0.60)
		return "0";
	if(LMScore > -0.75)
		return "1";
	if(LMScore > -0.9)//Probably a word
		return "2";
	if (LMScore > -1.05)//~1SD
		return "3";
	if (LMScore > -1.3)//~2SD
		return "4";
	if (LMScore > -1.45)//~3SD
		return "5";
	if (LMScore > -1.6)//~4SD
		return "6";
		
	return "7";
	

}
double phraseModel::getTurkishVHScore(param &myParam, string correctY, string currentY, vector_str jointY)
{
	double modifier = 1.0;
	double modifier2 = -1.0;
	string firstVowel = "";
	for(int i = 0; i < currentY.length(); i++)
	{
		//int len = 1;
		int len = (( 0xE5000000 >> (( currentY[i] >> 3 ) & 0x1e )) & 3 ) + 1; //This line determines the multi-byte length of the character
		string currentChar = currentY.substr(i, len);//And obtains the character
		i += len - 1; //update i

		if(containsTurkishVowel(currentChar))
		{
			firstVowel = currentChar;
			break;
		}
	}
	/*if(firstVowel.compare("") == 0)
	{
		cout << "No vowel in suffix" << endl;
	}
	else
	{	
		cout << "First vowel in suffix:\t" << firstVowel << endl;
	}*/

	string suffixClass = classOfLastVowel(firstVowel, "TRK");
	string previousWord = "";

	for (vector_str::iterator stringIterator = jointY.begin(); stringIterator != jointY.end(); ++stringIterator)
	{
		previousWord += *stringIterator;
	}
	//cout << "Stem:\t" << previousWord << endl;
	//cout << "Suffix:\t" << currentY << endl;
	string stemClass = classOfLastVowel(previousWord, "TRK");
	string currentWord = previousWord + currentY;
	currentWord.erase(std::remove(currentWord.begin(), currentWord.end(), '_'), currentWord.end());	
	correctY.erase(std::remove(correctY.begin(), correctY.end(), '_'), correctY.end());
	if(!myParam.atTesting && correctY.find(currentWord) != 0)
	{
		modifier2 = -1.0 * modifier2;
	//	cout << "Flipping modifier because " << correctY << " doesn't contain " << currentWord << endl;
	}

	if(stemClass.compare("Neutral") == 0)
	{
	//	cout << "Score(Positive): " << myWF.getFeature("TRKVH", "TRKVH", myParam.atTesting) * modifier2 << endl;
		return myWF.getFeature("TRKVH", "TRKVH", myParam.atTesting) * modifier2; //If the stem is neutral, it has no vowels, so any suffix if fine
	}
	else if(suffixClass.compare("Neutral") == 0)
	{
	//	cout << "Score(Positive): " << myWF.getFeature("TRKVH", "TRKVH", myParam.atTesting) * modifier2 << endl;
		return myWF.getFeature("TRKVH", "TRKVH", myParam.atTesting) * modifier2; //If the suffix is neutral, it has no vowels, so any suffix if fine
	}

	else if(stemClass.compare(suffixClass) != 0)
	{

	//	cout << "Score(Negative): " << myWF.getFeature("TRKVH", "TRKVH", myParam.atTesting) * modifier  << endl;
		return myWF.getFeature("TRKVH", "TRKVH", myParam.atTesting) * modifier; //Otherwise, if the stem and suffix don't match, return a negative score
	}

	//cout << "Score(Positive): " << myWF.getFeature("TRKVH", "TRKVH", myParam.atTesting) * modifier2 << endl;
	return myWF.getFeature("TRKVH", "TRKVH", myParam.atTesting) * modifier2; //Otherwise, the stem and suffix match, so return a positive score
 	

}

double phraseModel::getTurkishRHScore(param &myParam, string correctY, string currentY, vector_str jointY)
{
	double modifier = 1.0;
	//Penalize a word if it breaks harmony, but do nothing if it doesn't, because unbroken harmony isn't necessarily correct
	double modifier2 = -1.0;
	string firstVowel = "";
	string previousWord = "";
	
	int length;
  	wchar_t dest;

    	for (vector_str::iterator stringIterator = jointY.begin(); stringIterator != jointY.end(); ++stringIterator)
	{
		previousWord += *stringIterator;
	}

	//cout << "Stem:\t" << previousWord << endl;
	//cout << "Suffix:\t" << currentY << endl;
	//cout << "Correct:\t" << correctY << endl;	
	string currentWord = previousWord + currentY;
	currentWord.erase(std::remove(currentWord.begin(), currentWord.end(), '_'), currentWord.end());	
	correctY.erase(std::remove(correctY.begin(), correctY.end(), '_'), correctY.end());
	if(!myParam.atTesting && correctY.find(currentWord) != 0)
	{
	//	cout << "Flipping RH" << "\t" << correctY << "\t" << currentWord << "\t" << correctY.find(currentWord) << "\n";
		modifier2 = -1.0 * modifier2;
	}
	if(!myParam.atTesting && correctY.find(previousWord) != 0)
	{
	//	cout << "Flipping RH back, as the problem is due to the stem\n";
		modifier2 = -1.0 * modifier2;
	}

	//cout << "CurrentY:\t" << currentY << endl;
	string wordSoFar = "";

	for(int i = 0; i < currentY.size(); i++)
	{
		//int len = 1;
		int len = (( 0xE5000000 >> (( currentY[i] >> 3 ) & 0x1e )) & 3 ) + 1; //This determines the multi-byte length of the character
		string currentChar = currentY.substr(i, len);//And obtains the character
		i += len - 1; //update i
		if(currentChar.compare("i") == 0 || currentChar.compare("I") == 0)//If we have a candidate for rounding
		{
			if(lastVowelRounded(wordSoFar))//If the candidate itself breaks rounding harmony
			{
	//			cout << "Negative in Suffix\n";
	//			cout << myWF.getFeature("TRKRH", "TRKRH", myParam.atTesting) * modifier << "\n";	
				return myWF.getFeature("TRKRH", "TRKRH", myParam.atTesting) * modifier; //return the negative score
			}
		}
			
		wordSoFar += currentChar;
		

		if(firstVowel.compare("") == 0 && containsTurkishVowel(currentChar))
		{
			firstVowel = currentChar;//record the first vowel in the sequence
		}

	}
	if(firstVowel.compare("") != 0)
	{
	//	cout << "First vowel of suffix:\t" << firstVowel << endl;
	}
	if((firstVowel.compare("i") == 0 || firstVowel.compare("I") == 0) && lastVowelRounded(previousWord))//If we have a rounding conflict)
	{
	//	cout << "Negative\n";
	//	cout << myWF.getFeature("TRKRH", "TRKRH", myParam.atTesting) * modifier << "\n";	
		return myWF.getFeature("TRKRH", "TRKRH", myParam.atTesting) * modifier;  //return the negative score
	}
	//cout << "Positive\n";
	//cout << myWF.getFeature("TRKRH", "TRKRH", myParam.atTesting) * modifier2 << "\n";	

	return myWF.getFeature("TRKRH", "TRKRH", myParam.atTesting) * modifier2; //Otherwise, it's ok to use the sequence

}

double phraseModel::getFinnishVHScore(param &myParam, string correctY, string currentY, vector_str jointY)
{
	double modifier = 1.0;
	double modifier2 = -1.0;
	string firstVowel = "";
	//cout << "Considering:\t" << currentY << endl;
	for(int i = 0; i < currentY.length(); i++)
	{
		//int len = 1;
		int len = (( 0xE5000000 >> (( currentY[i] >> 3 ) & 0x1e )) & 3 ) + 1; //This determines the multi-byte length of the character
		string currentChar = currentY.substr(i, len);//And obtains the character
		i += len - 1; //update i
		if(containsFinnishVowel(currentChar) > 0)
		{
			firstVowel = currentChar;
			break;
		}
	}

	string suffixClass = classOfLastVowel(currentY, "FIN");
	string previousWord = "";
	for (vector_str::iterator stringIterator = jointY.begin(); stringIterator != jointY.end(); ++stringIterator)
	{
		previousWord += *stringIterator;
	}
	//cout << "Stem:\t" << previousWord << endl;
	//cout << "Correct:\t" << correctY << endl;

	string currentWord = previousWord + currentY;
	currentWord.erase(std::remove(currentWord.begin(), currentWord.end(), '_'), currentWord.end());	
	correctY.erase(std::remove(correctY.begin(), correctY.end(), '_'), correctY.end());
	
	if(!myParam.atTesting && correctY.find(currentWord) != 0)
	{
	//	cout << "Flipping Modifier" << endl;
		modifier2 = -1.0 * modifier2;
	}

	string stemClass = classOfLastVowel(previousWord, "FIN");
	
	//cout << "Stem Class:\t" << stemClass << endl;
	//cout << "Suffix Class:\t" << suffixClass << endl;

	if(stemClass.compare("Neutral") == 0 || suffixClass.compare("Neutral") == 0)
	{
	//	cout << "Positive: " << myWF.getFeature("FINVH", "FINVH", myParam.atTesting) * modifier2<< endl;;
		return myWF.getFeature("FINVH", "FINVH", myParam.atTesting) * modifier2; //If the stem is neutral, it has no vowels, so any suffix if fine
	}
	else if(stemClass.compare(suffixClass) != 0)
	{
	//	cout << "Negative: " << myWF.getFeature("FINVH", "FINVH", myParam.atTesting) * modifier << endl;;

		return myWF.getFeature("FINVH", "FINVH", myParam.atTesting) * modifier; //Otherwise, if the stem and suffix don't match, return a negative score
	}

	//cout << "Positive: " << myWF.getFeature("FINVH", "FINVH", myParam.atTesting) * modifier2 << endl;

	return myWF.getFeature("FINVH", "FINVH", myParam.atTesting) * modifier2; //Otherwise, the stem and suffix match, so return a positive score
 	

}

double phraseModel::getOrderFeatureScore(param &myParam, vector_str observation, string p_class, string c_class)
{
	double output = 0;

	if (myParam.markovOrder == 1)
	{
		output += myWF.getFeature("P:-1:" + p_class, c_class, myParam.atTesting);
	}

	if (myParam.linearChain)
	{
		for (vector_str::iterator feaListPos = observation.begin(); feaListPos != observation.end(); feaListPos++)
		{
			output += myWF.getFeature(*feaListPos + "P:-1:" + p_class, c_class, myParam.atTesting);
		}
	}
	return output;
}


double phraseModel::getHigherOrderFeatureScore(param &myParam, vector_str observation, vector_str jointY, string c_class)
{
        //We need to look at the last markovOrder characters in the prediction, but ignore _'s.
        //To do so, we look at the jointY, and iterate backwards until we either
 	//get the number of previous characters we want,
        //or hit the beginning of the string.  In this way, the function is similar
	//to getJointFeatureScore.

        double output = 0;

        if (myParam.markovOrder <= 0)
                return 0;

	int i = jointY.size();
		if (myParam.markovOrder >= 1)
		{
			string runningHistory = "";
			int k = 0;
                        for(int j = 1; j <= myParam.markovOrder;)
			{
				//cout << "ORDER:MARKOV" + stringify(i - j - k) + "\n";
				int x = i - j - k;

				if(x <= 0)
				{
					output += myWF.getFeature("P:-" + stringify(j) + ":{" + runningHistory, c_class, myParam.atTesting);
					break;
					
					break;
				}
				else
				{
					if(myParam.ignoreNull && jointY[x].compare("_") == 0)
					{
						k++;
					}
					else
					{
						runningHistory = jointY[x] + runningHistory;
						output += myWF.getFeature("P:-" + stringify(j) + ":" + runningHistory, c_class, myParam.atTesting);

						j++;
					}
				}//\else
			}//\for
		}//\if
	

		if (myParam.linearChain)
		{
			for(vector_str::iterator feaListPos = observation.begin(); feaListPos != observation.end(); feaListPos++)
			{
				string runningHistory = "";
				int k = 0;
                        	for(int j = 1; j <= myParam.markovOrder;)
				{
					//cout << "ORDER:CHAIN" + stringify(i - j - k) + "\n";
					int x = i - j - k;

					if(x <= 0)
					{
						output += myWF.getFeature(*feaListPos + "P:-" + stringify(j) + ":{" + runningHistory, c_class, myParam.atTesting);
						break;
					}
					else
					{
						if(myParam.ignoreNull && jointY[x].compare("_") == 0)
						{
							k++;
						}
						else
						{
							runningHistory = jointY[x] + runningHistory;
							output += myWF.getFeature(*feaListPos + "P:-" + stringify(j) + ":" + runningHistory, c_class, myParam.atTesting);
							j++;
						}
					}//\else
				}//\for
			}//\for
		}//\if

		

//        int max_history = min<int>(myParam.markovOrder, (jointY.size() + 1));

		

//        int max_history = min<int>(myParam.markovOrder, (jointY.size() + 1));
//
//	if(max_history < myParam.markovOrder)
//	{
//		max_history++;
//	}

//        string runningHistory("");
//        int j = 1;
//        for(int i = jointY.size() - 1; i >-1 && j <= max_history; i--)
//        {
//                string currentChar = "";//join(jointY, i, i, "", "");
//		if(i == 0)
//		{
//			currentChar = "{";
//		}
//		else
//		{
//               	string currentChar = join(jointY, i-1, i, "", "");
//                	//currentChar.erase(std::remove(currentChar.begin(), currentChar.end(), '_'), currentChar.end());
//		}
//               if(currentChar.compare("_") == 0)
//                {
//                        //continue;
//               }
//                else
//                {
//                        runningHistory = currentChar + runningHistory;
//                        output += myWF.getFeature("P:-" + stringify(j) + ":" + runningHistory, c_class, myParam.atTesting);
//                        if (myParam.linearChain)// && j == 1)
//                       {
//                                for(vector_str::iterator feaListPos = observation.begin(); feaListPos != observation.end(); feaListPos++)
//                                {
//                                        output += myWF.getFeature(*feaListPos + "P:-" + stringify(j) + ":" + runningHistory, c_class, myParam.atTesting);
//                                }
//                       }
//                        j++;
//                }
//        }

//        if(max_history == 0)//If there was no history
//        {
//                output += myWF.getFeature("P:-" + stringify(j) + ":" + "", c_class, myParam.atTesting);
//                if (myParam.linearChain)
//                {
//                        for(vector_str::iterator feaListPos = observation.begin(); feaListPos != observation.end(); feaListPos++)
//                        {
//                                output += myWF.getFeature(*feaListPos + "P:-" + stringify(j) + "" + ":", c_class, myParam.atTesting);
//                        }
//                }
//        }

    return output;
}
       

double phraseModel::getJointForwardGramFeatureScore(param &myParam, string currentX, string currentY, vector_str xForward)
{
	double output = 0;

	// start to count 1-gram, 2-gram ..., M-gram features //
	// note: if FM=0, it means we don't include the feature //
	// note: if FM=1, it is the unigram. We don't count it here //
	
	if (myParam.jointFMgram <= 1)
		return 0;
	//As the algorithm previously stood, word boundaries were ignored.
	//I've made a small fix to change that.
	int max_forward = min<int>(myParam.jointFMgram, xForward.size() + 1);

	vector_str yCurrentCandidate;
	vector_str yPastCandidate;

	// too much thinking now: so, for now, it works only single token (no phrasal input) //
	//As the algorithm previously stood, word boundaries were ignored.
	//I've made a small fix to change that.

	for (int i = 1; i < max_forward; i++)
	//for (int i = 1; i < max_forward; i++)
	{
		yCurrentCandidate = myAllPhoneme.getPhoneme(xForward[i-1], true);
		string feaStrX = join(xForward, 0, i, "-" , "");
		string feaStrY;

		vector_str yKeepHistory;
		for (vector_str::iterator pos = yCurrentCandidate.begin() ; pos != yCurrentCandidate.end(); pos++)
		{
			if ( i-1 > 0)
			{
				for (vector_str::iterator p_pos = yPastCandidate.begin() ; p_pos != yPastCandidate.end(); p_pos++)
				{
					feaStrY = *p_pos + "-" + *pos;
					yKeepHistory.push_back(feaStrY);

					output += myWF.getFeature("JL:1:" + stringify(i) + ":" + feaStrX + "JP:" + feaStrY + "L:" + currentX, 
								currentY, myParam.atTesting);
				}
			}
			else
			{
				feaStrY = *pos;
				yKeepHistory.push_back(feaStrY);

				output += myWF.getFeature("JL:1:" + stringify(i) + ":" + feaStrX + "JP:" + feaStrY + "L:" + currentX, 
							currentY, myParam.atTesting);
			}
		}
		yPastCandidate = yKeepHistory;
	}

	return output;
}

double phraseModel::getJointGramFeatureScore(param &myParam, vector_str jointX, vector_str jointY, string currentX, string currentY)
{
	double output = 0;

	// start to count 1-gram, 2-gram ..., M-gram features //
	// note: if M=0, it means we don't include the jointMgram feature //

	if (myParam.jointMgram <= 0)
		return 0;


	int max_history = min<int>(myParam.jointMgram, (jointX.size() + 1));
	//if(max_history < myParam.jointMgram)
	//{
	//	max_history++;
	//}
	// get score of a unigram (currentX,currentY) //
	output += myWF.getFeature("JL:0:0:JP:L:" + currentX, currentY, myParam.atTesting);

	// get scores of looking back i history;  //
	//As it is, the joint m-gram will only go to position 1;
	//For example, if we are calculating character 5 on a word, and allow a maxJoint mgram of 5
	//jointX is of size 4 (0,1,2,3).
        //max_history = 5, and i will reach a max of 4
	//For i = 1, we will look at joint(3)- joint(4)
	//For i = 2, we will look at joint(2)- joint(3); joint(2)-joint(4)
	//For i = 3, we will look at joint(1) - joint(2); joint(1)-joint(3); joint(1)-joint(4)
	//For i = 4, we will look at joint(0) - joint(1); joint(0)-joint(2); joint(0)-joint(3); joint(0)-joint(4);
	//However, this ignores the fact that this is the start of the word.
	//I fix it by allowing i to equal the max_history; if the jointSize - i == 0, then we use the buffer character {
	
        //for (int i = 1; i < max_history; i++)
	for (int i = 1; i <= max_history; i++)
	{
		for (int j = i ; j > 0; j--)
		{
			string feaStrX = "";
			string feaStrY = "";
			if(i > jointX.size())
			{
				feaStrX = join(jointX, 0, jointX.size() - j + 1, "-","");
				feaStrY = join(jointY, 0, jointY.size() - j + 1, "-", "");
				feaStrX = "{-" + feaStrX; //These lines here might be causing a problem
				feaStrY = "{-" + feaStrY;
			}

			else
			{
				feaStrX = join(jointX, jointX.size() - i, jointX.size() - j + 1, "-", "");
				feaStrY = join(jointY, jointY.size() - i, jointY.size() - j + 1, "-", "");
			}
			//if(i > jointX.size())//ie, if we are moving past the boundary of the word...
			//{
			//	feaStrX = "{" + feaStrX;
			//	feaStrY = "{" + feaStrY;
			//}	
			output += myWF.getFeature("JL:" + stringify(-i) + ":" + stringify(-j) + ":" + feaStrX + "JP:" + feaStrY + "L:" + currentX, 
			currentY, myParam.atTesting);
		}
	}
	
	return output;
}

vector_2str phraseModel::phrasalDecoder_beam(param &myParam, vector_str unAlignedX, vector_str unAlignedY,
                                                                                         vector_2str &alignedXnBest, vector_3str &featureNbest,
                                                                                         vector<double> &scoreNbest, hash_string_double& WLCounts, hash_string_double& LMProbs, hash_string_double& LMBackoff, int maxLM)
{
	vector_2str nBestOutput;
	vector_3str allFeatureStr;
	DD_btable beamTable;

	beamTable.resize(unAlignedX.size());
	allFeatureStr.resize(unAlignedX.size());
	std::string correctY = "";
	if(!myParam.atTesting)
	{
		for(std::vector<std::string>::const_iterator i = unAlignedY.begin(); i != unAlignedY.end(); ++i)
		{
			correctY += *i;
		}
	}


	//cout << "The unaligned = ";
	//for (std::vector<std::string>::iterator i = unAlignedX.begin(); i != unAlignedX.end(); ++i)
    	//cout << *i << ' ';
	//cout << "\n";

	// go over state //
	for (int i = 0; i < unAlignedX.size(); i++)
	{
		allFeatureStr[i].resize(myParam.maxX + 1);
		// go over phrases //
		for (int k = 1; k <= myParam.maxX ; k++)
		{
			// make sure "stage + phraseSize k" is reachable //
			if ((i+k) > unAlignedX.size())
			{
				continue;
			}

			int lPosMin = i - myParam.contextSize;
			if (lPosMin < 0)
			{
				lPosMin = 0;
			}

			int rPosMax = i + k + myParam.contextSize;
			if (rPosMax > unAlignedX.size())
			{
				rPosMax = unAlignedX.size();
			}
			vector_str focus(unAlignedX.begin() + i , unAlignedX.begin() + i + k);
			vector_str lContext(unAlignedX.begin() + lPosMin, unAlignedX.begin() + i);
			vector_str rContext(unAlignedX.begin() + i + k, unAlignedX.begin() + rPosMax);
			
			vector<string> allCandidate = myAllPhoneme.getPhoneme(join(focus,"",""), true);

			if (allCandidate.size() > myParam.maxCandidateEach)
			{
				myParam.maxCandidateEach = allCandidate.size();
			}

			// default case : add null when |phrase| = 1 and no candidate //
			// to ensure at least one source generates one target (null included)//
			if ((allCandidate.size() == 0 ) && (k == 1))
			{
				allCandidate.push_back("_");

				if(strpbrk(join(focus,"","").c_str(), "*") == 0)
				{
					allCandidate.push_back(join(focus,"","")); //Always allow a candidate to copy to itself.
				}
			}
			
			// skip any |phrase| > 1 and no candidate found //
			if (allCandidate.size() == 0)
			{	
				//We probably don't want to assume that a multi-character phrase 
				//can automatically generate itself.
				continue;
			}

			// all local features at the current phrase and stage //
			vector<string> featureStr;
			if (! myParam.noContextFea)
			{
				// extract n-gram feature // 
				featureStr = ngramFeatureGen(myParam, lContext, focus, rContext);
				allFeatureStr[i][k] = featureStr; // keep k start from 1 (|phrase| >= 1)
			}

			for (vector<string>::iterator c_pos = allCandidate.begin(); c_pos != allCandidate.end(); c_pos++)
			{
				// calculate local features //
				// getLocalFeatureScore contains only context features which
				// basically the function iterates over the vector<string> featureStr 
				// associated with c_pos 

				bool isCopy = ((*c_pos).compare(join(focus, "", "")) == 0);
				double localScore = getLocalFeatureScore(myParam, featureStr, *c_pos);
				double copyScore = 0;
    				double LMFeatureScore = 0.0;
				double WCFeatureScore = 0.0;
				double oldLMScore = 0.0;
				double oldWCScore = 0.0;
				double transScore;
				double jointMScore;
				double jointFMScore;
				double finnishVHScore = 0.0;	
				double turkishVHScore = 0.0;	
				double turkishRHScore = 0.0;	

				if (myParam.copyFeature == true)
				{
					copyScore = getCopyScore(myParam, isCopy, *c_pos);
				}
				if (myParam.jointFMgram > 1)
				{
					int startPos = i + k;
					int stopPos = i + k + myParam.jointFMgram - 1;
					
					if (startPos > unAlignedX.size()) startPos = unAlignedX.size();
					if (stopPos > unAlignedX.size()) stopPos = unAlignedX.size();

					vector_str xForward(unAlignedX.begin() + startPos, unAlignedX.begin() + stopPos);
					jointFMScore = getJointForwardGramFeatureScore(myParam, join(unAlignedX, i, i + k, "", ""), *c_pos, xForward);
				}
				else
				{
					jointFMScore = 0;
				}
			
				// no previous decision yet //
    				bool endOfWord = (i + k == unAlignedX.size());
				if (i == 0)
				{
					// getOrderFeatureScore contains only markov order 1 and linear-chain 
					// In linear-chain, we iterate over featureStr associated with ""(c-1_pos) and c_pos
					// In markov=1, we get score of ""(c-1_pos) and c_pos 
					//transScore = getOrderFeatureScore(myParam, featureStr, "", *c_pos);

					btable Btmp;

					Btmp.currentX = join(unAlignedX, i, i + k, "", "");
					Btmp.currentY = *c_pos;
					Btmp.phraseSize.push_back(k);
					Btmp.jointX.clear();
					Btmp.jointY.clear();
					Btmp.LMScore = 0.0;
					
					transScore = getHigherOrderFeatureScore(myParam, featureStr, Btmp.jointY, *c_pos);
					jointMScore = getJointGramFeatureScore(myParam, Btmp.jointX, Btmp.jointY, Btmp.currentX, Btmp.currentY);
					if(myParam.finnishVH == true && containsFinnishVowel(Btmp.currentY) > 0 && 
					(Btmp.currentX.find("*") != string::npos || Btmp.currentX.find("+") != string::npos))
					{
						finnishVHScore = getFinnishVHScore(myParam, correctY, Btmp.currentY, Btmp.jointY);
						//cout << "FVHScore: " << finnishVHScore << "\n";
					}
					if(myParam.turkishRH == true && Btmp.currentY.find_first_of("iIuU") != string::npos)
					{
						turkishRHScore = getTurkishRHScore(myParam, correctY, Btmp.currentY, Btmp.jointY);
						//cout << "Returned Score:\t" << turkishRHScore << endl;
					}
					//if(myParam.turkishVH == true && containsTurkishVowel(Btmp.currentY) && Btmp.currentX.compare(Btmp.currentY) != 0)
					if(myParam.turkishVH == true && containsTurkishVowel(Btmp.currentY) && 
					(Btmp.currentX.find("*") != string::npos || Btmp.currentX.find("+") != string::npos))
					//Turkish vowel harmony only concerns the suffixes
					{
						turkishVHScore = getTurkishVHScore(myParam, correctY, Btmp.currentY, Btmp.jointY);
					}

					if (myParam.WCInFilename != "")
					{
					 	string generated = "<w>" + join(Btmp.jointY,"","") + Btmp.currentY;
                                                removeSubString(generated, myParam.inChar); // remove inChar
                                                removeSubString(generated, "_");
                                                removeSubString(generated, "+");
                                                //cout << generated << endl;
						if(endOfWord)
						{
							generated += "<\\w>";
						}
						WCFeatureScore = getWCFeatureScore(myParam, WLCounts, generated);


					}	

                                        if (myParam.LMInFilename != "")
                                        {
                                                string generated = join(Btmp.jointY,"","") + Btmp.currentY;
                                                removeSubString(generated, myParam.inChar); // remove inChar
                                                removeSubString(generated, "_");
						removeSubString(generated, "+");	

                                                int wordLength = utf8::distance(generated.c_str(), generated.c_str() + generated.length()); 
						//wordLength += 1;//Account for word-start boundary
                                                if(endOfWord)
                                                    wordLength += 1; //And word-end boundary
                                                //cout << generated << endl;

 						double probability = getLMProbability(generated, wordLength, LMProbs, LMBackoff, maxLM, endOfWord);
						wordLength -= std::count(generated.begin(), generated.end(), '!');
						wordLength -= std::count(generated.begin(), generated.end(), '@');
						//wordLength -= std::count(generated.begin(), generated.end(), '+');

						
						if(wordLength != 0)
						{
                                                	LMFeatureScore = getLMFeatureScore(myParam, probability / wordLength);
						}
						else
						{
							LMFeatureScore = 0.0;
						}
						/*if(Btmp.currentY.compare("_") == 0)
						{
							LMFeatureScore = 0.0;
						}*/
			
						/*cout << "BEAM " << i << " " << k << " " << unAlignedX.size() <<  endl;
                                                cout << "WORD: "<< generated << endl;
                                                cout << "PROB: " << probability << endl;
                                               	cout << "NPROB: " << probability / wordLength << endl;
						cout << "LENGTH: " << wordLength << endl;
						*/

                                        }

					//Btmp.LMScore = LMFeatureScore;	
					Btmp.score = localScore + copyScore + WCFeatureScore + LMFeatureScore - oldLMScore + transScore + jointMScore + jointFMScore + finnishVHScore + turkishVHScore + turkishRHScore;
					// no previous joint-gram history //

					beamTable[i+k-1].push_back(Btmp);
				}
				else // consider previous decision
				{
					for (D_btable::iterator p_pos = beamTable[i-1].begin(); p_pos != beamTable[i-1].end() ; p_pos++)
					{
						//transScore = getOrderFeatureScore(myParam, featureStr, p_pos->currentY, *c_pos);

						btable Btmp;
						
						Btmp.phraseSize = p_pos->phraseSize;
						Btmp.phraseSize.push_back(k);

						Btmp.currentX = join(unAlignedX, i, i + k, "", "");
						Btmp.currentY = *c_pos;
						
						Btmp.jointX = p_pos->jointX;
						Btmp.jointY = p_pos->jointY;
						Btmp.jointX.push_back(p_pos->currentX);
						Btmp.jointY.push_back(p_pos->currentY);
		
						transScore = getHigherOrderFeatureScore(myParam, featureStr, Btmp.jointY, *c_pos);
						jointMScore = getJointGramFeatureScore(myParam, Btmp.jointX, Btmp.jointY, Btmp.currentX, Btmp.currentY);
						if(myParam.finnishVH == true && containsFinnishVowel(Btmp.currentY) > 0 && 
						(Btmp.currentX.find("*") != string::npos || Btmp.currentX.find("+") != string::npos))
						{
							finnishVHScore = getFinnishVHScore(myParam, correctY, Btmp.currentY, Btmp.jointY);
							//cout << "FVHScore: " << finnishVHScore << "\n";
						}	
						
						if(myParam.turkishRH == true && Btmp.currentY.find_first_of("iIUu") != string::npos)
						{
							turkishRHScore = getTurkishRHScore(myParam, correctY, Btmp.currentY, Btmp.jointY);
							//cout << "Returned Score:\t" << turkishRHScore << endl;
						}
						//if(myParam.turkishVH == true && containsTurkishVowel(Btmp.currentY) && Btmp.currentX.compare(Btmp.currentY) != 0)
						if(myParam.turkishVH == true && containsTurkishVowel(Btmp.currentY) && 
						(Btmp.currentX.find("*") != string::npos || Btmp.currentX.find("+") != string::npos))
						//Turkish vowel harmony only concerns the suffixes
						{
							turkishVHScore = getTurkishVHScore(myParam, correctY, Btmp.currentY, Btmp.jointY);
						}	
                                       		if (myParam.WCInFilename != "")
                                        	{
                                        	        string generated =  "<w>" + join(Btmp.jointY,"","") + Btmp.currentY;
                                        	        removeSubString(generated, myParam.inChar); // remove inChar
                                        	        removeSubString(generated, "_");
                                        	        removeSubString(generated, "+");

                                                        string oldGenerated = "<w>" + join(Btmp.jointY, "", "");//Have to remove old score, or LM gets too much weight
                                                        removeSubString(oldGenerated, myParam.inChar); // remove inChar
                                                        removeSubString(oldGenerated, "_");
                                                        removeSubString(oldGenerated, "+");

                                        	        //cout << generated << endl;
                                        	        if(endOfWord)
                                        	        {
                                        	                generated += "<\\w>";
                                        	        }
                                        	        WCFeatureScore = getWCFeatureScore(myParam, WLCounts, generated);
							oldWCScore = getWCFeatureScore(myParam, WLCounts, oldGenerated);

                                        	}

						if (myParam.LMInFilename != "")
                                                {
                                                        string generated = join(Btmp.jointY,"","") + Btmp.currentY;
                                                        removeSubString(generated, myParam.inChar); // remove inChar
                                                        removeSubString(generated, "_");
							removeSubString(generated, "+");
	
							string oldGenerated = join(Btmp.jointY, "", "");//Have to remove old score, or LM gets too much weight
                                                        removeSubString(oldGenerated, myParam.inChar); // remove inChar
                                                        removeSubString(oldGenerated, "_");
							removeSubString(oldGenerated, "+");

	                                                int wordLength = utf8::distance(generated.c_str(), generated.c_str() + generated.length()); 
						//	wordLength += 1;//Account for word-start boundary
                	                                if(endOfWord)
                        	                            wordLength += 1; //And word-end boundary

							wordLength -= std::count(generated.begin(), generated.end(), '!');//Word split characters
							wordLength -= std::count(generated.begin(), generated.end(), '@');
							//wordLength -= std::count(generated.begin(), generated.end(), '+');



							int oldWordLength =  utf8::distance(oldGenerated.c_str(), oldGenerated.c_str() + oldGenerated.length());
							oldWordLength -= std::count(oldGenerated.begin(), oldGenerated.end(), '!');
							oldWordLength -= std::count(oldGenerated.begin(), oldGenerated.end(), '@');
							//oldWordLength -= std::count(oldGenerated.begin(), oldGenerated.end(), '+');


                                                        double probability = getLMProbability(generated, wordLength, LMProbs, LMBackoff, maxLM, endOfWord);

                                                        double oldProbability = getLMProbability(oldGenerated, oldWordLength, LMProbs, LMBackoff, maxLM, false);
								
                                                        if(wordLength != 0)
							{
								LMFeatureScore = getLMFeatureScore(myParam, probability / wordLength);
							}
							else
							{
								LMFeatureScore = 0.0;
							}
							/*if(Btmp.currentY.compare("_") == 0)
							{
								LMFeatureScore = 0.0;
							}*/

	
                                                        if(oldWordLength != 0)
							{
								oldLMScore = getLMFeatureScore(myParam, oldProbability / oldWordLength);
							}
							else
							{
								oldLMScore = 0.0;
							}

						/*
							cout << "BEAM " << i << " " << k << " " << unAlignedX.size() <<  endl;
                                                	cout << "WORD: " << generated << endl;
							cout << "PROB: " << probability << endl; 
                                                	cout << "NPROB: " << probability / wordLength << endl;
							cout << "LENGTH: " << wordLength << endl;
                                                	cout << "OLDWORD: " << oldGenerated << endl;
							cout << "OLDPROB: " << oldProbability << endl; 
                                                	cout << "OLDNPROB: " << oldProbability / oldWordLength << endl;
							cout << "OLDLENGTH: " << oldWordLength << endl;
						*/		
							
                                                        //cout << i << " " << generated << endl;
                                                        //cout << probability / wordLength << endl;
                                                }
						//oldLMScore = Btmp.LMScore;
						//Btmp.LMScore = LMFeatureScore;
                                                Btmp.score = p_pos->score + transScore + localScore + copyScore + WCFeatureScore - oldWCScore + LMFeatureScore - oldLMScore + jointMScore + jointFMScore + finnishVHScore + turkishVHScore + turkishRHScore;



						beamTable[i+k-1].push_back(Btmp);	
					}
				}
			}
		}

		// reducing size to max(n-best,beam_size) //
		D_btable Dbtmp;
		int max_beamSize = max(myParam.nBest, myParam.beamSize);

		Dbtmp = beamTable[i];
		if (Dbtmp.size() > max_beamSize)
		{
			D_btable Dbtmp_sort(max_beamSize);
			partial_sort_copy(Dbtmp.begin(), Dbtmp.end(), Dbtmp_sort.begin(), Dbtmp_sort.end(), DbSortedFn);
			beamTable[i] = Dbtmp_sort;
		}
	}


	// sort score //
	sort(beamTable[unAlignedX.size() - 1].begin(), beamTable[unAlignedX.size()-1].end(), DbSortedFn);	
	
	// backtracking - nbest
	for (int k =0 ; (k < myParam.nBest) && (beamTable[unAlignedX.size() - 1].size() > 0) ; k++)
	{
		vector_str tempBestOutput;
		vector_str tempBestAlignedX;
		vector_2str tempBestFeatureStr;

		// current max element 
		scoreNbest.push_back(beamTable[unAlignedX.size() - 1][0].score);
		
		// current best output
		tempBestOutput = beamTable[unAlignedX.size() - 1][0].jointY;
		tempBestOutput.push_back(beamTable[unAlignedX.size() - 1][0].currentY);

		// current best structure
		tempBestAlignedX = beamTable[unAlignedX.size() - 1][0].jointX;
		tempBestAlignedX.push_back(beamTable[unAlignedX.size() - 1][0].currentX);

		// current best local features
		int sum_phraseSize=0;
		for (int i = 0; i < beamTable[unAlignedX.size() - 1][0].phraseSize.size(); i++)
		{
			int c_phraseSize = beamTable[unAlignedX.size() - 1][0].phraseSize[i];
			tempBestFeatureStr.push_back(allFeatureStr[sum_phraseSize][c_phraseSize]);

			sum_phraseSize += c_phraseSize;
		}
		
		nBestOutput.push_back(tempBestOutput);
		alignedXnBest.push_back(tempBestAlignedX);
		featureNbest.push_back(tempBestFeatureStr);

		// remove the top from the chart //

		beamTable[unAlignedX.size() -1].erase(beamTable[unAlignedX.size() - 1].begin());
	}
	
	return nBestOutput;
}

vector_2str phraseModel::phrasalDecoder(param &myParam, vector_str unAlignedX, 
										vector_2str &alignedXnBest, vector_3str &featureNbest,
										vector<double> &scoreNbest)
{
	//GN: Unable to make fix to markov order if not using phrasalDecoder, as table doesn't store previous X, Y;
	vector_2str nBestOutput;
	vector_3str allFeatureStr;
	D_hash_string_qtable Q;

	Q.resize(unAlignedX.size());
	allFeatureStr.resize(unAlignedX.size());

	// go over state //
	for (int i = 0; i < unAlignedX.size(); i++)
	{
		allFeatureStr[i].resize(myParam.maxX + 1);
		// go over phrases //
		for (int k = 1; k <= myParam.maxX ; k++)
		{
			if ((i + k) > unAlignedX.size())
			{
				continue;
			}

			int lPosMin = i - myParam.contextSize;
			if (lPosMin < 0)
			{
				lPosMin = 0;
			}

			int rPosMax = i + k + myParam.contextSize;
			if (rPosMax > unAlignedX.size())
			{
				rPosMax = unAlignedX.size();
			}
			vector_str focus(unAlignedX.begin() + i , unAlignedX.begin() + i + k);
			vector_str lContext(unAlignedX.begin() + lPosMin, unAlignedX.begin() + i);
			vector_str rContext(unAlignedX.begin() + i + k, unAlignedX.begin() + rPosMax);
			
			vector<string> allCandidate = myAllPhoneme.getPhoneme(join(focus,"",""), true);

			// default case : add null when |phrase| = 1 and no candidate //
			// to ensure at least one source generates one target (null included)//
			if ((allCandidate.size() == 0 ) && (k == 1))
			{
				allCandidate.push_back("_");
	
				if(strpbrk(join(focus,"","").c_str(), "*") == 0)
				{
					allCandidate.push_back(join(focus,"","")); //Always allow a candidate to copy to itself.
				}
			}
			
			// skip any |phrase| > 1 and no candidate found //
			if (allCandidate.size() == 0)
			{
				continue;
			}

			// extract n-gram feature // 
			vector<string> featureStr;
			featureStr = ngramFeatureGen(myParam, lContext, focus, rContext);

			allFeatureStr[i][k] = featureStr; // keep k start from 1 (|phrase| >= 1)

			for (vector<string>::iterator c_pos = allCandidate.begin(); c_pos != allCandidate.end(); c_pos++)
			{	
				bool isCopy = ((*c_pos).compare(join(focus,"","")) == 0);
				double localScore = getLocalFeatureScore(myParam, featureStr, *c_pos);
				double copyScore = 0;
				double transScore;
			

				if(myParam.copyFeature)
				{
					copyScore = getCopyScore(myParam, isCopy, *c_pos);
				}
				// no previous decision yet //
				if (i == 0)
				{
					transScore = getOrderFeatureScore(myParam, featureStr, "", *c_pos);

					qtable Qtmp;

					Qtmp.score = localScore + copyScore + transScore;
					Qtmp.phraseSize = k;
					Qtmp.backTracking = "";
					Qtmp.backRanking = -1;
					Qtmp.backQ = -1;

					Q[i+k-1][*c_pos].push_back(Qtmp);
				}
				else // consider previous decision
				{
					for (hash_string_Dqtable::iterator p_pos = Q[i-1].begin(); p_pos != Q[i-1].end(); p_pos++)
					{
						transScore = getOrderFeatureScore(myParam, featureStr, p_pos->first, *c_pos);

						//// debug //
						//if (p_pos->second.size() > 10)
						//{
						//	int ssi = p_pos->second.size();
						//	string ssp = p_pos->first;
						//	D_qtable ssQ = p_pos->second;
						//}

						for (unsigned int r = 0 ; r < p_pos->second.size(); r++)
						{
							qtable Qtmp;

							Qtmp.score = p_pos->second[r].score + copyScore + transScore + localScore;
							Qtmp.phraseSize = k;
							Qtmp.backTracking  = p_pos->first;
							Qtmp.backRanking = r;
							Qtmp.backQ = i-1;
							
							Q[i+k-1][*c_pos].push_back(Qtmp);

						}	
					}
				}
			}
		}

		// BIG possible bug here .. Q[i] Vs. Q[i-1] .. why did it still work?
		//for (hash_string_Dqtable::iterator c_pos = Q[i].begin(); c_pos != Q[i-1].end(); c_pos++)
		for (hash_string_Dqtable::iterator c_pos = Q[i].begin(); c_pos != Q[i].end(); c_pos++)
		{
			// reducing size to n-best
			D_qtable Dqtmp;

			Dqtmp = Q[i][c_pos->first];

			if (Dqtmp.size() > myParam.nBest)
			{
				D_qtable Dqtmp_sort(myParam.nBest);
				partial_sort_copy(Dqtmp.begin(), Dqtmp.end(), Dqtmp_sort.begin(), Dqtmp_sort.end(), DqSortedFn);
				Q[i][c_pos->first] = Dqtmp_sort;
			}
		}
	}

	// sort score //
	for (hash_string_Dqtable::iterator pos = Q[unAlignedX.size() - 1].begin(); pos != Q[unAlignedX.size() - 1].end(); pos++)
	{
		sort(pos->second.begin(), pos->second.end(), DqSortedFn);
	}	
	
	// backtracking - nbest
	for (int k =0 ; (k < myParam.nBest) && (Q[unAlignedX.size() - 1].size() > 0) ; k++)
	{
		vector_str tempBestOutput;
		vector_str tempBestAlignedX;
		vector_2str tempBestFeatureStr;

		double max_score = -10e20;
		string max_pos = "";
		
		
		// find the max score //
		for (hash_string_Dqtable::iterator pos = Q[unAlignedX.size() - 1].begin(); pos != Q[unAlignedX.size() - 1].end(); pos++)
		{
			double score_candidate = pos->second[0].score;

			if (score_candidate > max_score)
			{
				max_score = score_candidate;
				max_pos = pos->first;
			}
			else if (score_candidate == max_score)
			{
				if (pos->first > max_pos)
				{
					max_score = score_candidate;
					max_pos = pos->first;
				}
			}
		}

		if (max_score <= -10e20)
		{
			cout << "Can't find any candidate (buggy!!)" << endl;
			exit(-1);
		}

		scoreNbest.push_back(max_score);
		string last_element = max_pos;
		int last_rank = 0;
		int last_phraseSize = Q[unAlignedX.size() -1][last_element][last_rank].phraseSize;

		tempBestOutput.push_back(last_element);
		tempBestAlignedX.push_back(join(unAlignedX,unAlignedX.size() - last_phraseSize, unAlignedX.size(), "", ""));
		tempBestFeatureStr.push_back(allFeatureStr[unAlignedX.size() - last_phraseSize][last_phraseSize]);
		

		int i = unAlignedX.size() - 1;

		while (i > -1)
		{
			qtable Qtmp = Q[i][last_element][last_rank];
			string last_element_tmp = Qtmp.backTracking;
			int last_rank_tmp = Qtmp.backRanking;
			i = Qtmp.backQ;

			if (i > -1)
			{
				last_phraseSize = Q[i][last_element_tmp][last_rank_tmp].phraseSize;
				tempBestFeatureStr.push_back(allFeatureStr[i - last_phraseSize + 1][last_phraseSize]);
				tempBestAlignedX.push_back(join(unAlignedX,i - last_phraseSize + 1, i + 1, "",""));
				tempBestOutput.push_back(last_element_tmp);			
			}

			last_element = last_element_tmp;
			last_rank = last_rank_tmp;
		}

		reverse(tempBestOutput.begin(), tempBestOutput.end());
		reverse(tempBestAlignedX.begin(), tempBestAlignedX.end());
		reverse(tempBestFeatureStr.begin(), tempBestFeatureStr.end());

		nBestOutput.push_back(tempBestOutput);
		alignedXnBest.push_back(tempBestAlignedX);
		featureNbest.push_back(tempBestFeatureStr);

		// remove the top from the chart //

		D_qtable D_Qtmp;
		D_Qtmp = Q[unAlignedX.size()-1][max_pos];
		D_Qtmp.erase(D_Qtmp.begin());
		if (D_Qtmp.size() == 0)
		{
			Q[unAlignedX.size()-1].erase(max_pos);
		}
		else
		{
			Q[unAlignedX.size()-1][max_pos] = D_Qtmp;
		}
	}

	return nBestOutput;
}

double phraseModel::minEditDistance(vector<string> str1, vector<string> str2, string ignoreString)
{
	double distanceRate;
	double maxLength;
	double cost;

//	string str1,str2;

	vector<vector<double> > distance;

	if (ignoreString != "")
	{
		removeVectorElem(str1, ignoreString);
		removeVectorElem(str2, ignoreString);
	}

	//str1 = join(vstr1, "", ignoreString);
	//str2 = join(vstr2, "", ignoreString);

	//Initialize distance matrix
	distance.assign(str1.size()+1, vector<double>(str2.size()+1, 0));

	for (unsigned int i = 0; i < str1.size()+1; i++)
	{
		distance[i][0] = i;
	}
	for (unsigned int j = 0; j < str2.size()+1; j++)
	{
		distance[0][j] = j;
	}

	for (unsigned int i = 0; i < str1.size(); i++)
	{
		for (unsigned int j = 0; j < str2.size(); j++)
		{
			if (str1[i] == str2[j])
			{
				cost = 0;
			}
			else
			{
				cost = 1;
			}

			distance[i+1][j+1] = distance[i][j] + cost;

			if (distance[i+1][j+1] > (distance[i+1][j] + 1))
			{
				distance[i+1][j+1] = distance[i+1][j] + 1;
			}

			if (distance[i+1][j+1] > (distance[i][j+1] + 1))
			{
				distance[i+1][j+1] = distance[i][j+1] + 1;
			}
		}
	}

	distanceRate = distance[str1.size()][str2.size()];
	maxLength = (double)(max(str1.size(), str2.size()));
	distanceRate = distanceRate / maxLength;

	return distanceRate;

}

long phraseModel::my_feature_hash(string feaList, string phonemeTarget, hash_string_long *featureHash)
{
	long value;

	string key = feaList + "T:" + phonemeTarget;
	hash_string_long::iterator m_pos;

	m_pos = featureHash->find(key);
	if ( m_pos != featureHash->end() )
	{
		return m_pos->second;
	}
	else
	{
		value = featureHash->size() + 1;
		featureHash->insert(make_pair(key,value));
		return value;
	}
}


vector_2str phraseModel::genFeature(param &myParam, data dpoint)
{
	vector_2str output;

	int j = 0;
	for (int i = 0; i < dpoint.unAlignedX.size(); )
	{
		int k = dpoint.phraseSizeX[j];
		int lPosMin = i - myParam.contextSize;
		if (lPosMin < 0)
		{
			lPosMin = 0;
		}

		int rPosMax = i + k + myParam.contextSize;
		if (rPosMax > dpoint.unAlignedX.size())
		{
			rPosMax = dpoint.unAlignedX.size();
		}

		vector_str focus(dpoint.unAlignedX.begin() + i , dpoint.unAlignedX.begin() + i + k);
		vector_str lContext(dpoint.unAlignedX.begin() + lPosMin, dpoint.unAlignedX.begin() + i);
		vector_str rContext(dpoint.unAlignedX.begin() + i + k, dpoint.unAlignedX.begin() + rPosMax);

		if (! myParam.noContextFea)
		{
			output.push_back(ngramFeatureGen(myParam, lContext, focus, rContext));
		}

		i += k;
		j++;
	}
	
	return output;
}

void phraseModel::my_feature_hash_avg(param &myParam, vector_2str featureList, vector_str alignedTarget, 
											  hash_string_long *featureHash, double scaleAvg, map<long,double> &idAvg)
{

	// haven't implement this part to incorporate the jointMgram feature yet //
	for (unsigned int i = 0; i < alignedTarget.size(); i++)
	{

                for (vector_str::iterator feaListPos = featureList[i].begin(); feaListPos != featureList[i].end(); feaListPos++)
                {
                        //idSet.push_back(my_feature_hash(*feaListPos, alignedTarget[i], featureHash));
                        idAvg[my_feature_hash(*feaListPos, alignedTarget[i], featureHash)] += scaleAvg;
			string runningHistory = "";
                        if (myParam.linearChain)
                        {
				 int k = 0;
                               	 for(int j = 1; j <= myParam.markovOrder;)
				 {
					//cout << "HASH:CHAIN" + stringify(i - j - k);

					int x = i - j - k;
					if(x <= 0)
					{
                                       		idAvg[my_feature_hash(*feaListPos + "P:-" + stringify(j) + ":{" + runningHistory, alignedTarget[i], featureHash)] += scaleAvg;

						break;
					}
					else
					{
						if(myParam.ignoreNull && alignedTarget[x].compare("_") == 0)
						{
							k++;
						}
						else
						{
							runningHistory = alignedTarget[x] + runningHistory;
                                       			idAvg[my_feature_hash(*feaListPos + "P:-" + stringify(j) + ":" + runningHistory, alignedTarget[i], featureHash)] += scaleAvg;
							j++;
						}
					}
				}
                        }
                }

		if (myParam.markovOrder >= 1)
		{
			string runningHistory = "";
			int k = 0;
                        for(int j = 1; j <= myParam.markovOrder;)
			{
				//cout << "HASH:MARKOV" + stringify(i - j - k) + "\n";
				int x = i - j - k;

				if(x <= 0)
				{
                                       	idAvg[my_feature_hash("P:-" + stringify(j) + ":{" + runningHistory, alignedTarget[i], featureHash)] += scaleAvg;
					break;
				}
				else
				{
					if(myParam.ignoreNull && alignedTarget[x].compare("_") == 0)
					{
						k++;
					}
					else
					{
						runningHistory = alignedTarget[x] + runningHistory;
                                       		idAvg[my_feature_hash("P:-" + stringify(j) + ":" + runningHistory, alignedTarget[i], featureHash)] += scaleAvg;
						j++;
					}
				}//\else
			}//\for
		}//\if

	}//\for	


		// 1st order markov features
		//if (myParam.markovOrder >= 1)
		//{
		//	if (i == 0)
		//	{
		//		//idSet.push_back(my_feature_hash("P:-1:",alignedTarget[i],featureHash));
		//		idAvg[my_feature_hash("P:-1:",alignedTarget[i],featureHash)] += scaleAvg;
                //              for (vector_str::iterator feaListPos = featureList[i].begin(); feaListPos != featureList[i].end(); feaListPos++)
                //                {
                //                    	//idSet.push_back(my_feature_hash(*feaListPos, alignedTarget[i], featureHash));
                //                       	//idAvg[my_feature_hash(*feaListPos, alignedTarget[i], featureHash)] += scaleAvg;
		//			if (myParam.linearChain)// && j == 1)
                //                        {
                //                               	idAvg[my_feature_hash(*feaListPos + "P:-1:" , "{" + alignedTarget[i], featureHash)] += scaleAvg;
                //                       	}
                //                 }//\for
		//
		//	}
		//	else
		//	{
                //      		string tmp = "";//join(alignedTarget, 0, i-1, "", "");
		//		for(int x = 0; x < i; x++)
		//		{
		//			if(alignedTarget[x].compare("_") != 0)
		//			{
		//				tmp += alignedTarget[x];
		//			}
		//		}
                //            	//tmp.erase(std::remove(tmp.begin(), tmp.end(), '_'), tmp.end());
		//		//GN: std::remove doesn't like UTF
                //               	int max_history = min<int>(myParam.markovOrder, tmp.length() + 1);
		//		if(max_history < myParam.markovOrder)
		//		{
		//			max_history++;
		//		}
		//
                //               	string runningHistory("");
		//		int k = i-1;
		//		for(int j = 1; j <= max_history && k >= -1; k--)
		//		{
		//			string current(alignedTarget[i]);
		//			string previous("");
		//			if(k >= 0)
		//			{
		//				previous = alignedTarget[k];
		//				//previous.erase(std::remove(previous.begin(), previous.end(), '_'), previous.end());
		//			}
		//			else
		//			{
		//				previous = "{";//This will occur if we pass the beginning of the word.
		//			}
		//			if(previous.compare("_") == 0)
		//			{
		//				//continue;
		//			}
		//			runningHistory = previous + runningHistory;
		//
		//			//idSet.push_back(my_feature_hash("P:-1:" + alignedTarget[i-1], alignedTarget[i], featureHash));
		//			idAvg[my_feature_hash("P:-" + stringify(j) + ":" + runningHistory, current, featureHash)] += scaleAvg;
		//
                //                       for (vector_str::iterator feaListPos = featureList[i].begin(); feaListPos != featureList[i].end(); feaListPos++)
                //                       {
                //                               	//idSet.push_back(my_feature_hash(*feaListPos, alignedTarget[i], featureHash));
                //                               	//idAvg[my_feature_hash(*feaListPos, alignedTarget[i], featureHash)] += scaleAvg;
		//
                //                               	if (myParam.linearChain)// && j == 1)
                //                               	{
                //                                        //idSet.push_back(my_feature_hash(*feaListPos + "P:-1:" + alignedTarget[i-1], alignedTarget[i], featureHash));
		//					idAvg[my_feature_hash(*feaListPos + "P:-" + stringify(j) + ":" + runningHistory, current, featureHash)] += scaleAvg;
		//				}
                //                     }//\for
                //                        
                //                        j++;
		//		}//\for
		//	}//\else
		//}//\if*/
	//}//\for


}

WORD *phraseModel::my_feature_hash_map_word(param &myParam, map<long,double> &idAvg, long max_words_doc)
{
	long wpos = 0;
	WORD *outWORD = (WORD *) my_malloc(sizeof(WORD) * (max_words_doc + 10));

	for(map<long,double>::iterator pos = idAvg.begin(); pos != idAvg.end(); )
	{
		outWORD[wpos].wnum = pos->first;
		outWORD[wpos].weight = pos->second;
		pos++;

		while ((pos != idAvg.end()) && (pos->first == outWORD[wpos].wnum))
		{
			outWORD[wpos].weight += pos->second;
			pos++;
		}
		wpos++;
	}

	outWORD[wpos].wnum = 0;
	outWORD[wpos].weight = 0;

	return outWORD;

}

WORD *phraseModel::my_feature_map_word(param &myParam, vector_2str featureList, vector_str alignedTarget, 
									   hash_string_long *featureHash, long max_words_doc, vector_str alignedSource, hash_string_double& WLCounts, hash_string_double& LMProbs, hash_string_double& LMBackoff, int maxLM)
{
	long wpos = 0;
	WORD *outWORD = (WORD *) my_malloc(sizeof(WORD) * (max_words_doc + 10));
	vector<long> idSet;

	vector_str jointX;
	vector_str jointY;
	string targetSoFar = "";
	
	for (unsigned int i = 0; i < alignedTarget.size(); i++)
	{
		targetSoFar += alignedTarget[i];
		if (featureList.size() > 0)
		{
			for(vector_str::iterator feaListPos = featureList[i].begin();feaListPos != featureList[i].end(); feaListPos++)
			{
				idSet.push_back(my_feature_hash(*feaListPos, alignedTarget[i], featureHash));
				string runningHistory = "";
                        	if (myParam.linearChain)
                        	{
					 int k = 0;
                        	       	 for(int j = 1; j <= myParam.markovOrder;)
					 {
						//cout << "MAP:CHAIN" + stringify(i - j - k) + "\n";
						int x = i - j - k;
						if(x <= 0)
						{
							idSet.push_back(my_feature_hash(*feaListPos + "P:-" + stringify(j) + ":{" + runningHistory, alignedTarget[i], featureHash));
							
							break;
						}
						else
						{
							if(myParam.ignoreNull && alignedTarget[x].compare("_") == 0)
							{
								k++;
							}
							else
							{
								runningHistory = alignedTarget[x] + runningHistory;
								idSet.push_back(my_feature_hash(*feaListPos + "P:-" + stringify(j) + ":" + runningHistory, alignedTarget[i], featureHash));
								
								j++;
							}
						}//\else
					}//\for
                        	}//\if
                	}//\for
		}//\if

				//idSet.push_back(my_feature_hash(*feaListPos, alignedTarget[i], featureHash));
				//if (myParam.markovOrder >= 1)
				//{
				//	if (i == 0)
				//	{
				//		idSet.push_back(my_feature_hash("P:-1:","{" + alignedTarget[i],featureHash));
				//	}
				//	else
				//	{
				//		string tmp = "";//join(alignedTarget, 0, i-1, "", "");
				//		for(int x = 0; x < i; x++)
				//		{
				//			if(alignedTarget[x].compare("_") != 0)
				//			{
				//				tmp += alignedTarget[x];
				//			}
				//		}
				//		//tmp.erase(std::remove(tmp.begin(), tmp.end(), '_'), tmp.end());
				//		int max_history = min<int>(myParam.markovOrder, tmp.length() + 1);
				//		if(max_history < myParam.markovOrder)
				//		{
				//			max_history++;
				//		}
				//		string runningHistory("");
				//		int k = i-1;
				//		for(int j = 1; j <= max_history && k >= -1; k--)
				//		{
				//			string current(alignedTarget[i]);
				//			string previous("");
				//			if(k >= 0)
				//			{
				//				previous = alignedTarget[k];
				//				//previous.erase(std::remove(previous.begin(), previous.end(), '_'), previous.end());
				//			}
				//			else
				//			{
				//				previous = "{";//This will occur if we pass the beginning of the word.
				//			}
				//			if(previous.compare("_") == 0)
				//			{
				//				//continue;//Bug may be here;
				//			}
				//			runningHistory = previous + runningHistory;
				//	
				//			idSet.push_back(my_feature_hash("P:-" + stringify(j) + ":" + runningHistory, current, featureHash));
                                //     			if (myParam.linearChain)// && j == 1)
                                //       			{
                                //       				//if (i == 0)
                                //      				//{
                                //              			//	idSet.push_back(my_feature_hash(*feaListPos + "P:-" + stringify(j) + ":", alignedTarget[i], featureHash));			 //
                                //      				//}
                                //       				//else
                                //       				//{
                                //     					idSet.push_back(my_feature_hash(*feaListPos + "P:-" + stringify(j) + ":" + runningHistory, alignedTarget[i], featureHash));
                                //       				//}
                                //                        
				//			}//\if
				//			j++;
				//		}//\for
				//	}//\else
				//}//\if
			//}//\for
		//}//\else
		//Copy Feature
		//If the operation is just copying string x to x, then add in the copy score
		//This feature will be weighted by the number of copy operations made for the word pair
		if(myParam.copyFeature)
		{
			if(alignedSource[i].compare(alignedTarget[i]) == 0)
			{
				idSet.push_back(my_feature_hash("COPY", "COPY", featureHash));
				//idSet.push_back(my_feature_hash("COPY", alignedSource[i], featureHash));
			}
		}

		if (myParam.WCInFilename != "")
		{
			double count = 0.0;
			string generated = targetSoFar;
                	::setlocale(LC_ALL, "");
        		std::transform(generated.begin(), generated.end(), generated.begin(), ::towlower);

			removeSubString(generated, myParam.inChar);
			removeSubString(generated, "_");
			removeSubString(generated, "+");
			bool endOfWord = (i == alignedTarget.size() - 1);
			if(endOfWord)
			{
				generated += "<\\w>";	
			}

			std::vector<std::string> parts;
        		std::size_t pos = 0, found;
			
        		//cout << sequence << endl;
        		
			while((found = generated.find_first_of("!@", pos)) != std::string::npos) {

                	        if(found - pos != 0)
                	        {
                	                parts.push_back("<w>" + generated.substr(pos, found - pos) + "<\\w>");//It should be okay to split with ASCII, because "!" and "@" are ASCII chars
                	        }
                        	pos = found+1;
                	}
        		parts.push_back("<w>" + generated.substr(pos));

        		for(int j = 0; j < parts.size(); j++)
        		{
        	        	if(WLCounts.find(parts[j]) == WLCounts.end())
        	        	{
        	        	        count = 0.0;
        	        	}
        	        	else
        	        	{
        	        	        count = WLCounts[parts[j]];
        	        	}
			}
			count /= parts.size();
	                string bin = getWCBin(count);
                	for(int j = atoi(bin.c_str()); j < 8; j++)
                	{
				//cout << "BIN: " << bin << endl;

                        	stringstream converter;
                        	converter << j;
                        	string jString = converter.str();
				//cout << "J " << j << endl;
				//cout << "JString " << jString << endl;
				//if(endOfWord)
				{
                        		idSet.push_back(my_feature_hash("WCBIN:" + jString, "WC", featureHash));
				}
                	}

		}
	
	         if (myParam.LMInFilename != "")
	         {
	
		        removeSubString(targetSoFar, myParam.inChar); // remove inChar
	                removeSubString(targetSoFar, "_");
			removeSubString(targetSoFar, "+");

	 		int wordLength = utf8::distance(targetSoFar.c_str(), targetSoFar.c_str() + targetSoFar.length());
			bool endOfWord = (i == alignedTarget.size() - 1);
    			if(endOfWord)
			{
				wordLength += 1;
			}

 			wordLength -= std::count(targetSoFar.begin(), targetSoFar.end(), '!');
                        wordLength -= std::count(targetSoFar.begin(), targetSoFar.end(), '@');
                        //wordLength -= std::count(targetSoFar.begin(), targetSoFar.end(), '+');



			double probability = getLMProbability(targetSoFar, wordLength, LMProbs, LMBackoff, maxLM, endOfWord);
			//cout << targetSoFar << " " << wordLength << " " << probability << " " << probability / wordLength << endl;
			string bin = getLMBin(probability / wordLength);
							/*
							cout << "HASH" << i << endl;
                                                	cout << "WORD: " << targetSoFar << endl;
							cout << "PROB: " << probability << endl; 
                                                	cout << "NPROB: " << probability / wordLength << endl;
							cout << "LENGTH: " << wordLength << endl;
							*/
			if(wordLength != 0)
			{

			     for(int j = atoi(bin.c_str()); j < 8; j++)
        		     {
             			stringstream converter;
             			converter << j;
             			string jString = converter.str();
             			idSet.push_back(my_feature_hash("LMBIN:" + jString, "LM", featureHash));
        		     }

				//idSet.push_back(my_feature_hash("LMBIN:" + bin, "LM", featureHash));
			}
                }


		if(myParam.finnishVH && containsFinnishVowel(alignedTarget[i]) > 0 &&
		(alignedSource[i].find("*") != string::npos || alignedSource[i].find("+") != string::npos))
		{
			idSet.push_back(my_feature_hash("FINVH", "FINVH", featureHash));
		}
		
		if(myParam.turkishRH && alignedTarget[i].find_first_of("iIUu") != string::npos)
		{
			idSet.push_back(my_feature_hash("TRKRH", "TRKRH", featureHash));
		}
		//if(myParam.turkishVH && containsTurkishVowel(alignedTarget[i]) && alignedSource[i].compare(alignedTarget[i]) != 0)
		if(myParam.turkishVH && containsTurkishVowel(alignedTarget[i]) && 
		(alignedSource[i].find("*") != string::npos || alignedSource[i].find("+") != string::npos))
		{
			idSet.push_back(my_feature_hash("TRKVH", "TRKVH", featureHash));
		}


		if (myParam.markovOrder >= 1)
		{
			string runningHistory = "";
			int k = 0;
                        for(int j = 1; j <= myParam.markovOrder;)
			{
				//cout << "MAP:MARKOV " + stringify(i) + " " + stringify(j) + " " + stringify(k) + " " + stringify(alignedTarget.size()) + "\n";
				int x = i - j - k;
				if(x <= 0)
				{
					//cout << "In special case" + stringify(x) + "\n";
					idSet.push_back(my_feature_hash("P:-" + stringify(j) + ":{" + runningHistory, alignedTarget[i], featureHash));
					break;
				}
				else
				{
					//cout << "Not in special case" + stringify(x) + "\n";
					if(myParam.ignoreNull && alignedTarget[x].compare("_") == 0)
					{
						k++;
					}
					else
					{
						runningHistory = alignedTarget[x] + runningHistory;
						idSet.push_back(my_feature_hash("P:-" + stringify(j) + ":" + runningHistory, alignedTarget[i], featureHash));
                                
						j++;
					}
				}//\else
			}//\for
		}//\if

		
		// 1st order markov features
		//if (myParam.markovOrder == 1)
		//{
		//	if (i == 0)
		//	{
		//		idSet.push_back(my_feature_hash("P:-1:",alignedTarget[i],featureHash));
		//	}
		//	else
		//	{
		//		idSet.push_back(my_feature_hash("P:-1:" + alignedTarget[i-1], alignedTarget[i], featureHash));
		//	}
		//}

		// jointMgram features
		if (myParam.jointMgram > 0)
		{
			int max_history = min<int>(myParam.jointMgram, (jointX.size() + 1));
			if(max_history < myParam.jointMgram)
			{
				max_history++;
			}
			// a unigram (currentX,currentY) //
			idSet.push_back(my_feature_hash("JL:0:0:JP:L:" + alignedSource[i], alignedTarget[i], featureHash));

			// get scores of looking back i history;  //
			for (int j = 1; j < max_history; j++)
			{
				for (int k = j; k > 0; k--)
				{
					string feaStrX = join(jointX, jointX.size() - j, jointX.size() - k + 1, "-", "");
					string feaStrY = join(jointY, jointY.size() - j, jointY.size() - k + 1, "-", "");
					if(j >= jointX.size())//ie, if we are moving past the boundary of the word...
					{
						feaStrX = "{" + feaStrX;
						feaStrY = "{" + feaStrY;
					}	

                                   
					idSet.push_back(my_feature_hash("JL:" + stringify(-j) + ":" + stringify(-k) + ":" + feaStrX + "JP:" + feaStrY + "L:" + alignedSource[i], 
						alignedTarget[i], featureHash));				
				}
			}

			jointX.push_back(alignedSource[i]);
			jointY.push_back(alignedTarget[i]);
		}

		// most likely to be correct only 1-2 situation .. , for now. :-(
		if (myParam.jointFMgram > 1)
		{
			int startPos = i + 1;
			int stopPos = i + 1 + myParam.jointFMgram - 1;
					
			if (startPos > alignedSource.size()) startPos = alignedSource.size();
			if (stopPos > alignedSource.size()) stopPos = alignedSource.size();

			vector_str xForward(alignedSource.begin() + startPos, alignedSource.begin() + stopPos);
			int max_forward = min<int>(myParam.jointFMgram, xForward.size() + 1);

			vector_str yCurrentCandidate;
			vector_str yPastCandidate;

			// too much thinking now: so, for now, it works only single token (no phrasal input) //
			for (int ii = 1; ii < max_forward; ii++)
			{
				yCurrentCandidate = myAllPhoneme.getPhoneme(xForward[ii-1], true);
				string feaStrX = join(xForward, 0, ii, "-" , "");
				string feaStrY;

				vector_str yKeepHistory;
				for (vector_str::iterator pos = yCurrentCandidate.begin() ; pos != yCurrentCandidate.end(); pos++)
				{
					if ( ii-1 > 0)
					{
						for (vector_str::iterator p_pos = yPastCandidate.begin() ; p_pos != yPastCandidate.end(); p_pos++)
						{
							feaStrY = *p_pos + "-" + *pos;
							yKeepHistory.push_back(feaStrY);

							idSet.push_back(my_feature_hash("JL:1:" + stringify(ii) + ":" + feaStrX + "JP:" + feaStrY + "L:" + alignedSource[i], 
								alignedTarget[i], featureHash));
						}
					}
					else
					{
						feaStrY = *pos;
						yKeepHistory.push_back(feaStrY);

						idSet.push_back(my_feature_hash("JL:1:" + stringify(ii) + ":" + feaStrX + "JP:" + feaStrY + "L:" + alignedSource[i], 
								alignedTarget[i], featureHash));
					}
				}
				yPastCandidate = yKeepHistory;
			}
		}
	}
	// convert idSet to WORD
	sort(idSet.begin(),idSet.end());
	for (vector<long>::iterator pos = idSet.begin(); pos != idSet.end(); )
	{
		outWORD[wpos].wnum = *pos;
		outWORD[wpos].weight = 1.0;
		pos++;

		while ((pos != idSet.end()) && (*pos == outWORD[wpos].wnum))
		{
			outWORD[wpos].weight += 1;
			pos++;
		}

		wpos++;
	}
	outWORD[wpos].wnum = 0;
	outWORD[wpos].weight = 0;
	return outWORD;
}

double phraseModel::cal_score_hash_avg(param &myParam, map<long,double> &idAvg)
{
	double score = 0;

	for (map<long,double>::iterator pos = idAvg.begin(); pos != idAvg.end(); pos++)
	{
		score += myWF.getFeature(my_feature_hash_retrieve(&featureHash, pos->first), myParam.atTesting) * pos->second;
	}

	return score;
}

double phraseModel::cal_score_candidate(param &myParam, vector_2str featureList, vector_str alignedTarget, vector_str alignedSource, hash_string_double& WLCounts, hash_string_double& LMProbs, hash_string_double& LMBackoff, int maxLM)
{
	double score = 0;

	vector_str jointX;
	vector_str jointY;

	string candidateX = "";
	string candidateY = "";
	string correctY = "";

	for (int i = 0; i < alignedTarget.size(); i++)
	{
		correctY += alignedTarget[i];
	}
	for (int i = 0; i < alignedTarget.size(); i++)
	{
		// make sure we have featureList. It's empty when we don't use contextFeature
		if (featureList.size() > 0)
		{
			score += getLocalFeatureScore(myParam, featureList[i], alignedTarget[i]);

			//if (i == 0)
			//{
			//	score += getOrderFeatureScore(myParam, featureList[i], "", alignedTarget[i]);
			//}
			//else
			//{
				vector_str tmp;
				//std::string tmp("");
				for(int k = 0; k < i; k++)
				{
					string tmpChar(alignedTarget[k]);
					//tmpChar.erase(std::remove(tmpChar.begin(), tmpChar.end(), '_'), tmpChar.end());
					if(tmpChar.compare("_") != 0)
					{
					      tmp.push_back(tmpChar);
					}
					//Is this necessary?				
				}
				//score += getOrderFeatureScore(myParam, featureList[i], alignedTarget[i-1], alignedTarget[i]);
				score += getHigherOrderFeatureScore(myParam, featureList[i], tmp, alignedTarget[i]);
			//}
		}
		
		if (myParam.copyFeature == true)
		{
			
			//If we want the copy feature, then add the copy score.
			//If isCopy is false, ie this is not a copy operation, then getCopyScore returns 0.		
	
			bool isCopy = (alignedSource[i].compare(alignedTarget[i]) == 0);
			score += getCopyScore(myParam, isCopy, alignedSource[i]);
		}

		if (myParam.WCInFilename != "")
		{
                        string generated = "<w>" + join(jointY,"","") + alignedTarget[i];
                        removeSubString(generated, myParam.inChar); // remove inChar
                        removeSubString(generated, "_");
                        removeSubString(generated, "+");


			string oldGenerated = "<w>" + join(jointY, "", "");//Have to remove old score, or LM gets too heavy of weight
                        removeSubString(oldGenerated, myParam.inChar); // remove inChar
                        removeSubString(oldGenerated, "_");
                        removeSubString(oldGenerated, "+");

			bool endOfWord = (i ==(alignedTarget.size() - 1));
                        if(endOfWord)
                        {
                              generated += "<\\w>";
                        }

			double WCScore = getWCFeatureScore(myParam, WLCounts, generated);
			double oldWCScore = getWCFeatureScore(myParam, WLCounts, oldGenerated);
			score -= oldWCScore;
                        score += WCScore;

		
		}

		if (myParam.LMInFilename != "")
                {
 			string generated = join(jointY,"","") + alignedTarget[i];
                        removeSubString(generated, myParam.inChar); // remove inChar
                        removeSubString(generated, "_");
			removeSubString(generated, "+");

                        string oldGenerated = join(jointY, "", "");//Have to remove old score, or LM gets too heavy of weight
                        removeSubString(oldGenerated, myParam.inChar); // remove inChar
                        removeSubString(oldGenerated, "_");
			removeSubString(oldGenerated, "+");

                        int oldWordLength =  utf8::distance(oldGenerated.c_str(), oldGenerated.c_str() + oldGenerated.length());
 			oldWordLength -= std::count(oldGenerated.begin(), oldGenerated.end(), '!');
                        oldWordLength -= std::count(oldGenerated.begin(), oldGenerated.end(), '@');
                        //oldWordLength -= std::count(oldGenerated.begin(), oldGenerated.end(), '+');

					
	
                        double oldProbability = getLMProbability(oldGenerated, oldWordLength, LMProbs, LMBackoff, maxLM, false);
			double oldLMScore = 0.0;
                       	if(oldWordLength != 0)
                       	{
                       	      oldLMScore = getLMFeatureScore(myParam, oldProbability / oldWordLength);
                       	}
                       	else
                       	{
                       	      oldLMScore = 0.0;
                       	}



			int wordLength = utf8::distance(generated.c_str(), generated.c_str() + generated.length());
			bool endOfWord = (i ==(alignedTarget.size() - 1));
                        if(endOfWord)
                        {   
				wordLength+=1; //Account for  word-end boundary
                        }                  

 			wordLength -= std::count(generated.begin(), generated.end(), '!');
                        wordLength -= std::count(generated.begin(), generated.end(), '@');
                        //wordLength -= std::count(generated.begin(), generated.end(), '+');



                        double probability = getLMProbability(generated, wordLength, LMProbs, LMBackoff, maxLM, endOfWord);//Start of word, so we can get prob of current character
                        double LMFeatureScore = getLMFeatureScore(myParam, probability / wordLength);
			if(wordLength == 0)
			{
				LMFeatureScore = 0.0;
			}
			/*if(alignedTarget[i].compare("_") == 0)
			{
				LMFeatureScore = 0.0;
			}*/
							/*
							cout << "CAL" << i << endl;
                                                	cout << "WORD: " << generated << endl;
							cout << "PROB: " << probability << endl; 
                                                	cout << "NPROB: " << probability / wordLength << endl;
							cout << "LENGTH: " << wordLength << endl;
							*/
			score -= oldLMScore;
			score += LMFeatureScore;
		}
                                          

		// if we include jointMgram feature, M > 0 
		if(myParam.finnishVH == true && containsFinnishVowel(alignedTarget[i]) > 0 &&
		(alignedSource[i].find("*") != string::npos || alignedSource[i].find("+") != string::npos))
		{
			score += getFinnishVHScore(myParam, correctY, alignedTarget[i], jointY);
			//cout << "Calculating score...\n";
			//cout << alignedSource[i] << "\n";
		}
		if(myParam.turkishRH == true && alignedTarget[i].find_first_of("iIUu") != string::npos)
		{
			double turkishRHScore = 0.0;
			turkishRHScore = getTurkishRHScore(myParam, correctY, alignedTarget[i], jointY);
			score += turkishRHScore;//getTurkishRHScore(myParam, correctY, alignedTarget[i], jointY);
			//cout << "Returned Score:\t" << turkishRHScore << endl;

		}
		//if(myParam.turkishVH && containsTurkishVowel(alignedTarget[i]) && alignedSource[i].compare(alignedTarget[i]) != 0)
		if(myParam.turkishVH == true && containsTurkishVowel(alignedTarget[i]) && 
		(alignedSource[i].find("*") != string::npos || alignedSource[i].find("*") != string::npos))
		{
			score += getTurkishVHScore(myParam, correctY, alignedTarget[i], jointY);
		}

		if (myParam.jointMgram > 0)
		{
			
			score += getJointGramFeatureScore(myParam, jointX, jointY, alignedSource[i], alignedTarget[i]);
			
			jointX.push_back(alignedSource[i]);
			jointY.push_back(alignedTarget[i]);
			candidateX += alignedSource[i];
			candidateY += alignedTarget[i];
		}

		// most likely to be correct only 1-2 situation .. , for now. :-(
		if (myParam.jointFMgram > 1)
		{
			int startPos = i + 1;
			int stopPos = i + 1 + myParam.jointFMgram - 1;
					
			if (startPos > alignedSource.size()) startPos = alignedSource.size();
			if (stopPos > alignedSource.size()) stopPos = alignedSource.size();

			vector_str xForward(alignedSource.begin() + startPos, alignedSource.begin() + stopPos);

			score += getJointForwardGramFeatureScore(myParam, alignedSource[i], alignedTarget[i], xForward);
		}
	}

	//cout << "Candidate: " << candidateX << "\t" << candidateY << "\t" << score << "\n"; 
	return score;
}

string phraseModel::my_feature_hash_retrieve(hash_string_long *featureHash, long value)
{
	hash_string_long::iterator m_pos;
	
	m_pos = find_if(featureHash->begin(), featureHash->end(), value_equals<string, long>(value));
	
	if (m_pos != featureHash->end())
	{
		return m_pos->first;
	}
	else
	{
		cout << "ERROR: Can't find matching feature given map value: " << value << endl << endl;
		exit(-1);
	}
}

void phraseModel::readMaxPhraseSize(param &myParam, string filename)
{
	ifstream FILE;
	FILE.open(filename.c_str());

	if (! FILE)
	{
		cerr << "ERROR: Can't open file : " << filename << endl;
	}
	else
	{
		string line;
		getline(FILE,line);

		myParam.maxX = convertTo<int>(line);
	}
	FILE.close();
}
void phraseModel::writeMaxPhraseSize(param &myParam, string filename)
{
	ofstream FILE;

	FILE.open(filename.c_str(),ios_base::trunc);

	if (! FILE)
	{
		cerr << "ERROR: Can't write file : " << filename << endl;
		exit(-1);
	}
	else
	{
		FILE << myParam.maxX << endl;
	}
	FILE.close();
}

//void phraseModel::dataToUnique(param &myParam, vector<data> &inData, hash_string_vData &outUniqueData)
//{
//	// go over data //
//
//	for (long int i = 0; i < inData.size(); i++)
//	{
//		string unAlignedWord = join(inData[i].unAlignedX, "", "");
//		hash_string_vData::iterator it = outUniqueData.find(unAlignedWord);
//
//		if (it != outUniqueData.end())
//		{
//			vector<data> it_vData;
//
//			it_vData = outUniqueData[unAlignedWord];
//			it_vData.push_back(inData[i]);
//
//			outUniqueData[unAlignedWord] = it_vData;
//		}
//		else
//		{
//			vector<data> it_vData;
//			it_vData.push_back(inData[i]);
//
//			outUniqueData[unAlignedWord] = it_vData;
//		}
//	}
//}


void phraseModel::training(param& myParam)
{
	hash_string_vData trainUnique, devUnique;
	unsigned long actual_train, actual_dev;
        hash_string_double LMProbs;
        hash_string_double LMBackoff;
	hash_string_double WLCounts;
        int maxLM = 0;


	cout << endl << "Training starts " << endl;

	cout << "Read training file" << endl;
	readingAlignedFile(myParam, myParam.trainingFile, trainUnique, false);
	cout << "Max source phrase size : " << myParam.maxX << endl;
	cout << "Total number of unique training instances : " << trainUnique.size() << endl;

	// Initialize limited source-target set //
	initialize(myParam, trainUnique);

	if (myParam.devFile != "")
	{
		cout << "Read dev file" << endl;
		readingAlignedFile(myParam, myParam.devFile, devUnique);
		cout << "Total number of unique dev instances : " << devUnique.size() << endl;
	}

        if (myParam.LMInFilename != "")
        {
                readLMFile(myParam, myParam.LMInFilename, LMProbs, LMBackoff, maxLM);
        }
	if (myParam.WCInFilename != "")
	{
		readWCFile(myParam, myParam.WCInFilename, WLCounts);
	}


	int iter = 0;
	bool stillTrain = true;
	double error;
	double allPhonemeTrain;
	double p_error = 10e6; //initialized error history
	double p_error_train = 10e6; //initialized error history for training
	
	while (stillTrain)
	{
		iter++;
		error = 0;
		allPhonemeTrain = 0;
		actual_train = 0;

		// at training time 
		myParam.atTesting = false;

		cout << endl << "Iteration : " << iter << endl;

		// train over all training data // 
		for (hash_string_vData::iterator train_pos = trainUnique.begin(); train_pos != trainUnique.end(); train_pos++)
		{
			vector_2str nBestAnswer;
			vector_2str alignedXnBest;
			vector_3str featureNbest;
			vector<double> scoreNbest;
			vector<double> nBestPER;

			actual_train++;

			//dev
			//cout << actual_train << endl;
			myParam.maxCandidateEach = 0;
			if (myParam.useBeam)
			{
                                nBestAnswer = phrasalDecoder_beam(myParam, train_pos->second.mydata[0].unAlignedX, train_pos->second.mydata[0].unAlignedY, alignedXnBest, featureNbest, scoreNbest, WLCounts, LMProbs, LMBackoff, maxLM);

				//nBestAnswer = phrasalDecoder_beam(myParam, train_pos->second.mydata[0].unAlignedX, train_pos->second.mydata[0].unAlignedY, alignedXnBest, featureNbest, scoreNbest);
			}
			else
			{
				nBestAnswer = phrasalDecoder(myParam, train_pos->second.mydata[0].unAlignedX, alignedXnBest, featureNbest, scoreNbest);
			}

			dataplus multipleRefs = train_pos->second;

			vector<double>  allNbestPER(nBestAnswer.size());
			vector<int> refForNBest(nBestAnswer.size());

			//cout << actual_train << endl;
			// calculate PER for nBest answers respecting to multiple answers
			for (int nbi =0; nbi < nBestAnswer.size(); nbi++)
			{
				double PERjudge; 
				
				if (myParam.alignLoss == "minL")
				{
					PERjudge = 1e9;
				}
				else if (myParam.alignLoss == "maxL")
				{
					PERjudge = -1e9;
				}
				else
				{
					PERjudge = 0;
				}

				if (myParam.alignLoss == "mulA")
				{
					// all y are the same so, it shouldn't give difference in loss(y, y') //
					double pos_PER = minEditDistance(multipleRefs.mydata[0].alignedY, nBestAnswer[nbi], "_");

					allNbestPER[nbi] = pos_PER; 
					refForNBest[nbi] = 0; // just default, shouldn't use this // 

				}
				else if ((myParam.alignLoss != "minS") && (myParam.alignLoss != "maxS"))
				{
					int takingPos = 0;
					double sumAlignScore = 0;
					// go over multiple goal references //
					for (int rti = 0; rti < multipleRefs.mydata.size(); rti++)
					{
						double pos_PER = minEditDistance(multipleRefs.mydata[rti].alignedY, nBestAnswer[nbi], "_");
						/*std::string result = "";;
						std::string result2 = "";
						for(std::vector<std::string>::const_iterator i = multipleRefs.mydata[refForNBest[nbi]].alignedY.begin(); i != multipleRefs.mydata[refForNBest[nbi]].alignedY.end(); ++i)
						{
							result += *i;
						}

						for(std::vector<std::string>::const_iterator i = nBestAnswer[nbi].begin(); i != nBestAnswer[nbi].end(); ++i)
						{
							result2 += *i;
						}

						cout << "Correct: " << result << "\n";
						cout << "Prediction: " << result2 << "\n";
						cout << "Error: " << pos_PER << "\n";
						*/
						if (myParam.alignLoss == "minL")
						{
							if (pos_PER < PERjudge)
							{
								PERjudge = pos_PER;
								takingPos = rti;
								sumAlignScore = 1.0;
							}
						}
						else if (myParam.alignLoss == "maxL")
						{
							if (pos_PER > PERjudge)
							{
								PERjudge = pos_PER;
								takingPos = rti;
								sumAlignScore = 1.0;
							}
						}
						else if (myParam.alignLoss == "avgL")
						{
							PERjudge += pos_PER;
							sumAlignScore += 1.0;
						}
						else if (myParam.alignLoss == "ascL")
						{
							PERjudge += (1.0 / multipleRefs.mydata[rti].alignScore) * pos_PER;
							sumAlignScore += (1.0 / multipleRefs.mydata[rti].alignScore);
						}
						else if (myParam.alignLoss == "rakL")
						{
							PERjudge += (1.0 / multipleRefs.mydata[rti].alignRank) * pos_PER;
							sumAlignScore += (1.0 / multipleRefs.mydata[rti].alignRank);
						}
						else
						{
							cerr << "Can't file  " + myParam.alignLoss + " alignLoss handler" << endl; 
							exit(-1);
						}
					}
					allNbestPER[nbi] = (PERjudge / sumAlignScore);
					refForNBest[nbi] = (takingPos);
				}
			}

			// calculate max_words_dos for SVM feature creation //
			long max_words_doc = 0;
			long max_num_features =  (myParam.nGram + 1) * (myParam.nGram + 1) / 2 + myParam.copyFeature + myParam.finnishVH + myParam.turkishVH + myParam.turkishRH; //Add in an extra feature if copyFeature is true
                        if (myParam.LMInFilename != "")
                        {
                                max_num_features += 7; //Seven bins
                        }
		
			if (myParam.WCInFilename != "")
			{
				max_num_features += 8; //Eight bins
			}

			max_words_doc += (max_num_features + (train_pos->second.mydata[0].unAlignedX.size() + 1) * myParam.markovOrder+ ((myParam.jointMgram) * (myParam.jointMgram))) * 
				(train_pos->second.mydata[0].unAlignedX.size() + 2);

			if (myParam.jointFMgram > 1)
			{
				max_words_doc += long (pow(myParam.maxCandidateEach, (myParam.jointFMgram - 1))) * 
					(train_pos->second.mydata[0].unAlignedX.size() + 2);
			}

			if (myParam.linearChain)
			{
				max_words_doc += ( max_num_features + 1 ) * (train_pos->second.mydata[0].unAlignedX.size() + 2);
			}

			// SVM feature vectors, docs, rhs definitions //
			WORD *xi_yi;

			DOC **docs;
			double *rhs;

			// vector xi_yi //
			hash_string_SVECTOR vector_xi_yi;

			// score of yi //
			hash_string_double score_yi;

			// clear a mapping between featureString <-> featureSVM //
			featureHash.clear();

			// finding score of correct yi , vector_xi_yi
			if ((myParam.alignLoss == "minL") || (myParam.alignLoss == "maxL"))
			{
				for (int nbi = 0; nbi < nBestAnswer.size() ; nbi++)
				{
					// if it doesn't have score for yi, calculate; otherwise skip //
					// indexed by refForNBest[i] //
					if (score_yi.find(stringify(refForNBest[nbi])) == score_yi.end())
					{
						vector_2str feaVector = genFeature(myParam, multipleRefs.mydata[refForNBest[nbi]]);
						double score = cal_score_candidate(myParam, feaVector, multipleRefs.mydata[refForNBest[nbi]].alignedY, multipleRefs.mydata[refForNBest[nbi]].alignedX, WLCounts, LMProbs, LMBackoff, maxLM);

						score_yi[stringify(refForNBest[nbi])] = score;

						/*std::string result = "";;
						std::string result2 = "";
						for(std::vector<std::string>::const_iterator i = multipleRefs.mydata[refForNBest[nbi]].alignedY.begin(); i != multipleRefs.mydata[refForNBest[nbi]].alignedY.end(); ++i)
						{
							result += *i;
						}

						for(std::vector<std::string>::const_iterator i = multipleRefs.mydata[refForNBest[nbi]].alignedX.begin(); i != multipleRefs.mydata[refForNBest[nbi]].alignedX.end(); ++i)
						{
							result2 += *i;
						}
						std::ostringstream ss;
						ss << score;
						cout << "Prediction: " << result2 << "\t" << result << " " << ss.str() << "\n";
						*/
						xi_yi = my_feature_map_word(myParam, feaVector, multipleRefs.mydata[refForNBest[nbi]].alignedY, &featureHash, max_words_doc, multipleRefs.mydata[refForNBest[nbi]].alignedX, WLCounts, LMProbs, LMBackoff, maxLM);
						// vector_xi_yi indexed by refForNBest[i] //
						vector_xi_yi[stringify(refForNBest[nbi])] = create_svector(xi_yi,"",1);				
						free(xi_yi);
					}
				}
			} else if ((myParam.alignLoss == "minS") || (myParam.alignLoss == "maxS"))
			{
				hash_string_double::iterator maxMinElement;
				
				for (int rti = 0; rti < multipleRefs.mydata.size(); rti++)
				{
					vector_2str feaVector = genFeature(myParam, multipleRefs.mydata[rti]);
					double score = cal_score_candidate(myParam, feaVector, multipleRefs.mydata[rti].alignedY, multipleRefs.mydata[rti].alignedX, WLCounts, LMProbs, LMBackoff, maxLM);

					score_yi[stringify(rti)] = score;
				}
				
				// finding the max or min score //
				if (myParam.alignLoss == "minS"){
					maxMinElement = min_element(score_yi.begin(), score_yi.end());
				}
				else
				{
					maxMinElement = max_element(score_yi.begin(), score_yi.end());
				}

				vector_2str feaVector = genFeature(myParam, multipleRefs.mydata[convertTo<int>(maxMinElement->first)]);
				xi_yi = my_feature_map_word(myParam, feaVector, multipleRefs.mydata[convertTo<int>(maxMinElement->first)].alignedY, &featureHash, max_words_doc, multipleRefs.mydata[convertTo<int>(maxMinElement->first)].alignedX, WLCounts, LMProbs, LMBackoff, maxLM);
				vector_xi_yi[maxMinElement->first] = create_svector(xi_yi,"",1);				
				free(xi_yi);

				// go over nbest to assign allNbestPER, refForNBest, vector_xi_yi
				for (int nbi = 0; nbi < nBestAnswer.size(); nbi++)
				{
					refForNBest[nbi] = convertTo<int>(maxMinElement->first);
					allNbestPER[nbi] = minEditDistance(multipleRefs.mydata[refForNBest[nbi]].alignedY, nBestAnswer[nbi], "_");
					
				}
			} else if ((myParam.alignLoss == "avgL") || (myParam.alignLoss == "ascL") || (myParam.alignLoss == "rakL") )
			{
				// CHECK THIS PORTION .. possible bugs , cal_score_hash_avg and featureHash //

				// if it doesn't contain any before //
				if (multipleRefs.idAvg.size() == 0)
				{
					// calculate the average vector //
					map<long,double> idAvg;

					// calculate denominator // 
					double sumAlignScore = 0;
					for (int rti = 0; rti < multipleRefs.mydata.size(); rti++)
					{
						if (myParam.alignLoss == "avgL")
						{
							sumAlignScore += 1.0;
						}
						else if (myParam.alignLoss == "ascL")
						{
							sumAlignScore += (1.0 / multipleRefs.mydata[rti].alignScore);
						}
						else if (myParam.alignLoss == "rakL")
						{
							sumAlignScore += (1.0 / multipleRefs.mydata[rti].alignRank);
						}
						else
						{
							cerr << "Can't file  " + myParam.alignLoss + " alignLoss handler" << endl; 
							exit(-1);
						}
					}

					for (int rti = 0; rti < multipleRefs.mydata.size(); rti++)
					{	
						vector_2str feaVector = genFeature(myParam, multipleRefs.mydata[rti]);
						
						double scaleAvg;
						if (myParam.alignLoss == "avgL")
						{
							scaleAvg = 1.0;
						}
						else if (myParam.alignLoss == "ascL")
						{
							scaleAvg = (1.0 / multipleRefs.mydata[rti].alignScore);
						}
						else if (myParam.alignLoss == "rakL")
						{
							scaleAvg = (1.0 / multipleRefs.mydata[rti].alignRank);
						}
						else
						{
							cerr << "Can't file  " + myParam.alignLoss + " alignLoss handler" << endl; 
							exit(-1);
						}

						scaleAvg /= sumAlignScore;
						my_feature_hash_avg(myParam, feaVector, multipleRefs.mydata[rti].alignedY, &featureHash, scaleAvg, idAvg);
					}
					// get the average idAvg to create the vector of xi_yi and score 
					xi_yi = my_feature_hash_map_word(myParam, idAvg, max_words_doc);
					vector_xi_yi["avg"] = create_svector(xi_yi,"",1);		// average vector //	
					free(xi_yi);

					// save average vector back to xi_yi //
					// can't save the idAvg unless we keep featureHash .. 
					// but keeping featureHash all the way is very expensive 
					
					//train_pos->second.idAvg = idAvg;

					score_yi["avg"] = cal_score_hash_avg(myParam, idAvg);
				}
				else
				{
					// when we have the vector xi_yi already //
					// calculate score //

					xi_yi = my_feature_hash_map_word(myParam, multipleRefs.idAvg, max_words_doc);
					vector_xi_yi["avg"] = create_svector(xi_yi,"",1);
					free(xi_yi);

					score_yi["avg"] = cal_score_hash_avg(myParam, multipleRefs.idAvg);
				}
			}
			else if (myParam.alignLoss == "mulA")
			{
				for (int rti = 0; rti < multipleRefs.mydata.size(); rti++)
				{
					vector_2str feaVector = genFeature(myParam, multipleRefs.mydata[rti]);
					double score = cal_score_candidate(myParam, feaVector, multipleRefs.mydata[rti].alignedY, multipleRefs.mydata[rti].alignedX, WLCounts, LMProbs, LMBackoff, maxLM);

					score_yi[stringify(rti)] = score;

					xi_yi = my_feature_map_word(myParam, feaVector, multipleRefs.mydata[rti].alignedY, &featureHash, max_words_doc, multipleRefs.mydata[rti].alignedX, WLCounts, LMProbs, LMBackoff, maxLM);
						
						// vector_xi_yi indexed by rti //
					vector_xi_yi[stringify(rti)] = create_svector(xi_yi,"",1);				
					free(xi_yi);
				}
			}
			else
			{
				cerr << "Can't file  " + myParam.alignLoss + " alignLoss handler" << endl; 
				exit(-1);
			}

			// STOP HERE and THINK //
			
			// calculate number of constraints //
			if (myParam.alignLoss == "mulA")
			{
				docs = (DOC **)my_malloc(sizeof(DOC *) * nBestAnswer.size() * multipleRefs.mydata.size()); 
				rhs = (double *)my_malloc(sizeof(double) * nBestAnswer.size() * multipleRefs.mydata.size());
			}
			else
			{
				docs = (DOC **)my_malloc(sizeof(DOC *) * nBestAnswer.size());
				rhs = (double *)my_malloc(sizeof(double) * nBestAnswer.size()); // rhs constraints
			}

			// training PER //

			error += allNbestPER[0] * max(nBestAnswer[0].size(), multipleRefs.mydata[refForNBest[0]].alignedY.size());
			allPhonemeTrain += max(nBestAnswer[0].size(), multipleRefs.mydata[refForNBest[0]].alignedY.size());
			

			// create doc for updating weights //
			for (unsigned int i = 0; i < nBestAnswer.size(); i++)
			{
				SVECTOR *vector_xi_yk, *vector_diff;
				WORD *xi_yk;

				xi_yk = my_feature_map_word(myParam,featureNbest[i], nBestAnswer[i], &featureHash, max_words_doc, alignedXnBest[i], WLCounts, LMProbs, LMBackoff, maxLM);
				vector_xi_yk = create_svector(xi_yk,"",1);
				free(xi_yk);

				if (myParam.alignLoss == "mulA")
				{
					for (unsigned int rti = 0; rti < multipleRefs.mydata.size(); rti++)
					{
						vector_diff = sub_ss(vector_xi_yi[stringify(rti)], vector_xi_yk);
						long docsID = rti+ (i * multipleRefs.mydata.size());
						docs[docsID] = create_example(docsID, docsID, docsID, 1, vector_diff);

						if (allNbestPER[i] == 0)
						{
							rhs[docsID] = 0;
						}
						else
						{
							rhs[docsID] = allNbestPER[i] - score_yi[stringify(rti)] + scoreNbest[i] + 1;
						}
					}
					free_svector(vector_xi_yk);
				}
				else
				{
					if ((myParam.alignLoss == "avgL") || (myParam.alignLoss == "rakL") || (myParam.alignLoss == "ascL"))
					{
						vector_diff = sub_ss(vector_xi_yi["avg"], vector_xi_yk);
						rhs[i] = allNbestPER[i] - score_yi["avg"] + scoreNbest[i] + 1;

					}
					else
					{
						vector_diff = sub_ss(vector_xi_yi[stringify(refForNBest[i])], vector_xi_yk);
						rhs[i] = allNbestPER[i] - score_yi[stringify(refForNBest[i])] + scoreNbest[i] + 1;
					}
					
					docs[i] = create_example(i,i,i,1,vector_diff);
					free_svector(vector_xi_yk);

					if (allNbestPER[i] == 0)
					{
						rhs[i] = 0;
					}
				}
			}

			MODEL *model = (MODEL *)my_malloc(sizeof(MODEL));

			long int totdoc;

			if (myParam.alignLoss == "mulA")
			{
				totdoc = (long int) nBestAnswer.size() * multipleRefs.mydata.size();
			}
			else
			{
				totdoc = (long int) nBestAnswer.size();
			}
			
			
			long int totwords = (long int) featureHash.size();

			LEARN_PARM learn_parm;
			KERNEL_PARM kernel_parm;
			set_default_parameters(&learn_parm, &kernel_parm);

			learn_parm.svm_c = myParam.SVMcPara;
			//learn_parm.svm_c = 9999999;
			
			KERNEL_CACHE *kernel_cache=NULL;
			double *alpha = NULL;

			svm_learn_optimization(docs, rhs, totdoc, totwords, &learn_parm, &kernel_parm, kernel_cache, model, alpha);

			// update weights w = z + o; 
			// w = new weights
			// z = update part obtained from the optimizer
			// o = old weights
			long sv_num=1;
			SVECTOR *v;
			for(long i=1;i<model->sv_num;i++) 
			{
				for(v=model->supvec[i]->fvec;v;v=v->next) 
					sv_num++;
			}

			for(long i=1;i<model->sv_num;i++)
			{
				for(v=model->supvec[i]->fvec;v;v=v->next) 
				{
					double alpha_value = model->alpha[i]*v->factor;
							
					//vector<WORD> v_words_list;
					for (long j=0; (v->words[j]).wnum; j++)//the bug might be here 
					{
						myWF.updateFeature(my_feature_hash_retrieve(&featureHash, 
								(long)(v->words[j]).wnum),
								(double)(v->words[j]).weight * alpha_value);

					//		v_words_list.push_back(v->words[j]);
					} 
					//cout << "dummy for debug" << endl;
				}
			}

			// free memory

			for (hash_string_SVECTOR::iterator vectorPos = vector_xi_yi.begin() ; vectorPos != vector_xi_yi.end() ; vectorPos++)
			{
				free_svector(vectorPos->second);
			}
			for(long i=0;i<totdoc;i++) 
				free_example(docs[i],1);
			free(rhs);
			free_model(model,0);
			
		}// ending training all data one round

		

		//cout << "Trained on " << trainData.size() << " instances" << endl;
		cout << "Trained on " << actual_train << " instances" << endl;
		cout << "Training PER : " << (error / allPhonemeTrain) << endl;

		cout << "Error reduction on training : " << (p_error_train - error) / p_error_train << endl;

		p_error_train = error;

		cout << "Finalizing the actual/average weights .... " << endl;
		myWF.finalizeWeight(iter);
		

		if (devUnique.size() > 0)
		{
			error = 0;
			allPhonemeTrain = 0;
			actual_dev = 0;
			param myDevParam;
			myDevParam = myParam;

			myDevParam.nBest = 1;
			myDevParam.atTesting = true;

			// reset the dev check//
			//devUniqueChk.clear();

			cout << "Perform on Dev \n";

			for (hash_string_vData::iterator dev_pos = devUnique.begin(); dev_pos != devUnique.end(); dev_pos++)
			{
			//for (unsigned long di = 0; di < devData.size(); di++)
			//{
				vector_2str nBestAnswer;
				vector_2str alignedXnBest;
				vector_3str featureNbest;
				vector<double> scoreNbest;
				vector<double> nBestPER;
			
				// skip dev instance if we have already tested it in this iteration //
				/*string devWord = join(devData[di].unAlignedX,"","");
				if (devUniqueChk.find(devWord) != devUniqueChk.end())
				{
					continue;
				}
				else
				{
					devUniqueChk[devWord] = true;
				}*/

				actual_dev++;

				//nBestAnswer = phrasalDecoder(myDevParam, devData[di].unAlignedX, alignedXnBest, featureNbest, scoreNbest);
				//nBestAnswer = phrasalDecoder(myDevParam, dev_pos->second[0].unAlignedX, alignedXnBest, featureNbest, scoreNbest);
				
				if (myParam.useBeam)
				{
					nBestAnswer = phrasalDecoder_beam(myDevParam, dev_pos->second.mydata[0].unAlignedX, dev_pos->second.mydata[0].unAlignedY, alignedXnBest, featureNbest, scoreNbest, WLCounts, LMProbs, LMBackoff, maxLM);
				}
				else
				{
					nBestAnswer = phrasalDecoder(myDevParam, dev_pos->second.mydata[0].unAlignedX, alignedXnBest, featureNbest, scoreNbest);
				}

				

				vector<data> multipleRefs = dev_pos->second.mydata;
				int refForTheBest;
				
				// nbi = 0; // top list when testing //
				double minPER = 9999;
				for (int rti = 0; rti < multipleRefs.size(); rti++)
				{
					double pos_PER = minEditDistance(multipleRefs[rti].alignedY, nBestAnswer[0], "_");

					if (pos_PER < minPER)
					{
						minPER = pos_PER;
						refForTheBest = rti;
					}
				}

				error += minPER * max(nBestAnswer[0].size(), multipleRefs[refForTheBest].alignedY.size());
				allPhonemeTrain += max(nBestAnswer[0].size(), multipleRefs[refForTheBest].alignedY.size());

			}

			cout << "Test on the dev set of " << actual_dev << " instances" << endl;
			cout << "Dev PER : " << (error / allPhonemeTrain) << endl;
		}

		//if (myParam.keepModel)
		//{
		cout << "Writing weights to : " << myParam.modelOutFilename << "." << iter << endl;
		myWF.writeToFile(myParam.modelOutFilename + "." + stringify(iter));
	
		cout << "Writing max phrase size to : " << myParam.modelOutFilename << "." << iter << ".maxX" << endl;
		writeMaxPhraseSize(myParam, myParam.modelOutFilename + "." + stringify(iter) + ".maxX");
	
		cout << "Writing limited phoneme/letter units to: " << myParam.modelOutFilename << "." << iter << ".limit" << endl;
		myAllPhoneme.writeToFile(myParam.modelOutFilename + "." + stringify(iter) + ".limit", true);
		//}
		

		 // clean up past model (save some space) //
		if ( (iter > 2) && (! myParam.keepModel) )
		{
			string past_modelFilename;
			past_modelFilename = myParam.modelOutFilename + "." + stringify(iter-2);
			cout << "Clean up past model : " << past_modelFilename << endl;
			if (remove(past_modelFilename.c_str()) == 0)
				cout << "Delete file " << past_modelFilename << endl;
			else
				cout << "Cannot delete file " << past_modelFilename << endl;
		}

		cout << "Error reduction ((p_error - error) / p_error) : " << ((p_error - error) / p_error) << endl;

		if ((error >= p_error) && (iter > myParam.trainAtLeast))
		{
			// stop training //
			stillTrain = false;
		}
		else
		{
			// still train //
			p_error = error;
		}

		// train at most condition
		if (iter >= myParam.trainAtMost)
		{
			stillTrain = false;
		}
	}

	// if we haven't written the model, we should write it before finishing //
	// we should write the previous model because that is what the peak is //
	
	/*if ( ! myParam.keepModel )
	{
		cout << "Writing weights to : " << myParam.modelOutFilename << "." << iter-1 << endl;
		myWF.writeToFile(myParam.modelOutFilename + "." + stringify(iter-1),true);
		
		cout << "Writing max phrase size to : " << myParam.modelOutFilename << "." << iter-1 << ".maxX" << endl;
		writeMaxPhraseSize(myParam, myParam.modelOutFilename + "." + stringify(iter-1) + ".maxX");
		
		cout << "Writing limited phoneme/letter units to: " << myParam.modelOutFilename << "." << iter-1 << ".limit" << endl;
		myAllPhoneme.writeToFile(myParam.modelOutFilename + "." + stringify(iter-1) + ".limit", true);
	}*/

	// testing if a test file is given.
	//myParam.modelInFilename = myParam.modelOutFilename + "." + stringify(iter - 1);

	// Only weight parameter we have is from the last iteration, this is probably worse than 
	// the iter - 1 on dev but who knows on the test
	// Change back since we have the previous weights stored, so test on iter-1 //
	myParam.modelInFilename = myParam.modelOutFilename + "." + stringify(iter-1);
}

void phraseModel::testing(param &myParam)
{
	myWF.clear();
	myAllPhoneme.clear(true);

	cout << endl << "Testing starts " << endl;

	cout << "Reading model file : " << myParam.modelInFilename << endl;
	cout << "Please wait ... " << endl;
	myWF.updateFeatureFromFile(myParam.modelInFilename);

	cout << "Reading limited file : " << myParam.modelInFilename << ".limit" << endl;
	myAllPhoneme.addFromFile(myParam.modelInFilename + ".limit", true);

	cout << "Reading max phrase size file : " << myParam.modelInFilename << ".maxX" << endl;
	readMaxPhraseSize(myParam, myParam.modelInFilename + ".maxX");

	cout << "Max phrase size = " << myParam.maxX << endl;

	hash_string_double WLCounts;
        hash_string_double LMProbs;
        hash_string_double LMBackoff;
        int maxLM = 0;

	// at testing //
	myParam.atTesting = true;
	myParam.nBest = myParam.nBestTest;

	ofstream FILEOUT, PHRASEOUT;

	if (myParam.answerFile != "")
	{		
		FILEOUT.open(myParam.answerFile.c_str(), ios_base::trunc);
		if ( ! FILEOUT)
		{
			cerr << "error: unable to create " << myParam.answerFile << endl;
			exit(-1);
		}

		string phraseOutFilename = myParam.answerFile + ".phraseOut";
		PHRASEOUT.open(phraseOutFilename.c_str(), ios_base::trunc);
		if (! PHRASEOUT)
		{
			cerr << "error: unable to create " << phraseOutFilename << endl;
			exit(-1);
		}

		cout << "Answer file : " << myParam.answerFile << endl;
		cout << "Phrase output file : " << phraseOutFilename << endl;
	}

        if (myParam.LMInFilename != "")
        {
                readLMFile(myParam, myParam.LMInFilename, LMProbs, LMBackoff, maxLM);
        }
	if (myParam.WCInFilename != "")
	{
		readWCFile(myParam, myParam.WCInFilename, WLCounts);
	}



	if (myParam.testingFile == "")
	{
		string testword = "";
		vector_str unAlignedTestWord;
		vector_str nullY;

		cout << "Standard input mode: one instance per line. To quit, type an empty line" << endl;

		getline(cin, testword);
		while (testword != "")
		{
			unAlignedTestWord.clear();
			Tokenize(testword, unAlignedTestWord, "");

			vector_2str nBestAnswer;
			vector_2str alignedXnBest;
			vector_3str featureNbest;
			vector<double> scoreNbest;

			if (myParam.useBeam)
			{
				nBestAnswer = phrasalDecoder_beam(myParam, unAlignedTestWord, nullY, alignedXnBest, featureNbest, scoreNbest, WLCounts, LMProbs, LMBackoff, maxLM);
			}
			else
			{
				nBestAnswer = phrasalDecoder(myParam, unAlignedTestWord, alignedXnBest, featureNbest, scoreNbest);
			}

			for (int n = 0; n < nBestAnswer.size(); n++)
			{
				if (myParam.answerFile != "")
				{
					string generated = "N\\A";
					double probability = 0.0;
					int wordLength = 0;
					if(myParam.LMInFilename != "")
					{
 						generated = join(nBestAnswer[n],"","");
                                               	removeSubString(generated, myParam.inChar); // remove inChar
                                               	removeSubString(generated, "_");
						removeSubString(generated, "+");

                                               	wordLength = utf8::distance(generated.c_str(), generated.c_str() + generated.length());
                                               	wordLength += 1; //And word-end boundary
                                               	//cout << generated << endl;

 						wordLength -= std::count(generated.begin(), generated.end(), '!');
                                                wordLength -= std::count(generated.begin(), generated.end(), '@');
                                                //wordLength -= std::count(generated.begin(), generated.end(), '+');
							/*cout << "TEST" << endl;
                                                	cout << "WORD: " << generated << endl;
							cout << "PROB: " << probability << endl; 
                                                	cout << "NPROB: " << probability / wordLength << endl;
							cout << "LENGTH: " << wordLength << endl;
							*/

                                               	probability = getLMProbability(generated, wordLength, LMProbs, LMBackoff, maxLM, true);
					}

					FILEOUT << join(unAlignedTestWord, "", "") << "\t" << join(nBestAnswer[n], myParam.outChar, "_") << endl;
					PHRASEOUT << join(alignedXnBest[n], "|", "") << "|" << "\t";
					PHRASEOUT << join(nBestAnswer[n], "|","") << "|" << "\t";
					PHRASEOUT << n+1 << "\t";
					PHRASEOUT << scoreNbest[n] << "\t" << generated << "\t" << probability / wordLength << endl;
				}
				else
				{
					cout << join(unAlignedTestWord, "", "") << "\t" << join(nBestAnswer[n], myParam.outChar, "_") << endl;
				}
			}

			if (myParam.answerFile != "")
			{
				PHRASEOUT  << endl;
			}

			getline(cin, testword);
		}
	}
	else
	{
		//hash_string_vData testData;
		vector_vData testData;
		cout << "Reading the test file " << endl;
		//readingAlignedFile(myParam, myParam.testingFile, testData);

		size_t totRead = 0;
		cout << "Reading file: " << myParam.testingFile << endl;
		cout << endl << endl;
		
	
		ifstream INPUTFILE;

		INPUTFILE.open(myParam.testingFile.c_str());
		if (! INPUTFILE)
		{
			cerr << endl << "Error: unable to open file " << myParam.testingFile << endl;
			exit(-1);
		}

		while (! INPUTFILE.eof())
		{
			string line;
			vector<string> lineList;

			data lineData;

			getline(INPUTFILE, line);

			// ignore empty line
			if (line == "")
			{
				continue;
			}

			// ignore line that indicate no alignment //
			if (line.find("NO ALIGNMENT") != string::npos)
			{
				continue;
			}

			lineList = splitBySpace(line);

			if (lineList.size() > 4)
			{
				cerr << endl << "Warning: wrong expected format" << endl << line << endl;
				break;
			}

			processExample(myParam, totRead, lineList, testData);
	
		
			//readingTestingFile(myParam, myParam.testingFile, testData);
		
			for (vector_vData::iterator test_pos = testData.begin(); test_pos != testData.end(); test_pos++)
			{
			/*for (unsigned long ti = 0; ti < testData.size(); ti++)
			{*/
				vector_2str nBestAnswer;
				vector_2str alignedXnBest;
				vector_3str featureNbest;
				vector<double> scoreNbest;
				vector<double> nBestPER;
				vector_str nullY;

				if (myParam.useBeam)
				{
					//nBestAnswer = phrasalDecoder_beam(myParam, (test_pos->second).mydata[0].unAlignedX, alignedXnBest, featureNbest, scoreNbest);
					nBestAnswer = phrasalDecoder_beam(myParam, test_pos->mydata[0].unAlignedX, nullY, alignedXnBest, featureNbest, scoreNbest, WLCounts, LMProbs, LMBackoff, maxLM);
				}
				else
				{
					nBestAnswer = phrasalDecoder(myParam, test_pos->mydata[0].unAlignedX, alignedXnBest, featureNbest, scoreNbest);
				}

				for (int n = 0; n < nBestAnswer.size(); n++)
				{
					if (myParam.answerFile != "")
					{
        					string generated = "N\\A";
                        	                double probability = 0.0;
						int wordLength = 0;
                        	                if(myParam.LMInFilename != "")
                        	                {
                        	                        generated = join(nBestAnswer[n],"","");
                        	                        removeSubString(generated, myParam.inChar); // remove inChar
                        	                        removeSubString(generated, "_");
							removeSubString(generated, "+");

                        	                        wordLength = utf8::distance(generated.c_str(), generated.c_str() + generated.length());
                        	                        wordLength += 1; //And word-end boundary
                        	                        //cout << generated << endl;
	
 							wordLength -= std::count(generated.begin(), generated.end(), '!');
        	                                        wordLength -= std::count(generated.begin(), generated.end(), '@');
        	                                        //wordLength -= std::count(generated.begin(), generated.end(), '+');




        	                                        probability = getLMProbability(generated, wordLength, LMProbs, LMBackoff, maxLM, true);
							/*cout << "TEST " << endl << endl;
                                                	cout << "WORD: " << generated << endl;
							cout << "PROB: " << probability << endl; 
                                                	cout << "NPROB: " << probability / wordLength << endl;
							cout << "LENGTH: " << wordLength << endl;
							*/
                                        	}

						FILEOUT << join(test_pos->mydata[0].unAlignedX, "", "") << "\t" << join(nBestAnswer[n], myParam.outChar, "_") << endl;
						PHRASEOUT << join(alignedXnBest[n], "|", "") << "|" << "\t";
                                        	PHRASEOUT << join(nBestAnswer[n], "|","") << "|" << "\t";
                                        	PHRASEOUT << n+1 << "\t";
                                        	PHRASEOUT << scoreNbest[n] << "\t" << generated << "\t" << probability / wordLength<< endl;
                                	}
					else
					{
						cout << join(test_pos->mydata[0].unAlignedX, "", "") << "\t" << join(nBestAnswer[n], myParam.outChar, "_") << endl;
					}
				}

				if (myParam.answerFile != "")
				{
					PHRASEOUT  << endl;
				}
			}
			testData.clear();
		}
	}

	if (myParam.answerFile != "")
	{
		FILEOUT.close();
		PHRASEOUT.close();
	}

}

double phraseModel::getLMProbability(string sequence, int& wordLength, hash_string_double& LMProbs, hash_string_double& LMBackoff, int maxNGram, bool wholeWord)
//This function calculates the log (base 10) probability of a sequence, given the provided LM
{
	//cout << "Sequence: " << sequence << endl;

 	::setlocale(LC_ALL, "");
        std::transform(sequence.begin(), sequence.end(), sequence.begin(), ::towlower);

	//cout << "LCed: "<< sequence << endl;
        std::vector<std::string> parts;
        std::size_t pos = 0, found;

	//cout << sequence << endl;
        while((found = sequence.find_first_of("!@", pos)) != std::string::npos) {
			
			if(found - pos != 0)
			{
                        	parts.push_back(sequence.substr(pos, found - pos));//It should be okay to split with ASCII, because "!" and "@" are ASCII chars
			}
                        pos = found+1;
                }
        parts.push_back(sequence.substr(pos));
	wordLength += parts.size() - 1;
	/*
	if(parts.size() > 1)
	{
		for(int index = 0; index < parts.size(); index++)
		{	
			cout << parts[index] << endl;
		}
	}
	*/
	double probability = 0.0;
	string nGram = "";
	int nGramSize = maxNGram;
	for(int index = 0; index < parts.size(); index++)
	{
		
		string subSequence = parts[index];
		//cout << "SUBSEQ: " << subSequence << endl;
		int trueLength = utf8::distance(subSequence.c_str(), subSequence.c_str() + subSequence.length());
		//cout << "LENGTH: " << trueLength << endl;
		for(int i = 0; i < trueLength; i++)
		{
			if(i < maxNGram - 1)
			{
				nGram = "<w>" + utf8_substr(subSequence, 0, i+1);//sequence.substr(0, i+1);
				nGramSize = i + 2;
			}
			else
			{
				nGramSize = maxNGram;
				//cout << i << "\t" << trueLength << "\t" << nGramSize << endl;
				nGram = utf8_substr(subSequence, i+1-nGramSize, nGramSize);//sequence.substr(i + 1 - maxNGram, maxNGram); 
			}
			double prob = getNGramProb(nGram, LMProbs, LMBackoff, nGramSize);
			//if(trueLength == maxNGram - 2 && LMProbs.find(nGram) != LMProbs.end())
			//{
			//	probability = 0.9 * (maxNGram - 1);
			//	return probability;
			//}
			//cout << nGram << " " << prob << endl;
			probability += prob; 
		}

        	if (wholeWord || index != parts.size()-1)//ie, we are at the end of a word;
        	{
            	nGram += "<\\w>";
            	nGramSize ++;
        	
        	    if(nGramSize > maxNGram)
        	    {
        	        //cout << nGram << endl;
        	        if(nGram.find("<w>") != string::npos)
        	        {
        	            nGram = utf8_substr(nGram, 3, nGram.length() - 3);//nGram.substr(3);//Chop the start of word character
        	        }
        	        else
        	        {
        	             nGram = utf8_substr(nGram, 1, nGram.length() - 1);//nGram.substr(1);//Chop the first character
        	        }
        	        nGramSize--;
        	    }
            
	            double prob = getNGramProb(nGram, LMProbs, LMBackoff, nGramSize);
		    //cout << nGram << " " << prob << endl;
			
        	    probability += prob;

        
		}
	}
	return probability;
	
}

double phraseModel::getNGramProb(string nGram, hash_string_double& LMProbs, hash_string_double& LMBackoff, int nGramSize)
{

	double prob = 0.0;
	//cout << "Checking " << nGram << endl;
	int length = nGramSize;
 	::setlocale(LC_ALL, "");
        std::transform(nGram.begin(), nGram.end(), nGram.begin(), ::towlower);

	//In case uppercase letters are not in LM


		while(length > 0)
		{
			//cout << nGram << "\t" << length << endl;
        		if(length <= 2)//Allow backoff to bigrams if a bigram is unobserved, it's probably not a very good word
        		{
        		        if(LMProbs.find(nGram) == LMProbs.end())//ie, unigram is unknown
                		{
                        		//cout << "<UNK> " << LMProbs["<UNK>"] << endl;
                        		//return -;
					//cout << nGram << "PROB: " << prob + LMProbs["<UNK>"] << endl;
					return prob + LMProbs["<UNK>"];
                		}
                		else
                		{
                        		//cout << nGram + " " << LMProbs[nGram] << endl;
					//cout << nGram << "PROB: " << prob + LMProbs[nGram] << endl;
					return prob + LMProbs[nGram];
               		 	}
        		}

			//cout << "Length: " << length << endl;
			if(LMProbs.find(nGram) != LMProbs.end() && LMProbs[nGram] != 0.0)//ie, nGram exists
			{
				//cout << nGram << " found! " << LMProbs[nGram] << endl;
				prob += LMProbs[nGram];
			        //cout << nGram << "PROB: " << prob << endl;
				return prob;
			}

			else
			{
				//cout << nGram << " " << -99 << endl;
				string prefix = utf8_substr(nGram, 0, length - 1);
				//length already accounts for <\w> as one character 
				//cout << "Prefix: " << prefix << endl;
				string suffix = utf8_substr(nGram, 1, string::npos);
				if(nGram.find("<w>") != string::npos)
				{
					prefix = utf8_substr(nGram, 0, length + 1);//Account for extra characters of <w>
					suffix = utf8_substr(nGram, 3, string::npos);//<w> is 3 characters, not 1;
				}
				//cout << nGram << "\t" << prefix << "\t" << suffix << endl;
				
				//For backoff of p(w3|w2,w1,w0), we use BO(w0,w1,w2) * p(w3 | w2, w1)
				if(LMBackoff.find(prefix) != LMBackoff.end() && LMBackoff[prefix] != 0.0)
				{
					prob += LMBackoff[prefix];
				//	cout << nGram << " not found, backing off to " << suffix << " with backoff " << LMBackoff[prefix] << endl;
					nGram = suffix;
				}			

				else
				{
				//	cout << nGram << " not found, backing off to " << suffix << " with no backoff" << endl;
					nGram = suffix;
				}
				length -= 1;//utf8::distance(nGram.c_str(), nGram.c_str() + nGram.length());
				//cout << "New length: " << length << endl;
			}
		}

	return prob;
}
