#include "weightWF.h"

weightWF::weightWF()
{
	wf.clear();
}

weightWF::~weightWF()
{

}

double weightWF::getFeature(string featureStr, bool getAvg)
{
	my_hash_map::iterator pos;

	pos = wf.find(featureStr);

	if (pos != wf.end())
	{
		if (getAvg)
			return pos->second.avg;
		else
			return pos->second.actual;
	}
	else
	{
		return 0;
	}
}

double weightWF::getFeature(string featureStr, string phonemeTag, bool getAvg)
{
	my_hash_map::iterator pos;
	string tempStr;

	tempStr = featureStr + "T:" + phonemeTag;

	pos = wf.find(tempStr);

	if (pos != wf.end())
	{
		if (getAvg)
			return pos->second.avg;
		else
			return pos->second.actual;
	}
	else
	{
		return 0;
	}
}

void weightWF::updateFeature(std::string featureStr, std::string phonemeTag, double weightUpdate,bool onAvg)
{
	string tempStr;
	tempStr = featureStr + "T:" + phonemeTag;

	if (onAvg)
	{
		wf[tempStr].avg += weightUpdate;
	}
	else
	{
		wf[tempStr].actual += weightUpdate;
	}
}

void weightWF::updateFeature(string featureStr,double weightUpdate, bool onAvg)
{
	if (onAvg)
	{
		wf[featureStr].avg += weightUpdate;
	}
	else
	{
		wf[featureStr].actual += weightUpdate;
	}
}

// format: feature phoneme actual avg.
bool weightWF::updateFeatureFromFile(std::string modelFilename)
{
	ifstream MODELFILE;
	string line;
	vector<string> lineList;

	MODELFILE.open(modelFilename.c_str());
	if (! MODELFILE)
	{
		return false;
	}

	while ( ! MODELFILE.eof() )
	{
		getline(MODELFILE,line);
		
		lineList = splitBySpace(line);

		if (lineList.size() < 4)
		{
			continue;
		}

		if (lineList.size() > 0)
		{
			updateFeature(lineList[0],lineList[1], convertTo<double>(lineList[2]),false);
			updateFeature(lineList[0],lineList[1], convertTo<double>(lineList[3]),true);
		}
	}
	MODELFILE.close();
	return true;
}

// format: feature phoneme actual avg.
bool weightWF::writeToFile(std::string modelFilename, bool usePrevious)
{
	ofstream MODELFILE;
	MODELFILE.open(modelFilename.c_str(),ios_base::trunc);
	
	if (! MODELFILE)
	{
		cerr << "ERROR: Can't write file : " << modelFilename << endl;
		exit(-1);
	}
	else
	{
		for (my_hash_map::iterator wfPos = wf.begin(); wfPos != wf.end(); wfPos++)
		{
			MODELFILE << replaceStrTo(wfPos->first, "T:", "\t");

			MODELFILE << "\t";
			MODELFILE << wfPos->second.actual;
			MODELFILE << "\t";
			/*if (usePrevious == true)
			{
				MODELFILE << wfPos->second.pre_avg;
			}
			else
			{
				MODELFILE << wfPos->second.avg;
			}
			*/
			MODELFILE << wfPos->second.avg;
			MODELFILE << endl;
		}

		MODELFILE.close();
		return true;
	}
}

bool weightWF::finalizeWeight(double atIteration)
{
	if (atIteration < 1)
	{
		return false;
	}

	// omitted the number of training instances
	// only use the number of iterations
	if (atIteration == 1)
	{
		for (my_hash_map::iterator wfPos = wf.begin(); wfPos != wf.end(); wfPos++)
		{
			wfPos->second.avg = wfPos->second.actual; 
			//wfPos->second.pre_avg = wfPos->second.actual; //initially pre = current avg.
		}
	}
	else
	{
		for (my_hash_map::iterator wfPos = wf.begin(); wfPos != wf.end(); wfPos++)
		{
			// save the previous avg. iteration // 
			//wfPos->second.pre_avg = wfPos->second.avg;

			// real average based on iteration goes ///
			//wfPos->second.avg = (wfPos->second.avg * ((atIteration - 1) / atIteration)) + (wfPos->second.actual / atIteration);
			
			// a more aggressive update -- average between old and new weights regardless which iteration //
			wfPos->second.avg += wfPos->second.actual;
		}
	}

	return true;
}

my_hash_map weightWF::getWF()
{
	return wf;
}


void weightWF::clear()
{
	wf.clear();
}
