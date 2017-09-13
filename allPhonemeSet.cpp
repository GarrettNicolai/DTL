#include "allPhonemeSet.h"
#include "util.h"

allPhonemeSet::allPhonemeSet(void)
{
}

allPhonemeSet::~allPhonemeSet(void)
{
}

map<string,set<string> > allPhonemeSet::getAllPhonemeLimit()
{
	return allPhonemeLimit;
}

set<string> allPhonemeSet::getAllPhoneme()
{
	return allPhoneme;
}

vector<string> allPhonemeSet::getPhoneme(string letter, bool limitCandidate)
{
	vector<string> out;

	out.clear();
	if (limitCandidate)
	{
		map<string, set<string> >::iterator pos;
		pos = allPhonemeLimit.find(letter);
		if (pos != allPhonemeLimit.end())
		{
			for (set<string>::iterator pos = allPhonemeLimit[letter].begin(); pos != allPhonemeLimit[letter].end(); pos++)
			{
				out.push_back(*pos);
			}
		}
		else
		{
			// Edit Feb 6, 08 for not returning non-posible letter/phoneme pair
			//out.push_back("_");
		}
	}
	else
	{
		if (allPhoneme.size() < 1)
		{
			out.push_back("_");
		}
		else
		{
			for (set<string>::iterator pos = allPhoneme.begin(); pos != allPhoneme.end(); pos++)
			{
				out.push_back(*pos);
			}
		}
	}
	return out;
}

void allPhonemeSet::clear(void)
{
	allPhonemeLimit.clear();
	allPhoneme.clear();
}

void allPhonemeSet::clear(bool limitCandidate)
{
	if (limitCandidate)
	{
		allPhonemeLimit.clear();
	}
	else
	{
		allPhoneme.clear();
	}
}

void allPhonemeSet::addPhoneme(string phoneme, string letter, bool limitCandidate)
{
	if (limitCandidate)
	{
		allPhonemeLimit[letter].insert(phoneme);
	}
	else
	{
		allPhoneme.insert(phoneme);
	}
}

void allPhonemeSet::addFromFile(string filename, bool limitCandidate)
{
	string line;
	vector<string> lineList;
	if (limitCandidate)
	{
		ifstream FILEIN;
		FILEIN.open(filename.c_str());

		while (! FILEIN.eof())
		{
			getline(FILEIN,line);
			lineList = splitBySpace(line);

			if (lineList.size() > 1)
			{
				addPhoneme(lineList[1],lineList[0],limitCandidate);
			}
		}
		FILEIN.close();
	}
}

bool allPhonemeSet::writeToFile(string filename, bool limitCandidate)
{
	if (limitCandidate)
	{
		ofstream FILEOUT;
		FILEOUT.open(filename.c_str(), ios_base::trunc);
		if (! FILEOUT)
		{
			cerr << "ERROR: Can't write file to : " << filename << endl;
			exit(-1);
		}
		else
		{
			for (map<string, set<string> >::iterator allLetterPos = allPhonemeLimit.begin(); allLetterPos != allPhonemeLimit.end(); allLetterPos++)
			{
				for (set<string>::iterator allPhonemePos = allPhonemeLimit[allLetterPos->first].begin(); allPhonemePos != allPhonemeLimit[allLetterPos->first].end(); allPhonemePos++)
				{
					FILEOUT << allLetterPos->first << "\t" << *allPhonemePos << endl;
				}
			}

			FILEOUT.close();
			return true;
		}
	}
	else
	{
// do not thing for now
		return true;
	}
}

allPhonemeSet& allPhonemeSet::operator= (const allPhonemeSet& param)
{
	allPhoneme = param.allPhoneme;
	allPhonemeLimit = param.allPhonemeLimit;

	return *this;
}

