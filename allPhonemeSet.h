#pragma once

#include <string>
#include <map>
#include <set>
#include <vector>
#include <fstream>

using namespace std;

class allPhonemeSet
{
private:
	set<string> allPhoneme;
	map<string, set<string> > allPhonemeLimit;
public:
	allPhonemeSet(void);
	~allPhonemeSet(void);
	vector<string> getPhoneme(string letter, bool limitCandidate=true);
	void addPhoneme(string phoneme, string letter, bool limitCandidate=true);
	bool writeToFile(string filename, bool limitCandidate=true);
	void addFromFile(string filename, bool limitCandidate=true);
	set<string> getAllPhoneme();
	map<string, set<string> > getAllPhonemeLimit();
	void clear(bool limitCandidate=true);
	void clear(void);

	allPhonemeSet& operator= (const allPhonemeSet& param);
};


