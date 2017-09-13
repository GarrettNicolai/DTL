#pragma once

#include <string>
#include <fstream>
#include "util.h"

#include <hash_map>

using namespace std;

typedef struct WEIGHTS {
	double actual; //actual weight
	double avg; //average weight
	//double pre_avg; // average weight of previous trained iteration.
} weights;

typedef std::hash_map<string, weights, hash<string>, eqstr> my_hash_map;


class weightWF
{
private:
	my_hash_map wf;
	
public:
	double getFeature(string featureStr, string phonemeTag, bool getAvg=false);
	double getFeature(string featureStr, bool getAvg=false);
	
	void updateFeature(string featureStr, string phonemeTag, double weightUpdate, bool onAvg=false);
	void updateFeature(string featureStr, double weightUpdate, bool onAvg=false);
	
	// when update feature from file, we get both actual and avg are the same
	bool updateFeatureFromFile(string modelFilename);
	bool writeToFile(string modeFilename, bool usePrevious=false);

	bool finalizeWeight(double atIteration); // to be called at the end of each training iteration
	 
	my_hash_map getWF(); // note: make sure if we need this method after fixing average model

	void clear();
	weightWF(void);
	~weightWF(void);
};

