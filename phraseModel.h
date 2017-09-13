#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <clocale>
#include <locale>
#include "util.h"
#include "allPhonemeSet.h"
#include "weightWF.h"
#include "param.h"
//#include "utf8.h"
//extern "C" {
//# include "svm_common.h"
//# include "svm_learn.h"
//} 
#include "svm_common.h"
#include "svm_learn.h"

using namespace std;

void set_default_parameters(LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm);

class phraseModel
{
	weightWF myWF;
	allPhonemeSet myAllPhoneme;

	// hash map between myFeature and SVMfeature
	hash_string_long featureHash;
	
public:
	phraseModel(void);
	~phraseModel(void);

	void training(param& myParam);
	void testing(param& myParam);

	//void readingAlignedFile(param &myParam, string filename, vector<data>& output);
	void readingAlignedFile(param &myParam, string filename, hash_string_vData& output, bool genFea=false);
	//void dataToUnique(param &myParam, vector<data>& inData, hash_string_vData& outUniqueData);

	void readingTestingFile(param &myParam, string filename, vector_vData& output, bool genFea=false);
	void readLMFile(param &myParam, string filename, hash_string_double& LMProbs, hash_string_double& LMBackoff, int& maxLM);
	void readWCFile(param &myParam, string filename, hash_string_double& WLCounts);
       
	void readingExtraFeature(param &myParam, string filename, hash_string_vData& output);

	//void initialize(param& myParam, vector<data> trainingData);
	void initialize(param& myParam, hash_string_vData& trainDataUnique);

	vector_2str phrasalDecoder(param& myParam, vector_str unAlignedX, vector_2str &alignedXnBest, vector_3str &featureNbest, vector<double>& scoreNbest);

 	vector_2str phrasalDecoder_beam(param& myParam, vector_str unAlignedX, vector_str unAlignedY, vector_2str &alignedXnBest, vector_3str &featureNbest, vector<double>& scoreNbest, hash_string_double& WLCounts, hash_string_double& LMProbs, hash_string_double& LMBackoff, int maxLM);

	vector_str ngramFeatureGen(param& myParam, vector_str lcontext, vector_str focus, vector_str rcontext);
	vector_2str genFeature(param& myParam, data dpoint);

	double getLocalFeatureScore(param& myParam, vector_str observation, string classLabel);
	double getCopyScore(param& myParam, bool isCopy, string classLabel);
	double getFinnishVHScore(param& myParam, string correctY, string currentY, vector_str jointY);
	double getTurkishVHScore(param& myParam, string correctY, string currentY, vector_str jointY);
	double getTurkishRHScore(param& myParam, string correctY, string currentY, vector_str jointY);
	double getOrderFeatureScore(param& myParam, vector_str observation, string p_class, string c_class);
        double getHigherOrderFeatureScore(param& myParam, vector_str observation
, vector_str jointY, string c_class);
        double getJointGramFeatureScore(param& myParam, vector_str jointX, vector_str jointY, string currentX, string currentY);

	double getJointForwardGramFeatureScore(param& myParam, string currentX, string currentY, vector_str xForward);

	double minEditDistance(vector<string> str1, vector<string> str2, string ignoreString = "");

	double getLMProbability(string sequence, int& wordLength, hash_string_double& LMProbs, hash_string_double& LMBackoff, int maxNGram, bool wholeWord);
	double getNGramProb(string nGram, hash_string_double& LMProbs, hash_string_double& LMBackoff, int nGramSize);

	double getLMFeatureScore(param &myParam, double LMScore);
	string getLMBin(double LMScore);

	double getWCFeatureScore(param &myParam, hash_string_double& WLCounts, string word);
	string getWCBin(double count);

	long my_feature_hash(string feaList, string phonemeTarget, hash_string_long *featureHash);

	WORD *my_feature_map_word(param &myParam, vector_2str featureList, vector_str alignedTarget, hash_string_long *featureHash, long max_words_doc, vector_str alignedSource, hash_string_double& WLCounts, hash_string_double& LMProbs, hash_string_double& LMBackoff, int maxLM);

	void my_feature_hash_avg(param &myParam, vector_2str featureList, vector_str alignedTarget, 
											  hash_string_long *featureHash, double scaleAvg, map<long,double> &idAvg);

	WORD *my_feature_hash_map_word(param &myParam, map<long,double> &idAvg, long max_words_doc);


	double cal_score_candidate(param &myParam, vector_2str featureList, vector_str alignedTarget, vector_str alignedSource, hash_string_double& WLCounts, hash_string_double& LMProbs, hash_string_double& LMBackoff, int maxLM);

	double cal_score_hash_avg(param &myParam, map<long,double> &idAvg);

	string my_feature_hash_retrieve(hash_string_long *featureHash, long value);

	void writeMaxPhraseSize(param &myParam, string filename);
	void readMaxPhraseSize(param &myParam, string filename);

	//std::string utf8_substr(std::string str, unsigned int start, unsigned int leng)
};


