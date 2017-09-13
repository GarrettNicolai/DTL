#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>
#include <stdexcept>

#include <vector>
#include <map>
#include <algorithm>
#include <set>
#include <hash_map>
#include "svm_common.h"
#include "svm_learn.h"
#include <wctype.h>

using namespace std;

template <typename T, typename V>
inline void removeVectorElem(T& x, const V removeValue)
{
	x.erase(remove(x.begin(), x.end(), removeValue), x.end());
	T(x).swap(x);
}

inline string classOfLastVowel(string toConsider, string language)
{
	//cout << "Considering\t" << toConsider << endl;
	string lastVowel = "Neutral";//No vowels yet, so neutral
	if(language.compare("FIN") != 0 && language.compare("TRK") != 0)
	{
		return "";//Not a valid language
	}
	else
	{
		if(language.compare("FIN") == 0)
		{
			for(int i = 0; i < toConsider.length(); i++)
			{

				//int len = 1;
				int len = (( 0xE5000000 >> (( toConsider[i] >> 3 ) & 0x1e )) & 3 ) + 1; //This determines the multi-byte length of the character
        			string currentChar = toConsider.substr(i, len);//And obtains the character
				i += len - 1;	
				if(currentChar.compare("A") == 0
				|| currentChar.compare("O") == 0
				|| currentChar.compare("y") == 0)
				{	
					lastVowel = "Front";
				}
				else if(currentChar.compare("a") == 0
				|| currentChar.compare("o") == 0
				|| currentChar.compare("u") == 0)
				{
					lastVowel = "Back";
				}
				else
				{
					continue;
				}
			}
		}
		if(language.compare("TRK") == 0)
		{
			for(int i = 0; i < toConsider.length(); i++)
			{
				//int len = 1;
				int len = (( 0xE5000000 >> (( toConsider[i] >> 3 ) & 0x1e )) & 3 ) + 1; //This determines the multi-byte length of the character
        			string currentChar = toConsider.substr(i, len);//And obtains the character
			//	cout << "Current:\t" << currentChar << "\t";
				i += len - 1;	
				
				if(currentChar.compare("i") == 0
				|| currentChar.compare("e") == 0
				|| currentChar.compare("O") == 0
				|| currentChar.compare("U") == 0)
				{
					lastVowel = "Front";
			//		cout << "Front" << endl;
				}
				else if(currentChar.compare("I") == 0
				|| currentChar.compare("a") == 0
				|| currentChar.compare("o") == 0
				|| currentChar.compare("u") == 0)
				{
					lastVowel = "Back";
			//		cout << "Back" << endl;
				}
			}
		}
	}

	//cout << "Returning:\t" << lastVowel << endl;
	return lastVowel;
}

inline bool lastVowelRounded(string previous)
{
	bool rounded = false;
	//cout << "Considering:\t" << previous << "\tfor rounding" << endl;
	//We can't go backwards through the word, because Unicode characters are identified in the first bits; which also identify their length in bytes
	for(int i = 0; i < previous.length(); i++)
	{

		//int len = 1;
		int len = (( 0xE5000000 >> (( previous[i] >> 3 ) & 0x1e )) & 3 ) + 1; //This determines the multi-byte length of the character
        	string currentChar = previous.substr(i, len);//And obtains the character
		i += len - 1;	
		if(currentChar.compare("u") == 0
		|| currentChar.compare("o") == 0
		|| currentChar.compare("O") == 0
		|| currentChar.compare("U") == 0)
		{
			rounded = true; //If we encounter a rounded vowel, rounding is currently true
		}
		else if(currentChar.compare("I") == 0
		|| currentChar.compare("a") == 0
		|| currentChar.compare("i") == 0
		|| currentChar.compare("e") == 0)
		{
			rounded = false; //Otherwise, if we hit a vowel, rounding is false
		}
		else continue;
	}
	
	//cout << "Returning:\t" << rounded << endl;		
	return rounded; //If there was no previous vowel, rounded is false; otherwise, it is only true if the final vowel encountered was rounded 
}

inline int containsTurkishVowel(string current)
{
	string front = "ieOU";
	string back = "Iaou";

	if(current.find_first_of(front) != string::npos || current.find_first_of(back) != string::npos)
	{
		return true;
	}
	return false;
}

inline int containsFinnishVowel(string current)
{
	string neutral = "ei";
	string front = "yAO";
	string back = "uao";
	bool frontFound = false;
	bool backFound = false;
	bool neutralFound = false;

	if(current.find_first_of(front) != string::npos)
	{
		frontFound = true;
	}
	if(current.find_first_of(back) != string::npos)
	{
		backFound = true;
	}
	if(current.find_first_of(neutral) != string::npos)
	{
		neutralFound = true;
	}

	if(frontFound && backFound)
	{
		return 4;
	}
	else if (frontFound)
	{
		return 1;
	}
	else if (backFound)
	{
		return 2;
	}
	else if (neutralFound)
	{
		return 3;
	}
	else return -1;
	
}

inline int removeSubString(string& str, string removeStr)
{
	int nDelete = 0;

	if (removeStr == "")
	{
		return nDelete;
	}

	string::size_type pos = str.find(removeStr,0);

	while (string::npos != pos)
	{
		str.erase(pos,removeStr.length());
		nDelete++;

		pos = str.find(removeStr,0);
	}

	return nDelete;
}

inline void Tokenize(const string& str,
                      vector<string>& tokens,
                      const string& delimiters = "\t")
{
	if (delimiters != "")
	{
		// Skip delimiters at beginning.
		string::size_type lastPos = str.find_first_not_of(delimiters, 0);
		// Find first "non-delimiter".
		string::size_type pos     = str.find_first_of(delimiters, lastPos);
		while (string::npos != pos || string::npos != lastPos)
		{
			// Found a token, add it to the vector.
			tokens.push_back(str.substr(lastPos, pos - lastPos));
			// Skip delimiters.  Note the "not_of"
			lastPos = str.find_first_not_of(delimiters, pos);
			// Find next "non-delimiter"
			pos = str.find_first_of(delimiters, lastPos);
		}
	}
	else
	{
		for (int pos = 0; pos < str.size(); pos++)
		{
			tokens.push_back(str.substr(pos,1));
		}
	}
}
 
 class BadConversion : public std::runtime_error {
 public:
   BadConversion(const std::string& s)
     : std::runtime_error(s)
     { }
 };

 template<typename T>
 inline std::string stringify(const T& x)
 {
   std::ostringstream o;
   if (!(o << x))
     throw BadConversion(std::string("stringify(")
                         + typeid(x).name() + ")");
   return o.str();
 }


 template<typename T>
 inline void convert(const std::string& s, T& x,
                     bool failIfLeftoverChars = true)
 {
   std::istringstream i(s);
   char c;
   if (!(i >> x) || (failIfLeftoverChars && i.get(c)))
     throw BadConversion(s);
 }

 template<typename T>
 inline T convertTo(const std::string& s,
                    bool failIfLeftoverChars = true)
 {
   T x;
   convert(s, x, failIfLeftoverChars);
   return x;
 }
 
 template<typename T>
 inline std::string join(const T& x, std::string delimeter, std::string ignoreChar = "")
 {
	 std::string sout;

	 sout.clear();
	 for (typename T::const_iterator iter = x.begin(); iter != x.end(); iter++)
	 {
		 if (*iter != ignoreChar)
		 {
			sout = sout + *iter + delimeter;
		 }
	 }
	 if (sout.size() > 0) {
	 	sout.erase(sout.end() - delimeter.length(), sout.end());
	 }

	 return sout;
 }

  template<typename T>
 inline std::string join(const T& x, int pos, int npos, std::string delimeter, std::string ignoreChar = "")
 {
	 std::string sout;

	 sout.clear();

	 if (pos < 0)
	 {
		 pos = 0;
	 }

	 for (typename T::const_iterator iter = x.begin()+pos; (iter != x.end()) && (iter != x.begin()+npos); iter++)
	 {
		 if (*iter != ignoreChar)
		 {
			sout = sout + *iter + delimeter;
		 }
	 }

	 if (sout.size() > 0)
	 {
		sout.erase(sout.end() - delimeter.length(), sout.end());
	 }

	 return sout;
 }

 inline std::vector<std::string> splitBySpace(std::string line)
 {
	 std::vector<std::string> lineList;
	 std::string buff;
	 std::stringstream ss(line);

	 while (ss >> buff)
	 {
		 lineList.push_back(buff);
	 }
	
	 return lineList;
 }

 inline std::string replaceStrTo(std::string line, std::string searchPattern, std::string replaceTo)
 {
	 std::string output;
	 std::string::size_type pos;

	 output = line;

	 while (std::string::npos != (pos = output.find(searchPattern)))
	 {
		 output.replace(pos, searchPattern.length(), replaceTo);
	 }
	 return output;
 }

inline bool value_comparer(const std::pair<std::string, double>& lhs,
					const std::pair<std::string, double>& rhs)
{
	if (lhs.second == rhs.second)
	{
		return (lhs.first < rhs.first);
	}
	else
		return (lhs.second < rhs.second);
}

/* function object to check the value of a map element
 */
template <class K, class V>
class value_equals {
  private:
    V value;
  public:
    // constructor (initialize value to compare with)
    value_equals (const V& v)
     : value(v) {
    }
    // comparison
	bool operator() (std::pair<const K, V> elem) {
        return elem.second == value;
    }
};

// for hash map eqsrt function // 
struct eqstr
{
  bool operator()(const string s1, const string s2) const
  {
	  return s1.compare(s2) == 0;
  }
};



typedef std::hash_map<string, double, hash<string>, eqstr> hash_string_double;
typedef std::hash_map<string, int, std::hash<string>, eqstr> hash_string_int;
typedef std::hash_map<string, string, hash<string>, eqstr> hash_string_string;
typedef std::hash_map<string, bool, hash<string>, eqstr> hash_string_bool;
typedef std::hash_map<string, long, hash<string>, eqstr> hash_string_long;
typedef std::hash_map<string, unsigned long, hash<string>, eqstr> hash_string_unsigned_long;

typedef vector<hash_string_double> D_hash_string_double;
typedef vector<hash_string_string> D_hash_string_string;
typedef vector<hash_string_int> D_hash_string_int;
typedef vector<hash_string_bool> D_hash_string_bool;
typedef vector<hash_string_long> D_hash_string_long;

typedef vector<D_hash_string_double> DD_hash_string_double;
typedef vector<D_hash_string_string> DD_hash_string_string;
typedef vector<D_hash_string_int> DD_hash_string_int;
typedef vector<D_hash_string_bool> DD_hash_string_bool;
typedef vector<D_hash_string_long> DD_hash_string_long;

typedef std::hash_map<string, vector<double>, hash<string>, eqstr> hash_string_vectorDouble;
typedef std::hash_map<string, vector<int>, hash<string>, eqstr> hash_string_vectorInt;
typedef std::hash_map<string, vector<string>, hash<string>, eqstr> hash_string_vectorString;
typedef std::hash_map<string, vector<bool>, hash<string>, eqstr> hash_string_vectorBool;
typedef std::hash_map<string, SVECTOR *, hash<string>, eqstr> hash_string_SVECTOR;

typedef struct BTABLE{
	double score;
	vector<int> phraseSize;
	vector<string> jointX;
	vector<string> jointY;
	string currentX;
	string currentY;
	double LMScore;
} btable;

typedef vector<btable> D_btable;
typedef vector<D_btable> DD_btable;

typedef struct QTABLE{
	double score;
	int phraseSize;
	string backTracking;
	int backRanking;
	int backQ;
} qtable;

typedef vector<qtable> D_qtable;

typedef std::hash_map<string, D_qtable, hash<string>, eqstr> hash_string_Dqtable;
typedef vector<hash_string_Dqtable> D_hash_string_qtable;

inline bool DqSortedFn (qtable i, qtable j)
{
	return (i.score > j.score);
}

inline bool DbSortedFn (btable i, btable j)
{
	return (i.score > j.score);
}

inline std::string utf8_substr(std::string str, unsigned int start, unsigned int leng)
{
    if (leng==0) { return ""; }
    unsigned int c, i, ix, q, min=std::string::npos, max=std::string::npos;
    for (q=0, i=0, ix=str.length(); i < ix; i++, q++)
    {
        if (q==start){ min=i; }
        if (q<=start+leng || leng==std::string::npos){ max=i; }

        c = (unsigned char) str[i];
        if      (
                 //c>=0   &&
                 c<=127) i+=0;
        else if ((c & 0xE0) == 0xC0) i+=1;
        else if ((c & 0xF0) == 0xE0) i+=2;
        else if ((c & 0xF8) == 0xF0) i+=3;
        //else if (($c & 0xFC) == 0xF8) i+=4; // 111110bb //byte 5, unnecessary in 4 byte UTF-8
        //else if (($c & 0xFE) == 0xFC) i+=5; // 1111110b //byte 6, unnecessary in 4 byte UTF-8
        else return "";//invalid utf8
    }
    if (q<=start+leng || leng==std::string::npos){ max=i; }
    if (min==std::string::npos || max==std::string::npos) { return ""; }
    return str.substr(min,max-min);
}

typedef vector<string> vector_str;
typedef vector<vector_str> vector_2str;
typedef vector<vector_2str> vector_3str;

typedef struct DATA{
	vector_str alignedX;
	vector_str alignedY;
	vector_str unAlignedX;
	vector_str unAlignedY;
	vector<int> phraseSizeX;
	int	alignRank;
	double	alignScore;
	vector_2str feaVec;
	vector_str extraFea;
} data;

typedef struct DATAPLUS{
	vector<data>		mydata;
	map<long,double>	idAvg;	
} dataplus;

//typedef std::unordered_map<string, vector<data>, hash<string>, eqstr> hash_string_vData;
typedef std::hash_map<string, dataplus, hash<string>, eqstr> hash_string_vData;

typedef vector<dataplus> vector_vData;
