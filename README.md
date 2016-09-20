=DirecTL+ : String transduction model=

DirecTL+ is an online discriminative training model for string transduction problems.
More specifically, it has been applied to name transliteration and grapheme-to-phoneme conversion tasks. Please
see the list of known publications that utilized the DirecTL+.
In short, the model is trained with the Margin Infused Relaxed Algorithm (MIRA) or
known as the PA-I algorithm with the phrasal decoders (exact and Beam ones) in an online training framework.

DirecTL+ is implemented by [http://www.cs.ualberta.ca/~sj Sittichai Jiampojamarn] during the PhD's years at
[http://www.cs.ualberta.ca/ Department of Computing Science], [http://www.ualberta.ca University of Alberta].
The first version of this model, so called DirecTL, includes: context n-gram, 1st order Markov, and linear-chain
features. It was first introduced at ACL-08:

{{{
@InProceedings{jiampojamarn2008,
  author    = {Jiampojamarn, Sittichai  and  Cherry, Colin  and  Kondrak, Grzegorz},
  title     = {Joint Processing and Discriminative Training for Letter-to-Phoneme Conversion},
  booktitle = {Proceedings of ACL-08: HLT},
  month     = {June},
  year      = {2008},
  address   = {Columbus, Ohio},
  publisher = {Association for Computational Linguistics},
  pages     = {905--913},
  url       = {http://www.aclweb.org/anthology/P/P08/P08-1103}
}
}}}

Later, joint n-gram features were added for further improvements. So, we named it as DirecTL+ and described in

{{{
@InProceedings{jiampojamarn2010,
  author    = {Jiampojamarn, Sittichai and Cherry, Colin and Kondrak, Grzegorz},
  title     = {Integrating Joint n-gram Features into a Discriminative Training Framework},
  booktitle = {Proceedings of NAACL-2010},
  month     = {June},
  year      = {2010},
  address   = {Los Angeles, CA},
  publisher = {Association for Computational Linguistics}
}
}}}

You are welcome to use the code without any warranty; however, please acknowledge its use with a citation to the
ACL-08 paper and let me know.

==VERSIONS:==
{{{
        1.0: The first released version of DirecTL+ to public. All previous versions were for in-house users
available upon requests in the past.
}}}

==INSTALL:==

DirecTL+ has been tested on Linux systems with gcc version 4.1.2.
There are two required packages:

1. STLport can be downloaded from http://www.stlport.org/

2. SVMlight can be downloaded from http://svmlight.joachims.org/

Note, it requires a part of the SVMlight library. Please see the makefile for more detail.

You need to edit makefile for STLport and SVMlight locations in order to compile the codes.

To install, simply run "make"

==USAGE:==
{{{
./directlp  [--extFeaTest <string>] [--extFeaDev <string>]
               [--extFeaTrain <string>] [--jointFMgram <int>] [--beam]
               [--beamSize <int>] [--jointMgram <int>] [--noContextFea]
               [--SVMc <double>] [--alignLoss <minL|maxL|avgL|ascL|rakL
               |minS|maxS|mulA>] [--keepModel] [--outChar <string>]
               [--inChar <string>] [--nBestTest <int>] [--tam <int>] [--tal
               <int>] [--linearChain] [--order <int>] [--ng <int>] [--cs
               <int>] [--nBest <int>] [--mi <string>] [--mo <string>] [-a
               <string>] [-t <string>] [-d <string>] [-f <string>] [--]
               [--version] [-h]


Where:

   --extFeaTest <string>
     Extra feature for testing file (default null)

   --extFeaDev <string>
     Extra feature for dev file (default null)

   --extFeaTrain <string>
     Extra feature for training file (default null)

   --jointFMgram <int>
     Use joint forward M-gram features (default FM=0)

   --beam
     Use Beam search instead of Viterbi search (default false)

   --beamSize <int>
     Beam size (default 20)

   --jointMgram <int>
     Use joint M-gram features (default M=0)

   --noContextFea
     Do not use context n-gram features (default false)

   --SVMc <double>
     SVM c parameter (default 9999999)

   --alignLoss <minL|maxL|avgL|ascL|rakL|minS|maxS|mulA>
     Multiple-alignments loss computation criteria [minL, maxL, avgL, ascL,
     rakL, minS, maxS] (default minL)

   --keepModel
     Keep all trained model (default false)

   --outChar <string>
     Token delimeter output (default null)

   --inChar <string>
     Token delimeter string (default null)

   --nBestTest <int>
     Output n-best answers (default 1)

   --tam <int>
     Train at most n iteration (default 99)

   --tal <int>
     Train at least n iteration (default 1)

   --linearChain
     Linear chain features (default false)

   --order <int>
     Markov order (default 0)

   --ng <int>
     n-gram size (default 11)

   --cs <int>
     Context size (default 5)

   --nBest <int>
     n-best size for training (default 10)

   --mi <string>
     Model filename for testing

   --mo <string>
     Model output filename

   --mo <string>
     Model output filename

   -a <string>,  --answer <string>
     Answer output filename -- answer file

   -t <string>,  --testingFile <string>
     Testing filename -- testing file

   -d <string>,  --devFile <string>
     Development filename -- dev set

   -f <string>,  --trainingFile <string>
     Training filename -- alignment file

   --,  --ignore_rest
     Ignores the rest of the labeled arguments following this flag.

   --version
     Displays version information and exits.

   -h,  --help
     Displays usage information and exits.
}}}

==File formats:==

For training file (-f), it takes aligned examples each sub-alignment separated by a pipe (|), (:) separates each
token in the sub-alignment, a tab separates between source x and target y, one line per (x,y) pair.
You can use m2m-aligner to generate this alignment file, please see:

http://code.google.com/p/m2m-aligner/

or an example file:
trainEx.aligned

The development file (-d) is in the same format as the training file. It is used to determined when the training
should stop. If it isn't specified, the model will train until the performance drops.

Testing file (-t) is one line per test word, each token is separated by a pipe (|). Please example file:
testEx.words

==Example run:==

Training: Example run:

Training: ./directlp -f trainEx.aligned --inChar : --cs 3 --ng 7 --nBest 5 --tam 5

From this run, we train using "trainEx.aligned".
"--inChar :" indicates that each token in the sub-alignment is separated by ":".
"--cs 3" indicates the context size of 3.
"--ng 7" indicates the n-gram features of 7 (=cs `*` 2 + 1).
"--nBest 5" indicates that we're using 5-best to update the model.
"--tam 5" indicates that we train at most 5 iterations.

Output files from the above run:
trainEx.aligned.5nBest.5 : model file
trainEx.aligned.5nBest.5.limit : helper file to indicate the limited generations
trainEx.aligned.5nBest.5.maxX : helper file to indicate the maximum of phrase size.

Testing:  ./directlp --mi trainEx.aligned.5nBest.5 -t testEx.words --cs 3 --ng 7 --outChar ' ' -a
testEx.words.output
From this run, we test the model with "testEx.words" file.
"--mi trainEx.aligned.5nBest.5" read the model from the model file.
"-t testEx.words" read test words from the test file.
"--cs 3 --ng 7" using the same features as training.
"--outChar ' '" using a space to separate each output token.
"-a testEx.words.output" specify the output file.

Output files from the above run:
testEx.words.output.phraseOut : output file in detail with aligned productions, rank, and score.
testEx.words.output : output file in simple version.

==Acknowledgments:==
This work was supported by the Alberta Ingenuity and Informatics Circle of Research Excellence (iCORE)
throughout the Alberta Ingenuity Graduate Student Scholarship and iCORE ICT Graduate Student Scholarship.

==The list of known publications that utilized the DirecTL+:==
_(Please contact me to include your usage in this list)_

Sittichai Jiampojamarn, Kenneth Dwyer, Shane Bergsma, Aditya Bhargava,
Qing Dou, Mi-Young Kim, Grzegorz Kondrak "Transliteration generation and mining with limited training resources"
In Proceeding of the Named Entities Workshop (NEWS) 2010,
Sweden, July 2010.

Sittichai Jiampojamarn and Grzegorz Kondrak "Letter-Phoneme Alignment: An Exploration" In Proceeding of the 48th
Annual Meeting of the Association for Computational Linguistics (ACL 2010), Sweden, July 2010

Sittichai Jiampojamarn, Colin Cherry and Grzegorz Kondrak "Integrating Joint n-gram Features into a
Discriminative Training Framework" In Proceeding of the 11th Annual Conference of the North American Chapter of
the Association for Computational Linguistics (NAACL-HLT 2010), June 2010

Sittichai Jiampojamarn and Grzegorz Kondrak "Online Discriminative Training for Grapheme-to-Phoneme Conversion"
In
Proceeding of the 10th Annual Conference of the International Speech Communication Association (INTERSPEECH),
Brighton, U.K., September 2009, pp.1303-1306.

Sittichai Jiampojamarn, Aditya Bhargava, Qing Dou, Kenneth Dwyer and Grzegorz Kondrak "DIRECTL: a
Language-Independent Approach to Transliteration". In Proceedings of the 2009 Named Entities Workshop: Shared
Task on Transliteration (NEWS 2009), Singapore, August 2009, pp.28-31.

Qing Dou, Shane Bergsma, Sittichai Jiampojamarn and Grzegorz Kondrak "A Ranking Approach to Stress Prediction
for Letter-to-Phoneme Conversion". Proceedings of the Joint Conference of the 47th Annual Meeting of the ACL and
the 4th International Joint Conference on Natural Language Processing of the AFNLP, Singapore, August 2009,
pp.118-126.

Sittichai Jiampojamarn, Colin Cherry and Grzegorz Kondrak. "Joint Processing and Discriminative Training for
Letter-to-Phoneme Conversion". In Proceeding of the Annual Meeting of the Association for Computational
Linguistics: Human Language Technologies (ACL-08: HLT), Columbus, OH, June 2008, pp.905-913.


*Author: Sittichai Jiampojamarn*

*Date: May 19th, 2010*

http://code.google.com/p/directl-p
