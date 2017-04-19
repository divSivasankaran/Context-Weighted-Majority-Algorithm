#pragma once
#include<fstream>
#include<WMA_Multi.h>
#include<Test_SynData.h>

extern std::string outDir;

namespace alpehnull {
	namespace core {
		namespace algo {


class Test_SynDataM
{
public:
	Test_SynDataM()
	{
		Test_SynDataM(1, 8, 24,5);
	}
	Test_SynDataM(int experts, int context, int rounds,int classes)
	{
		mExperts = experts;
		mRounds = rounds;
		mContexts = context;
		mWMA.setExperts(experts,classes);
		mCWMA.setExperts(experts, context,classes);
		bias = mContexts; //fully biased
		bias_percentage = 100;
		mType = Data_Dist::Biased;
		mClasses = classes;
	}
	void generateData(Data_Dist type);
	void gen_test();
	void runTests();
	void runConvergence(std::ofstream &ofile);
	void printStat(std::ofstream &mOutFile);
	double getWMALoss() { return mWMA.getLoss(); }
	double getCWMALoss() { return mCWMA.getLoss(); }
	double getBestExpertLoss();
	double getBestActionLoss();
	double getBestContextLoss();
	int bias, bias_percentage;
	int mType;
protected:
	WMA_Multi mWMA;
	ContextWMA_Multi mCWMA;
	int mExperts, mRounds, mContexts;
	double mLoss_total, mLoss_Avg,mLossBest;
	std::vector<std::vector<int>> expert_decisions;
	std::vector<int> actual_decisions;
	std::vector<int> contexts;
	std::vector<std::vector<int>> bias_factor;
	int mClasses;
};

		}
	}
}