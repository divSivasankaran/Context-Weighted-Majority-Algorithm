#pragma once
#include<fstream>
#include<WeightedMajorityAlgorithm.h>

extern std::string outDir;

namespace alpehnull {
	namespace core {
		namespace algo {

			enum Data_Dist {Random, Biased};

class Test_SynData
{
public:
	Test_SynData()
	{
		Test_SynData(1, 8, 24);
	}
	Test_SynData(int experts, int context, int rounds)
	{
		mExperts = experts;
		mRounds = rounds;
		mContexts = context;
		mWMA.setExperts(experts);
		mCWMA.setExperts(experts, context);
		bias = mContexts; //fully biased
		bias_percentage = 100;
		mType = Data_Dist::Biased;
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
	WeightedMajorityAlgorithm mWMA;
	ContextWMA mCWMA;
	int mExperts, mRounds, mContexts;
	double mLoss_total, mLoss_Avg,mLossBest;
	std::vector<std::vector<bool>> expert_decisions;
	std::vector<bool> actual_decisions;
	std::vector<int> contexts;
	std::vector<std::vector<int>> bias_factor;
};

		}
	}
}