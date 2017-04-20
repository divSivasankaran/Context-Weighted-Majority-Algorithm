#include<..\include\SynDataMulti.h>
#include<random>
#include<math.h>
#include<time.h>
#include<iostream>
#include<string>
#include<iomanip>
#include<set>
using namespace alpehnull::core::algo;

void Test_SynDataM::generateData(Data_Dist type)
{
	srand(time(NULL));
	std::vector<std::set<int>> bias_list;
	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution(0, 1000);
	if (type == Data_Dist::Random)
	{	
		for (int i = 0; i < mRounds; i++)
		{
			std::vector<int> eD;
			int decision;
			decision = (rand() % 100) % mClasses;
			actual_decisions.push_back(decision);
			for (int j = 0; j < mExperts; j++)
			{
				int d = (rand() % 100) % mClasses ;
				eD.push_back(d);
			}
			int context = distribution(generator) % mContexts;
			contexts.push_back(context);
			expert_decisions.push_back(eD);
		}
	}
	else if (type == Data_Dist::Biased)
	{
		gen_test();
	}
	
}

void Test_SynDataM::gen_test()
{
	srand(time(NULL));
	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution(0, 1000);
	std::vector<std::uniform_int_distribution<int>> prob;
	bias_factor.resize(mExperts);
	for (int i = 0; i < mExperts; i++)
	{
		std::uniform_int_distribution<int> p(0, 100);
		prob.push_back(p);
		std::cout << "Bias for expert "<< i << " in context :";
		bias_factor[i].resize(mContexts);
		for (int j = 0; j < mContexts; j++)
		{
			int num = rand() % 100;
			if (mType == Data_Dist::Biased && num <= bias_percentage)
			{
				bias_factor[i][j] = rand()%100;
				std::cout << j << "-" <<bias_factor[i][j]<<",";
			}
			else
			{
				bias_factor[i][j] = 50;
 			}
		}
		std::cout << std::endl;
	}


	for (int i = 0; i < mRounds; i++)
	{
		std::vector<int> eD;
		bool decision;
		decision = (rand() % 100) % 2==0?0:1/*mClasses;*/;
		actual_decisions.push_back(decision);
		int context = distribution(generator) % mContexts;
		contexts.push_back(context);
		for (int j = 0; j < mExperts; j++)
		{
			int d;
			int var = prob[j](generator);
			if (var <= bias_factor[j][context])
			{
				d = decision *80 + rand() % 10;
			}
			else
			{
				d = (rand() % 100) % mClasses;
			}
			eD.push_back(d);
		}
		expert_decisions.push_back(eD);
	}
}

void Test_SynDataM::printStat(std::ofstream &mOutFile)
{
	mOutFile << "TEST DATA: Experts, contexts, rounds"<<std::endl;
	mOutFile << mExperts << ", " << mContexts <<"," << mRounds<<std::endl;
	mOutFile << "actual, context, expert 1, .. expert n"<<std::endl;
	for (int i = 0; i < mRounds; i++)
	{
		mOutFile << actual_decisions[i] << "," << contexts[i];
		for (int j = 0; j < mExperts; j++)
		{
			mOutFile << "," << expert_decisions[i][j];
		}
		mOutFile << std::endl;
	}

	mOutFile << "Context,Expert,Bias" << std::endl;
	for (int i = 0; i < mContexts; i++)
	{
		for (int j = 0; j < mExperts; j++)
			mOutFile << i << ","<< j << "," << bias_factor[j][i] << std::endl;
	}
	

	mOutFile << "WMA RESULTS" << std::endl;
	mWMA.printStat(mOutFile);
	mOutFile << "ContextWMA RESULTS" << std::endl;
	mCWMA.printStat(mOutFile);
}

void Test_SynDataM::runTests()
{
	mType = Data_Dist::Random;
	gen_test();
	mWMA.train(expert_decisions, actual_decisions);
	mCWMA.train(expert_decisions, actual_decisions, contexts);
	//printStat();
}

void Test_SynDataM::runConvergence(std::ofstream& ofile)
{
	gen_test();
	mWMA.setConfidenceMatrix(bias_factor);
	mCWMA.setConfidenceMatrix(bias_factor); 
	std::string outfile = outDir + "\\experiments\\Convergence_Stat.csv";
	std::ofstream f;
	f.open(outfile);
	for (int i = 0; i < 10; i++)
	{
		ofile << "TRIAL RUN " << i << std::endl;
		mWMA.converge(expert_decisions, actual_decisions, ofile);
		//mWMA.printStat(ofile);
		mCWMA.converge(expert_decisions, actual_decisions, contexts, ofile);
		//mCWMA.printStat(ofile);
		printStat(f);
	}
}
double Test_SynDataM::getBestActionLoss()
{
	return mWMA.getBestActionLoss();
}

double Test_SynDataM::getBestExpertLoss()
{
	return mWMA.getBestExpertLoss();
}

double Test_SynDataM::getBestContextLoss()
{
	return mCWMA.getBestLoss();
}