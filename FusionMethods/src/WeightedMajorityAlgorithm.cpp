#include<..\include\WeightedMajorityAlgorithm.h>
#include<math.h>
#include<iostream>
#include<fstream>
#include<time.h>
#include<random>
using namespace alpehnull::core::algo;
#define BETA 0.5
#define EPSILON 0.1

double WeightedMajorityAlgorithm::updateRandomVersion(std::vector<bool>& expert_decisions, bool actual_decision)
{
	std::random_device rd;
	std::mt19937 e2(rd());
	std::uniform_real_distribution<> dist(0, 1);
	double l_current = 0;
	double total_weight = 0.0;
	bool best_action = false;
	for (int i = 0; i < mExperts; i++)
	{
		if (expert_decisions[i])
			l_current += mProbability[i];
		if (expert_decisions[i] != actual_decision)//Expert was wrong. Update weights.
		{
			mWeights[i] = mWeights[i] * exp(-BETA);
			mExpertLoss[i]++;
		}
		else
		{
			best_action = true;
		}
		total_weight += mWeights[i];
	}
	for (int i = 0; i < mExperts; i++)
	{
		mProbability[i] = mWeights[i] / total_weight;
	}
	bool decision = dist(e2) <= l_current ? true : false;
	bool res = (int)(decision != actual_decision);
	mLoss += res;
	if (!best_action)
	{
		mBestActionLoss += 1;
	}
	return res;
}

double WeightedMajorityAlgorithm::evaluateDistance(int curRound)
{
	double distance = 0;

	/*for (int i = 0; i < mConfidenceMatrix.size(); i++)
	{
		for (int j = 0; j < mExperts; j++)
		{
			double dist_rec = 1 - (std::log(mWeights[j])*(-2)) / (curRound);
			distance += std::pow(std::abs(mConfidenceMatrix[i][j] - dist_rec),2);
		}
	}*/
	for (int i = 0; i < mReconWt.size(); i++)
	{
		distance += std::pow((mReconWt[i]-mWeights[i]), 2);
	}

	return distance;
}

bool WeightedMajorityAlgorithm::updateWeights(std::vector<bool>& expert_decisions, bool actual_decision)
{
	double l_current = 0;
	double total_weight = 0.0;
	bool best_action = false;
	mReconWt.assign(mWeights.begin(), mWeights.end());
	for (int i = 0; i < mExperts; i++)
	{
		if (expert_decisions[i])
			l_current += mProbability[i];
		else
			l_current -= mProbability[i];
		if (expert_decisions[i] == actual_decision)
			best_action = true;
	}
	bool decision = l_current >= 0 ? true : false;
	int res = (int)(decision != actual_decision);
	mLoss += res;
	//if (decision != actual_decision)
	{
		for (int i = 0; i < mExperts; i++)
		{
			if (expert_decisions[i] != actual_decision)//Expert was wrong. Update weights.
			{
				mWeights[i] = mWeights[i] * exp(- BETA);
				mExpertLoss[i]++;
			}
			else
			{
				//mWeights[i] = mWeights[i] * exp(EPSILON);
			}
			total_weight += mWeights[i];
		}
		for (int i = 0; i < mExperts; i++)
		{
			mProbability[i] = mWeights[i] / total_weight;
		}
	}
	if (!best_action)
	{
		mBestActionLoss += 1;
	}
	return decision;
}

double WeightedMajorityAlgorithm::getBestExpertLoss()
{
	{
		mBestExpertLoss = mExpertLoss[0];
		for (int i = 1; i < mExperts; i++)
		{
			if (mBestExpertLoss > mExpertLoss[i])
				mBestExpertLoss = mExpertLoss[i];
		}
		return mBestExpertLoss;
	}
}

double WeightedMajorityAlgorithm::getBestActionLoss()
{
	return mBestActionLoss;
}

void WeightedMajorityAlgorithm::printStat()
{
	for (int i = 0; i < mExperts; i++)
	{
		std::cout << "Total Loss: " << mLoss << std::endl;
		std::cout << "Average Loss: " << (double)(mLoss / mRounds) << std::endl;
		std::cout << "Expert, "<<i<<"--- Weight:, "<<mWeights[i] << " ---- Probability:, "<<mProbability[i]<<std::endl;
	}
	
}

void WeightedMajorityAlgorithm::printStat(std::ofstream &file)
{
	file << "Total Loss: " << mLoss << std::endl;
	file << "Average Loss: " << (double)(mLoss / (mRounds+1)) << std::endl;
	file << "Expert, Weight, Probability, distribution_reconstructed" << std::endl;
	for (int i = 0; i < mExperts; i++)
	{
		double dist_rec = (std::log(mWeights[i])*(-2)) / (mRounds+1);
		file <<i <<","  << mWeights[i] << "," << mProbability[i] << ","<<dist_rec<<std::endl;
	}
	
}

void WeightedMajorityAlgorithm::initialize()
{
	mProbability.clear();
	mWeights.clear();
	mExpertLoss.clear();
	for (int i = 0; i < mExperts; i++)
	{
		mProbability.push_back(double(1.0 / (double)mExperts));
		mWeights.push_back(1);
		mExpertLoss.push_back(0);
	}
	mBestActionLoss = 0.0;
	mLoss = 0.0;
}
bool WeightedMajorityAlgorithm::train(std::vector<std::vector<bool>>& expert_decisions,std::vector<bool>& actual_decisions)
{
	std::string outfile = outDir + "\\stat_decision.csv";
	std::ofstream f;
	f.open(outfile);
	mRounds = (int)(actual_decisions.size());
	initialize();
	f << "WMA, avgLoss, currtotalLoss, BestActionloss,BestExpertLoss " << std::endl;
	for (int i = 0; i < mRounds; i++)
	{
		if ((int)(expert_decisions[i].size()) != mExperts)
			return false;
		updateWeights(expert_decisions[i], actual_decisions[i]);
		//updateRandomVersion(expert_decisions[i], actual_decisions[i]);
		f << i << "," << mLoss / (i+1) <<","<< mLoss <<"," << getBestActionLoss() << ","  << getBestExpertLoss() << std::endl;
	}
	f.close();
	return true;

}

bool WeightedMajorityAlgorithm::predict(std::vector<bool>& expert_decisions)
{
	if ((int)(expert_decisions.size()) != mExperts)
		return false;
	int final_result = 0;
	for (int i = 0; i < mExperts; i++)
	{
		if (expert_decisions[i])
			final_result += mProbability[i];
		else
			final_result -= mProbability[i];
	}
	bool result;
	if (final_result >= 0)
		result = true;
	else
		result = false;
	return result;
}

bool WeightedMajorityAlgorithm::converge(std::vector<std::vector<bool>>& expert_decisions, std::vector<bool>& actual_decisions,std::ofstream &file)
{
	mRounds = (int)(actual_decisions.size());
	initialize();
	file << "Rounds,Loss,avgLoss,bestLoss,distance, bestAction " << std::endl;
	for (int i = 0; i < mRounds; i++)
	{
		if ((int)(expert_decisions[i].size()) != mExperts)
			return false;
		double l = updateWeights(expert_decisions[i], actual_decisions[i]);
		auto dist = evaluateDistance(i+1);
		file << i << ","<<mLoss<<"," << mLoss/(i+1) <<"," << getBestExpertLoss() << "," << dist<<"," << getBestActionLoss() <<std::endl;
	}
	return true;
}

void WeightedMajorityAlgorithm::setConfidenceMatrix(std::vector<std::vector<int>> CM)
{
	mConfidenceMatrix.resize(CM[0].size());
	for (int j = 0; j < CM[0].size(); j++)
	{
		double total = 0;
		mConfidenceMatrix[j].resize(mExperts);
		for (int i = 0; i < mExperts; i++)
		{
			total += CM[i][j];
		}
		total = 100;
		for (int i = 0; i < mExperts; i++)
		{
			mConfidenceMatrix[j][i] = (double)((double)(CM[i][j]) / total);
		}
	}
}

void ContextWMA::initialize()
{
	for (int i = 0; i < mContext; i++)
	{
		mWMA[i].initialize();
		mWMA[i].mRounds = 0;
	}
	mLoss = 0.0;
}

void ContextWMA::printStat()
{
	std::cout << "Total Loss of CWMA: " << mLoss << std::endl;
	std::cout << "Average Loss of CWMA: " << (double)(mLoss / mRounds) << std::endl;
	for (int i = 0; i < mContext; i++)
	{
		std::cout << "Characteristics of experts in context " << i << std::endl;
		mWMA[i].printStat();

	}
}

void ContextWMA::printStat(std::ofstream &f)
{
	f << "Total Loss of CWMA: " << mLoss <<std::endl;
	f << "Average Loss of CWMA: " << (double)(mLoss / mRounds)<<std::endl;
	for (int i = 0; i < mContext; i++)
	{
		f<< "Characteristics of experts in context " << i << std::endl;
		mWMA[i].printStat(f);

	}
}

bool ContextWMA::updateWeights(std::vector<bool>& expert_decisions, bool actual_decision,int context)
{
	mWMA[context].mRounds++;
	return mWMA[context].updateWeights(expert_decisions, actual_decision);
	//return mWMA[context].updateRandomVersion(expert_decisions, actual_decision);
}

bool ContextWMA::train(std::vector<std::vector<bool>>& expert_decisions, std::vector<bool>& actual_decisions, std::vector<int>& contexts)
{
	std::string outfile = outDir + "\\stat_decision.csv";
	std::ofstream f;
	f.open(outfile, std::ios::app);
	if ((int)(contexts.size()) != (int)(actual_decisions.size()))
		return false;
	mRounds = (int)(contexts.size());
	initialize();
	f << "CWMA, avgLoss, currLoss,BestContextLoss" << std::endl;
	for (int i = 0; i < mRounds; i++)
	{
		if ((int)(expert_decisions[i].size()) != mExperts)
			return false;
		auto d = updateWeights(expert_decisions[i], actual_decisions[i],contexts[i]);
		if (d != actual_decisions[i])
			mLoss++;
		f << i << "," << mLoss / (i+1) <<","<< mLoss  << "," << getBestLoss() << std::endl;
	}
	f.close();
	return true;
}

bool ContextWMA::predict(std::vector<bool>& expert_decisions,int context)
{
	return mWMA[context].predict(expert_decisions);
}

double ContextWMA::getBestLoss()
{
	mTotalBestLoss = 0;
	for (int i = 0; i < mContext; i++)
		mTotalBestLoss += mWMA[i].getBestExpertLoss();
	return mTotalBestLoss;
}

bool ContextWMA::converge(std::vector<std::vector<bool>>& expert_decisions, std::vector<bool>& actual_decisions, std::vector<int>& contexts,std::ofstream &file)
{
	if ((int)(contexts.size()) != (int)(actual_decisions.size()))
		return false;
	mRounds = (int)(contexts.size());
	initialize();
	file << "mLoss,avgLoss,bestLoss,distance" << std::endl;
	for (int i = 0; i < mRounds; i++)
	{
		if ((int)(expert_decisions[i].size()) != mExperts)
			return false;
		double  l = updateWeights(expert_decisions[i], actual_decisions[i], contexts[i]);
		mLoss += l;
		auto dist = evaluateDistance(i+1);
		file << i <<","<<mLoss<<"," << mLoss/(i+1)<<","<<getBestLoss() << "," << dist <<std::endl;
	}
	return true;
}

double ContextWMA::evaluateDistance(int curRound)
{
	double distance = 0;
	/*for (int j = 0; j < mContext; j++)
	{
		auto weights = mWMA[j].getWeights();
		for (int i = 0; i < mExperts; i++)
		{
			double dist_rec = -(std::log(weights[i])*(-2)) / (mWMA[j].mRounds);
			distance += std::pow(std::abs(mConfidenceMatrix[j][i] - weights[i]), 2);
		}
	}*/
	for (int j = 0; j < mContext; j++)
	{
		distance += mWMA[j].evaluateDistance(curRound)/mContext;
	}

	return distance;
}

void ContextWMA::setConfidenceMatrix(std::vector<std::vector<int>> CM)
{
	mConfidenceMatrix.resize(mContext);
	for (int j = 0; j < mContext; j++)
	{
		double total = 0;
		mConfidenceMatrix[j].resize(mExperts);
		for (int i = 0; i < mExperts; i++)
		{
			total += CM[i][j];
		}
		total = 100;
		for (int i = 0; i < mExperts; i++)
		{
			mConfidenceMatrix[j][i] = (double)((double)(CM[i][j]) / total);
		}
	}
}