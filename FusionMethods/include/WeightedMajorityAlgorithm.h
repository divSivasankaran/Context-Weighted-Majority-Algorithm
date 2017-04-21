#pragma once
#include<vector>

extern std::string outDir;

namespace alpehnull {
	namespace core {
		namespace algo {
			
			
			class WeightedMajorityAlgorithm {
			public:
				WeightedMajorityAlgorithm() {
				}
				WeightedMajorityAlgorithm(int experts) {
					setExperts(experts);
				}
				virtual void setExperts(int experts)
				{
					mExperts = experts;
				}
				virtual void initialize();
				virtual double getLoss() { return mLoss; }
				virtual double getBestActionLoss();
				virtual double getBestExpertLoss();
				virtual void printStat();
				virtual void printStat(std::ofstream &file);
				virtual bool updateWeights(std::vector<bool>& expert_decisions,bool actual_decision);
				virtual double updateRandomVersion(std::vector<bool>& expert_decisions, bool actual_decision);
				virtual bool train(std::vector<std::vector<bool>>& expert_decisions, std::vector<bool>& actual_decisions);
				virtual bool predict(std::vector<bool>& expert_decisions);
				void setConfidenceMatrix(std::vector<std::vector<int>> confidenceMatrix);
				bool converge(std::vector<std::vector<bool>>& expert_decisions, std::vector<bool>& actual_decisions, std::ofstream& file);
				double evaluateDistance(int);
				std::vector<double> getWeights() { return mWeights; }
				std::vector<double> getReconWt() { return mReconWt; }
				int mRounds;
				protected:
					int mExperts;
					std::vector<double> mProbability;
					std::vector<double> mWeights;
					std::vector<double> mExpertLoss;
					double mLoss, mBestActionLoss, mBestExpertLoss;
					std::vector<std::vector<double>> mConfidenceMatrix;
					std::vector<double> mReconWt;
			};

			class ContextWMA{
			public:
				ContextWMA()
				{
				}
				ContextWMA(int experts,int contexts){
					setExperts(experts, contexts);
				}
				virtual void setExperts(int experts, int contexts) {
					mExperts = experts;
					mContext = contexts;
					for (int i = 0; i < mContext; i++)
					{
						WeightedMajorityAlgorithm t(mExperts);
						mWMA.push_back(t);
					}
				}
				virtual double getLoss() { return mLoss; }
				virtual double getBestLoss();
				virtual void initialize();
				virtual bool updateWeights(std::vector<bool>& expert_decisions, bool actual_decision,int context);
				virtual bool train(std::vector<std::vector<bool>>& expert_decisions, std::vector<bool>& actual_decisions, std::vector<int>& contexts);
				virtual bool predict(std::vector<bool>& expert_decisions, int context);
				virtual void printStat();
				virtual void printStat(std::ofstream &file);
				double evaluateDistance(int);
				void setConfidenceMatrix(std::vector<std::vector<int>> confidenceMatrix);
				bool converge(std::vector<std::vector<bool>>& expert_decisions, std::vector<bool>& actual_decisions, std::vector<int>& contexts, std::ofstream& file);
			protected:
				int mExperts,mContext, mRounds;
				std::vector<WeightedMajorityAlgorithm> mWMA;
				double mLoss, mTotalBestLoss;
				std::vector<std::vector<double>> mConfidenceMatrix;
			};

}}}