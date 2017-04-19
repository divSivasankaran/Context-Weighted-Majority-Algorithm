#pragma once
#include<vector>

extern std::string outDir;
namespace alpehnull {
	namespace core {
		namespace algo {

			class WMA_Multi {
			public:
				WMA_Multi() {
				}
				WMA_Multi(int experts, int classes) {
					setExperts(experts,classes);
				}
				virtual void setExperts(int experts,int classes)
				{
					mExperts = experts;
					mClasses = classes;
				}
				virtual void initialize();
				virtual double getLoss() { return mLoss; }
				virtual double getBestActionLoss();
				virtual double getBestExpertLoss();
				virtual void printStat();
				virtual void printStat(std::ofstream &file);
				virtual double updateWeights(std::vector<int>& expert_decisions,int actual_decision);
				virtual double updateRandomVersion(std::vector<int>& expert_decisions, int actual_decision);
				virtual bool train(std::vector<std::vector<int>>& expert_decisions, std::vector<int>& actual_decisions);
				virtual bool predict(std::vector<int>& expert_decisions);
				void setConfidenceMatrix(std::vector<std::vector<int>> confidenceMatrix);
				bool converge(std::vector<std::vector<int>>& expert_decisions, std::vector<int>& actual_decisions, std::ofstream& file);
				double evaluateDistance(int);
				std::vector<double> getWeights() { return mWeights; }

				int mRounds;
				protected:
					int mExperts;
					std::vector<double> mProbability;
					std::vector<double> mWeights;
					std::vector<double> mExpertLoss;
					double mLoss, mBestActionLoss, mBestExpertLoss;
					std::vector<std::vector<double>> mConfidenceMatrix;
					int mClasses;
			};

			class ContextWMA_Multi{
			public:
				ContextWMA_Multi()
				{
				}
				ContextWMA_Multi(int experts,int contexts,int classes){
					setExperts(experts, contexts,classes);
				}
				virtual void setExperts(int experts, int contexts,int classes) {
					mExperts = experts;
					mContext = contexts;
					mClasses = classes;
					for (int i = 0; i < mContext; i++)
					{
						WMA_Multi t(mExperts,classes);
						mWMA.push_back(t);
					}
				}
				virtual double getLoss() { return mLoss; }
				virtual double getBestLoss();
				virtual void initialize();
				virtual double updateWeights(std::vector<int>& expert_decisions, int actual_decision,int context);
				virtual bool train(std::vector<std::vector<int>>& expert_decisions, std::vector<int>& actual_decisions, std::vector<int>& contexts);
				virtual bool predict(std::vector<int>& expert_decisions, int context);
				virtual void printStat();
				virtual void printStat(std::ofstream &file);
				double evaluateDistance(int);
				void setConfidenceMatrix(std::vector<std::vector<int>> confidenceMatrix);
				bool converge(std::vector<std::vector<int>>& expert_decisions, std::vector<int>& actual_decisions, std::vector<int>& contexts, std::ofstream& file);
			protected:
				int mExperts,mContext, mRounds;
				std::vector<WMA_Multi> mWMA;
				double mLoss, mTotalBestLoss;
				std::vector<std::vector<double>> mConfidenceMatrix;
				int mClasses;
			};

}}}