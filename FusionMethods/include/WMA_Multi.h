#pragma once
#include<vector>
//This can be used to decide multi-class decisions or score level fusion.
//For multi-class, set mode to MULTI and set mClasses to the # of classes 
//For score level fusion, set mode to SCORE and send confidence values in the range 0-100, and set mThreshold 0<t<100 as well.
extern std::string outDir;
namespace alpehnull {
	namespace core {
		namespace algo {
			enum MODE {MULTI,SCORE};
			class WMA_Multi {
			public:
				WMA_Multi() {
				}
				WMA_Multi(int experts) {
					setExperts(experts);
					setMode(SCORE);
				}
				virtual void setExperts(int experts)
				{
					mExperts = experts;
				}
				virtual void setMode(MODE mode) { mMode = mode; }
				virtual MODE getMode() { return mMode; }
				virtual void setThreshold(int t) { mThreshold = t; }
				virtual void setClasses(int c) { mClasses = c; }
				virtual void initialize();
				virtual double getLoss() { return mLoss; }
				virtual double getBestActionLoss();
				virtual double getBestExpertLoss();
				virtual void printStat();
				virtual void printStat(std::ofstream &file);
				virtual double updateWeights(std::vector<int>& expert_decisions, bool actual_decision);
				virtual double updateRandomVersion(std::vector<int>& expert_decisions, bool actual_decision);
				virtual bool train(std::vector<std::vector<int>>& expert_decisions, std::vector<bool>& actual_decisions);
				virtual bool predict(std::vector<int>& expert_decisions);
				void setConfidenceMatrix(std::vector<std::vector<int>> confidenceMatrix);
				bool converge(std::vector<std::vector<int>>& expert_decisions, std::vector<bool>& actual_decisions, std::ofstream& file);
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
					int mThreshold;
					MODE mMode;
			};

			class ContextWMA_Multi{
			public:
				ContextWMA_Multi()
				{
				}
				ContextWMA_Multi(int experts,int contexts){
					setExperts(experts, contexts);
				}
				virtual void setExperts(int experts, int contexts) {
					mExperts = experts;
					mContext = contexts;
					for (int i = 0; i < mContext; i++)
					{
						WMA_Multi t(mExperts);
						mWMA.push_back(t);
					}
				}
				virtual void setMode(MODE mode) { mMode = mode; 
				for (int i = 0; i < mContext; i++)
				{
					mWMA[i].setMode(mode);
				}
				}
				virtual MODE getMode() { return mMode; }
				virtual void setThreshold(int t) { mThreshold = t; 
				for (int i = 0; i < mContext; i++)
				{
					mWMA[i].setThreshold(t);
				}
				}
				virtual void setClasses(int c) { mClasses = c; 
				for (int i = 0; i < mContext; i++)
				{
					mWMA[i].setClasses(c);
				}
				}
				virtual double getLoss() { return mLoss; }
				virtual double getBestLoss();
				virtual void initialize();
				virtual double updateWeights(std::vector<int>& expert_decisions, bool actual_decision,int context);
				virtual bool train(std::vector<std::vector<int>>& expert_decisions, std::vector<bool>& actual_decisions, std::vector<int>& contexts);
				virtual bool predict(std::vector<int>& expert_decisions, int context);
				virtual void printStat();
				virtual void printStat(std::ofstream &file);
				double evaluateDistance(int);
				void setConfidenceMatrix(std::vector<std::vector<int>> confidenceMatrix);
				bool converge(std::vector<std::vector<int>>& expert_decisions, std::vector<bool>& actual_decisions, std::vector<int>& contexts, std::ofstream& file);
			protected:
				int mExperts,mContext, mRounds;
				std::vector<WMA_Multi> mWMA;
				double mLoss, mTotalBestLoss;
				std::vector<std::vector<double>> mConfidenceMatrix;
				int mClasses;
				int mThreshold;
				MODE mMode;
			};

}}}