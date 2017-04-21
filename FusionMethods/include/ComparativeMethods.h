#pragma once
#include<vector>

extern std::string outDir;

namespace alpehnull {
	namespace core {
		namespace algo {

			class C_Methods {
				C_Methods() {
				}
				C_Methods(int experts) {
					setExperts(experts);
				}
				virtual void setExperts(int experts)
				{
					mExperts = experts;
				}
				virtual void initialize();
				virtual double getLoss() { return mLoss; }
				virtual void printStat();
				virtual void printStat(std::ofstream &file);
				virtual bool train(std::vector<std::vector<bool>>& expert_decisions, std::vector<bool>& actual_decisions);
				virtual bool predict(std::vector<bool>& expert_decisions);
				int mRounds;
			protected:
				int mExperts;
				std::vector<double> mProbability;
				std::vector<double> mWeights;
				double mLoss;
			};


		}
	}
}