#pragma once
#include<vector>
#include<WMA_Multi.h>
#include<WeightedMajorityAlgorithm.h>

extern std::string outDir;

namespace alpehnull {
	namespace core {
		namespace algo {
			enum CMODE {sum, max,product, wma, cwma,wms,cwms,exp1,exp2,exp3};
			class C_Methods {
			public:
				C_Methods() {
					mBeginT = 0;
					mEndT = 0;
					mStepSize = 0.05;
				}
				C_Methods(double start, double end, int steps) {
					mBeginT = start/100;
					mEndT = end/100;
					mStepSize = (mEndT - mBeginT)/steps;
				}
				bool evaluate(int i,std::vector<std::vector<int>>& expert_decisions, std::vector<bool>& actual_decisions, std::vector<int>& contexts);
				double evaluate_sum(std::vector<int>& scores);
				double evaluate_max(std::vector<int>& scores);
				double evaluate_product(std::vector<int>& scores);
				int mContext;
			private:
				WeightedMajorityAlgorithm mWMA;
				ContextWMA mCWMA;
				WMA_Multi mWS;
				ContextWMA_Multi mCWS;
				double mBeginT, mEndT, mStepSize;
			};
		}
	}
}