#pragma once
#include<vector>
#include<WMA_Multi.h>
#include<WeightedMajorityAlgorithm.h>

extern std::string outDir;

namespace alpehnull {
	namespace core {
		namespace algo {
			enum CMODE {sum, max,product, wma, cwma,wms,cwms};
			class C_Methods {
			public:
				C_Methods() {
				}
				bool evaluate(CMODE mode,std::vector<std::vector<int>>& expert_decisions, std::vector<bool>& actual_decisions, std::vector<int>& contexts);
				double evaluate_sum(std::vector<int>& scores);
				double evaluate_max(std::vector<int>& scores);
				double evaluate_product(std::vector<int>& scores);
				int mContext;
			private:
				WeightedMajorityAlgorithm mWMA;
				ContextWMA mCWMA;
				WMA_Multi mWS;
				ContextWMA_Multi mCWS;
			};
		}
	}
}