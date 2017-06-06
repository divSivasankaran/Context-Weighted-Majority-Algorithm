#include<..\include\ComparativeMethods.h>
#include<fstream>
#include<string>
#include<iostream>
#include<math.h>

using namespace alpehnull::core::algo;
using namespace std;
bool C_Methods::evaluate(int i, std::vector<std::vector<int>>& expert_decisions, std::vector<bool>& actual_decisions, std::vector<int>& contexts)
{
	for (int k = 0; k < 9; k++)
	{
		std::string outfile = outDir + "\\roc_" +  std::to_string(k) + std::to_string(i)  +".csv";
		std::ofstream f;
		f.open(outfile);
		for (double threshold = mBeginT; threshold <= mEndT; threshold += mStepSize)
		{
			for (int mode = 0; mode < 10; mode++)
			{
				if (mode == CMODE::wma)
				{
					mWMA.setExperts(expert_decisions[0].size());
					mWMA.initialize();
				}
				else if (mode == CMODE::cwma)
				{
					mCWMA.setExperts(expert_decisions[0].size(), mContext);
					mCWMA.initialize();
				}
				else if (mode == CMODE::wms)
				{
					mWS.setExperts(expert_decisions[0].size());
					mWS.setThreshold(threshold * 100);
					mWS.setMode(MODE::SCORE);
					mWS.initialize();
				}
				else if (mode == CMODE::cwms)
				{
					mCWS.setExperts(expert_decisions[0].size(), mContext);
					mCWS.setThreshold(threshold * 100);
					mCWS.setMode(MODE::SCORE);
					mCWS.initialize();
				}
				double FRR = 0;
				double FAR = 0;
				double genuine = 0;
				double imposter = 0;
				int rounds = actual_decisions.size();
				for (int i = 0; i < rounds; i++)
				{
					if(k!=8)
						if (contexts[i] != k)
							continue;

					auto scores = expert_decisions[i];
					auto true_decision = actual_decisions[i];
					bool fused_decision = false;
					std::vector<bool> e_d;
					if ((mode == CMODE::wma) || (mode == CMODE::cwma))
						for (int j = 0; j < scores.size(); j++)
						{
							if (scores[j] < threshold * 100)
							{
								e_d.push_back(true);
							}
							else
							{
								e_d.push_back(false);
							}
						}
					double score;
					switch (mode)
					{
					case CMODE::sum: score = evaluate_sum(scores);
						if (score < threshold)
							fused_decision = true;
						break;
					case CMODE::max:
						score = evaluate_max(scores);
						if (score < threshold)
							fused_decision = true;
						break;
					case CMODE::product:
						score = evaluate_product(scores);
						if (score < threshold / 5)
							fused_decision = true;
						break;
					case CMODE::wma:
						fused_decision = mWMA.updateWeights(e_d, true_decision);
						break;
					case CMODE::cwma:
						fused_decision = mCWMA.updateWeights(e_d, true_decision, contexts[i]);
						break;
					case CMODE::wms:
						fused_decision = mWS.updateWeights(scores, true_decision);
						break;
					case CMODE::cwms:
						fused_decision = mCWS.updateWeights(scores, true_decision, contexts[i]);
						break;
					case CMODE::exp1:
						if (scores[0] < threshold * 100)
							fused_decision = true;
						break;
					case CMODE::exp2:
						if (scores[1] < threshold * 100)
							fused_decision = true;
						break;
					case CMODE::exp3:
						if (scores[2] < threshold * 100)
							fused_decision = true;
						break;
					default:
						score = evaluate_sum(scores);
					}
					if (true_decision)
						genuine += 1;
					else
						imposter += 1;
					if (fused_decision && !true_decision) //We accept when it is an imposter
						FAR++;
					if (!fused_decision && true_decision) //We reject when it is the user
						FRR++;
				}
				f << (FAR / imposter) * 100 << "," << (FRR / genuine) * 100 << ",";
			}
			f << std::endl;
		}
		f.close();
	}
	return true;
}

double C_Methods::evaluate_sum(std::vector<int>& scores)
{
	double max = scores.size() * 100;
	double min = 0;
	double sum = 0;
	for (int i = 0; i < scores.size(); i++)
	{
		sum += scores[i];
	}

	//normalize score to range 0-1
	sum = (sum - min) / (max - min);
	return sum;
}

double C_Methods::evaluate_max(std::vector<int>& scores)
{
	double abs_max = 100;
	double min = 0;
	double max = 0;
	for (int i = 0; i < scores.size(); i++)
	{
		if (max <= scores[i])
		{
			max = scores[i];
		}
	}
	//normalize score to range 0-1
	max = (max - min) / (abs_max - min);
	return max;
}

double C_Methods::evaluate_product(std::vector<int>& scores)
{
	double max = pow(100, scores.size());
	double min = 1;
	double prod = 1;
	for (int i = 0; i < scores.size(); i++)
	{
		prod *= scores[i];
	}
	//normalize score to range 0-1
	prod = (prod - min) / (max - min);
	return prod;
}