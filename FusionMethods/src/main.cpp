#include <iostream>
#include<conio.h>
#include<fstream>
#include<string>
#include <iterator>
#include <sstream>
#include<map>
#include<SynDataMulti.h>
#include <Test_SynData.h>
#include<ComparativeMethods.h>

using namespace alpehnull::core::algo;
//std::string outDir = "C:\\Users\\e0013178\\Documents\\GitHub\\CWMA\\FusionMethods\\output";
std::string outDir = "C:\\Users\\div_1\\OneDrive\\Documents\\GitHub\\CWMA\\FusionMethods\\output";


#define e_repeat 1

void test_WMA()
{
	WeightedMajorityAlgorithm test(3);
	std::vector<std::vector<bool>> expert_decisions;
	bool e[3][8] ={ { true, true, false, false, true, true, false, false}, { true, true, true, false, false, false, false, false }, { false, false, false, true, true, true, true, true }
};
	bool actual[] = { true, true, true, true, true, true, true, true };
	std::cout << "Testing WMA Algorithm:" << std::endl;
	std::vector<bool> actual_decisions;
	for (int i = 0; i < 8; i++)
	{
		actual_decisions.push_back(actual[i]);
		std::vector<bool> temp;
		for (int j = 0; j < 3; j++)
		{
			temp.push_back(e[j][i]);
		}
		expert_decisions.push_back(temp);
	}
	test.train(expert_decisions, actual_decisions);
	test.printStat();
	std::cout <<std::endl<<"Total Loss = " << test.getLoss() << std::endl;
}
void test_CWMA()
{
	ContextWMA test(3,2);
	std::vector<std::vector<bool>> expert_decisions;
	bool e[3][8] = { { true, true, false, false, true, true, false, false },{ true, true, true, false, false, false, false, false },{ false, false, false, true, true, true, true, true }
	};
	bool actual[] = { true, false, true, false, false, true, false, true };
	bool context[] = { 0,0,0,0,1,1,1,1 };
	std::cout << "Testing ContextWMA Algorithm:" << std::endl;
	std::vector<bool> actual_decisions;
	std::vector<int> contexts;
	for (int i = 0; i < 8; i++)
	{
		actual_decisions.push_back(actual[i]);
		contexts.push_back(context[i]);
		std::vector<bool> temp;
		for (int j = 0; j < 3; j++)
		{
			temp.push_back(e[j][i]);
		}
		expert_decisions.push_back(temp);
	}
	test.train(expert_decisions, actual_decisions,contexts);
	test.printStat();
	std::cout << std::endl << "Total Loss = " << test.getLoss() << std::endl;
	
}

void e_Experts()
{
	int contexts = 10;
	int rounds = 100;
	//std::vector<std::pair<int, double>> wmaLoss, cwmaLoss;
	std::string outfile = outDir + "\\experiments\\VaryExpert.csv";
	std::ofstream f;
	f.open(outfile);
	f << "Context, Rounds: " << contexts << "," << rounds << std::endl;
	f << "expert, wmaLoss, cwmaLoss, bestActionLoss, bestExpertloss" << std::endl;
	for (int i = 1; i < 60; i += 2)
	{
		double avgLoss_wma = 0;
		double avgLoss_cwma = 0;
		double avgBestAction = 0;
		double avgBestExpert = 0;
		double avgBestContext = 0;
		for (int j = 0; j < e_repeat; j++)
		{
			Test_SynData t(i, contexts, rounds);
			t.runTests();
			avgLoss_cwma += t.getCWMALoss();
			avgLoss_wma += t.getWMALoss();
			avgBestAction += t.getBestActionLoss();
			avgBestExpert += t.getBestExpertLoss();
			avgBestContext += t.getBestContextLoss();
		}
		f << i << "," << (double)(avgLoss_wma / e_repeat) << "," << (double)(avgLoss_cwma / e_repeat) << "," << (double)(avgBestAction / e_repeat) << "," << (double)(avgBestExpert / e_repeat) << "," << (double)(avgBestContext / e_repeat) << std::endl;
	}
	
	f.close();
}
void e_Rounds()
{
	int experts = 5;
	int contexts = 10;
	int rounds = 500;

	std::string outfile = outDir + "\\experiments\\VaryRounds.csv";
	std::ofstream f;
	f.open(outfile);
	f << "Context, experts: ," << contexts << "," << experts <<std::endl;
	f << "Rounds, wmaLoss, cwmaLoss, bestActionLoss, bestExpertloss, bestContextLoss" << std::endl;
	for (int i = 0; i < 1; i +=5)
	{
		double avgLoss_wma = 0;
		double avgLoss_cwma = 0;
		double avgBestAction = 0, avgBestExpert = 0, avgBestContext = 0;
		for (int j = 0; j < e_repeat; j++)
		{
			Test_SynData t(experts, contexts, rounds);
			t.runTests();
			avgLoss_cwma += t.getCWMALoss();
			avgLoss_wma += t.getWMALoss();
			avgBestAction += t.getBestActionLoss();
			avgBestExpert += t.getBestExpertLoss();
			avgBestContext += t.getBestContextLoss();
		}
		f << i << "," << (double)(avgLoss_wma / e_repeat) << "," << (double)(avgLoss_cwma / e_repeat) << "," << (double)(avgBestAction / e_repeat) <<"," << (double) (avgBestExpert / e_repeat) << "," << (double)(avgBestContext / e_repeat) << std::endl;
	}
	f.close();
}


void e_Contexts()
{
	int experts = 10;
	int contexts = 1;
	int rounds = 100;

	//std::vector<std::pair<int, double>> wmaLoss, cwmaLoss;
	std::string outfile = outDir + "\\experiments\\VaryContexts.csv";
	std::ofstream f;
	f.open(outfile);
	f << "Rounds, experts: " << rounds << "," << experts << std::endl;
	f << "context, wmaLoss, cwmaLoss, bestActionLoss, bestExpertloss" << std::endl;
	for (int i = contexts; i < 100; i += 4)
	{
		double avgLoss_wma = 0;
		double avgLoss_cwma = 0;
		double avgBestAction = 0, avgBestExpert = 0, avgBestContext = 0;
		for (int j = 0; j < e_repeat; j++)
		{
			Test_SynData t(experts, i, rounds);
			t.runTests();
			avgLoss_cwma += t.getCWMALoss();
			avgLoss_wma += t.getWMALoss();
			avgBestAction += t.getBestActionLoss();
			avgBestExpert += t.getBestExpertLoss();
			avgBestContext += t.getBestContextLoss();
		}
		f << i << "," << (double)(avgLoss_wma / e_repeat) << "," << (double)(avgLoss_cwma / e_repeat) << "," << (double)(avgBestAction / e_repeat) << "," << (double)(avgBestExpert / e_repeat) << "," << (double)(avgBestContext / e_repeat) << std::endl;
	}
	f.close();
}


void e_Bias_WithNoise()
{
	int experts = 10;
	int contexts = 20;
	int rounds = 400;

	std::string outfile = outDir + "\\experiments\\VaryBias.csv";
	std::ofstream f;
	f.open(outfile);
	f << "Context, Rounds, experts: " << contexts << "," << rounds << "," << experts << std::endl;
	f << "bias_percentage, wmaLoss, cwmaLoss, bestActionLoss, bestExpertloss" << std::endl;
	for (int i = 0; i < 100; i += 10)
	{
		double avgLoss_wma = 0;
		double avgLoss_cwma = 0;
		double avgBestAction = 0, avgBestExpert = 0;
		double avgBestContext = 0;
		for (int j = 0; j < e_repeat; j++)
		{
			Test_SynData t(experts, contexts, rounds);
			t.bias_percentage = i;
			t.runTests();
			avgLoss_cwma += t.getCWMALoss();
			avgLoss_wma += t.getWMALoss();
			avgBestAction += t.getBestActionLoss();
			avgBestExpert += t.getBestExpertLoss();
			avgBestContext += t.getBestContextLoss();
		}
		f << i << "," << (double)(avgLoss_wma / e_repeat) << "," << (double)(avgLoss_cwma / e_repeat) << "," << (double)(avgBestAction / e_repeat) << "," << (double)(avgBestExpert / e_repeat) << "," << (double)(avgBestContext / e_repeat) << std::endl;
	}
	f.close();
}


void convergenceTest()
{
	std::string outfile = outDir + "\\experiments\\Convergence_Test.csv";
	std::ofstream f;
	int experts = 5;
	int contexts = 5;
	int rounds = 400;
	f.open(outfile);
	f << "Context, experts: " << contexts << "," << experts << std::endl;
	Test_SynData t(experts, contexts, rounds);
	//t.bias_percentage = 90;
	t.runConvergence(f);
	f.close();
}

template<typename Out>
void split(const std::string &s, char delim, Out result) {
	std::stringstream ss;
	ss.str(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		*(result++) = item;
	}
}

std::vector<int> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	std::vector<int> res;
	split(s, delim, std::back_inserter(elems));
	for (int i = 0; i < elems.size(); i++)
	{
		res.push_back(std::stoi(elems[i]));
	}
	return res;
}

void real_data()
{

	std::string outfile = outDir + "\\experiments\\pie_test.csv";
	std::ofstream of;
	int experts = 3;
	int contexts = 3;
	int rounds = 285;
	std::vector<int> context_data;
	std::vector<bool> actual_decisions;
	std::vector<std::vector<bool>> expert_decisions;

	//expert_decisions.resize(experts);
	//context_data.resize(rounds);
	//actual_decisions.resize(rounds);

	/*for (int i = 0; i < experts; i++)
	{
		expert_decisions[i].resize(rounds);
	}*/

	//Read & transfer data
	std::string infile = outDir + "\\pie_data.csv";
	std::ifstream f;
	f.open(infile);
	std::string str;
	while (std::getline(f, str)) {
		std::vector<int> res;
		res = split(str, ',');
		actual_decisions.push_back((bool)(res[9]));
		std::vector<bool> e;
		e.push_back((bool)(res[6]));
		e.push_back((bool)(res[7]));
		e.push_back((bool)(res[8]));
		expert_decisions.push_back(e);
		context_data.push_back(res[1]);
	}

	of.open(outfile);
	WeightedMajorityAlgorithm mWMA(experts);
	ContextWMA mCWMA(experts,contexts);
	mWMA.train(expert_decisions, actual_decisions);//, of);
	mCWMA.train(expert_decisions, actual_decisions, context_data);//, of);
	mWMA.printStat(of);
	mCWMA.printStat(of);
}

void e_Multi()
{
	int experts = 5;
	int contexts = 10;
	int rounds = 500;
	int classes = 100;
	std::string outfile = outDir + "\\experiments\\Multi_test.csv";
	std::ofstream f;
	f.open(outfile);
	f << "Context, experts: ," << contexts << "," << experts << std::endl;
	f << "Rounds, wmaLoss, cwmaLoss, bestActionLoss, bestExpertloss, bestContextLoss" << std::endl;
	//for (int i = rounds; i < 400; i += 5)
	{
		double avgLoss_wma = 0;
		double avgLoss_cwma = 0;
		double avgBestAction = 0, avgBestExpert = 0, avgBestContext = 0;
		for (int j = 0; j < e_repeat; j++)
		{
			Test_SynDataM t(experts, contexts, rounds,classes);
			t.runTests();
			avgLoss_cwma += t.getCWMALoss();
			avgLoss_wma += t.getWMALoss();
			avgBestAction += t.getBestActionLoss();
			avgBestExpert += t.getBestExpertLoss();
			avgBestContext += t.getBestContextLoss();
		}
		f /*<< i */<< "," << (double)(avgLoss_wma / e_repeat) << "," << (double)(avgLoss_cwma / e_repeat) << "," << (double)(avgBestAction / e_repeat) << "," << (double)(avgBestExpert / e_repeat) << "," << (double)(avgBestContext / e_repeat) << std::endl;
	}
	f.close();

}
// Tests against real data at the decision level
void real_decisionlvl()
{
	std::string outfile = outDir + "\\pie_test_decision.csv";
	std::ofstream of;
	int experts = 3;
	int contexts = 0;
	int rounds = 0;
	std::map<int,int> contextmap;
	std::vector<int> context_data;
	std::vector<bool> actual_decisions;
	std::vector<std::vector<bool>> expert_decisions;

	//expert_decisions.resize(experts);
	//context_data.resize(rounds);
	//actual_decisions.resize(rounds);

	/*for (int i = 0; i < experts; i++)
	{
	expert_decisions[i].resize(rounds);
	}*/

	//Read & transfer data
	std::string infile = outDir + "\\..\\input\\Data_2.csv";
	std::ifstream f;
	f.open(infile);
	std::string str;
	int context_count = 0;
	while (std::getline(f, str)) {
		std::vector<int> res;
		res = split(str, ',');
		actual_decisions.push_back((bool)(res[10]));
		std::vector<bool> e;
		e.push_back((bool)(res[7]));
		e.push_back((bool)(res[8]));
		e.push_back((bool)(res[9]));
		expert_decisions.push_back(e);

		if (contextmap.find(res[6]) == contextmap.end())
			contextmap[res[6]] = context_count++;

		context_data.push_back(contextmap[res[6]]);
	}
	contexts = contextmap.size();
	of.open(outfile);
	WeightedMajorityAlgorithm mWMA(experts);
	ContextWMA mCWMA(experts, contexts);
	mWMA.train(expert_decisions, actual_decisions);//, of);
	mCWMA.train(expert_decisions, actual_decisions, context_data);//, of);
	mWMA.printStat(of);
	mCWMA.printStat(of);
}

void real_scorelvl(int threshold)
{
	std::string outfile = outDir + "\\pie_test_score.csv";
	std::ofstream of;
	int experts = 3;
	int contexts = 0;
	std::map<int, int> contextmap;
	std::vector<int> context_data;
	std::vector<bool> actual_decisions;
	std::vector<std::vector<int>> expert_decisions;

	//Read & transfer data
	std::string infile = outDir + "\\..\\input\\Data_2.csv";
	std::ifstream f;
	f.open(infile);
	std::string str;
	int context_count = 0;
	while (std::getline(f, str)) {
		std::vector<int> res;
		res = split(str, ',');
		actual_decisions.push_back(bool(res[10]));
		std::vector<int> e;
		e.push_back(res[3]);
		e.push_back(res[4]);
		e.push_back(res[5]);
		expert_decisions.push_back(e);

		if (contextmap.find(res[6]) == contextmap.end())
			contextmap[res[6]] = context_count++;

		context_data.push_back(contextmap[res[6]]);
	}
	contexts = contextmap.size();
	of.open(outfile);
	WMA_Multi mWMA(experts);
	ContextWMA_Multi mCWMA(experts, contexts);
	mWMA.setThreshold(threshold);
	mCWMA.setThreshold(threshold);
	mWMA.setMode(MODE::SCORE);
	mCWMA.setMode(MODE::SCORE);
	mWMA.train(expert_decisions, actual_decisions);//, of);
	mCWMA.train(expert_decisions, actual_decisions, context_data);//, of);
	mWMA.printStat(of);
	mCWMA.printStat(of);
	C_Methods methods;
	methods.mContext = contexts;
	methods.evaluate(CMODE::sum, expert_decisions, actual_decisions, context_data);
	//methods.evaluate(CMODE::product, expert_decisions, actual_decisions, context_data);
	//methods.evaluate(CMODE::max, expert_decisions, actual_decisions, context_data);
	//methods.evaluate(CMODE::wma, expert_decisions, actual_decisions, context_data);
	//methods.evaluate(CMODE::cwma, expert_decisions, actual_decisions, context_data);
}


int main(int argc, char* argv[])
{
	//test_WMA();
	//test_CWMA();
	//Test_SynData t(10,5,24);
	//t.runTests();
	
	//e_Experts();
	//e_Contexts();
	//e_Rounds();
	//e_Bias_WithNoise();
	//convergenceTest();

	//e_Multi();

	//real_data();
	real_scorelvl(30);
	real_decisionlvl();
	/*Test_SynDataM t(5, 10, 1000,100);
	t.runTests();*/
	//getch();
	return 0;
}