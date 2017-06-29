#include<DecisionTree.h>
#include<../src/helper.cpp>

using namespace alpehnull::core::algo;


double stdDeviation(std::vector<double> v)
{
	double E = 0, ave = 0;
	double inverse = 1.0 / static_cast<double>(v.size());
	for (unsigned int i = 0; i < v.size(); i++)
	{
		ave += v[i];
	}
	ave *= inverse;
	for (unsigned int i = 0; i < v.size(); i++)
	{
		E += pow(static_cast<double>(v[i]) - ave, 2);
	}
	return sqrt(inverse * E);
}


int DecisionTree::find_best_split(std::vector<std::vector<double>> &data, std::vector<std::vector<double>>& left, std::vector<std::vector<double>>& right)
{
	int best_pos = -1;
	double max_IG = 0;
	double std_complete = 0;
	bool calculated = false;
	for (int k = 1, pos = 0; k < data.size(); k = k * 2, pos++)
	{
		double std_left = 0;
		double std_right = 0;
		std::vector<std::vector<double>> l, r, complete;
		l.resize(data[0].size());
		r.resize(data[0].size());
		complete.resize(data[0].size());
		
		for (int i = 0; i < data.size(); i++)
		{
			for (int j = 0; j < data[i].size(); j++)
			{
				complete[j].push_back(data[i][j]);
				if (i & (1 << pos))
				{
					l[j].push_back(data[i][j]);
				}
				else
				{
					r[j].push_back(data[i][j]);
				}
			}
		}
		for (int i = 0; i < data.size(); i++)
		{
			for (int j = 0; j < data[i].size(); j++)
			{
				if (!calculated)
				{
					std_complete += stdDeviation(complete[j]);
				}
				std_left += stdDeviation(l[j]);
				std_right += stdDeviation(r[j]);
			}
		}
		calculated = true;
		auto info_gain = std_complete - (std_left/2) - (std_right/2);
		if (info_gain > max_IG)
		{
			left.clear();
			right.clear();
			best_pos = pos;
			max_IG = info_gain;
			left.resize(l[0].size());
			right.resize(r[0].size());
			for (int i = 0; i < l.size(); i++)
			{
				for (int j = 0; j < l[i].size(); j++)
				{
					left[j].push_back(l[i][j]);
					right[j].push_back(r[i][j]);
				}
			}
		}
	}
	std::cout << std_complete << " CHOSEN ATTRIBUTE " << best_pos << " INFORMATION GAIN " << max_IG  <<std::endl;

	if (max_IG <1)
		best_pos = -1;
	return best_pos;
}


Node* DecisionTree::train(std::vector<std::vector<double>>& data)
{
	if (data.size() < 2)
		return NULL;
	std::vector<std::vector<double>> left_split,right_split;
	int attr = find_best_split(data,left_split,right_split);
	auto root = createNode();
	root->data = data;
	root->value = attr;
	if (attr == -1)
	{
		root->left = NULL;
		root->right = NULL;
	}
	else {
		//std::cout << "DATA SIZE " << data.size() << "LEFT" << std::endl;
		root->left = train(left_split);
		//std::cout << "DATA SIZE " << data.size() << "RIGHT" << std::endl;
		root->right = train(right_split);
	}
	return root;
}

void DecisionTree::printTree(Node* root)
{
	if (root != NULL)
	{
		std::cout << "current height " << root->height << std::endl;
		for (int i = 0; i < root->data.size(); i++)
		{
			PrintVector<double>(root->data[i]);
		}
		if (root->left != NULL)
		{
			std::cout << "Attribute: " << root->data.size() << " Left " << std::endl;
			root->left->height = root->height + 1;
			printTree(root->left);
		}
		if (root->right != NULL)
		{
			std::cout << "Attribute: " << root->data.size() << " Right " << std::endl;
			root->right->height = root->height + 1;
			printTree(root->right);
		}
	}
}