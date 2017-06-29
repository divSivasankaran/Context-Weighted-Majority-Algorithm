#pragma once
#include<vector>

extern std::string outDir;

namespace alpehnull {
	namespace core {
		namespace algo {

			class Node {
				friend class DecisionTree;
			private:
				Node() {
					left = NULL;
					right = NULL;
					height = 0;
				}
				//Node* parent;
				Node* left;
				Node* right;
				int value;
				int height;
				std::vector<std::vector<double>> data;
			};
			class DecisionTree {
			public:
				Node* createNode() {
					return new Node;
				};
				int find_best_split(std::vector<std::vector<double>> &data, std::vector<std::vector<double>>& left, std::vector<std::vector<double>>& right);
				//int classify();
				void printTree(Node* root);
				Node* train(std::vector<std::vector<double>> &data);
			};
		}
	}
}