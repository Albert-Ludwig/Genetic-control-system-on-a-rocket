#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stack>
#include <vector>

#include "cartCentering.h"

using namespace std;

// return a double unifomrly sampled in (0,1)
double randDouble(mt19937 &rng)
{
  return std::uniform_real_distribution<>{0, 1}(rng);
}
// return uniformly sampled 0 or 1
bool randChoice(mt19937 &rng)
{
  return std::uniform_int_distribution<>{0, 1}(rng);
}
// return a random integer uniformly sampled in (min, max)
int randInt(mt19937 &rng, const int &min, const int &max)
{
  return std::uniform_int_distribution<>{min, max}(rng);
}

// return true if op is a suported operation, otherwise return false
bool isOp(string op)
{
  if (op == "+")
    return true;
  else if (op == "-")
    return true;
  else if (op == "*")
    return true;
  else if (op == "/")
    return true;
  else if (op == ">")
    return true;
  else if (op == "abs")
    return true;
  else if (op == "read")
    return true;
  else if (op == "write")
    return true;
  else
    return false;
}

int arity(string op)
{
  if (op == "abs")
    return 1;
  else if (op == "read")
  {
    return 0;
  }
  else if (op == "write")
  {
    return 1;
  }
  else
    return 2;
}

typedef string Elem;

class LinkedBinaryTree
{
public:
  struct Node
  {
    Elem elt;
    string name;
    Node *par;
    Node *left;
    Node *right;
    Node() : elt(), par(NULL), name(""), left(NULL), right(NULL) {}
    int depth()
    {
      if (par == NULL)
        return 0;
      return par->depth() + 1;
    }
  };

  class Position
  {
  private:
    Node *v;

  public:
    Position(Node *_v = NULL) : v(_v) {}
    Elem &operator*() { return v->elt; }
    Position left() const { return Position(v->left); }
    void setLeft(Node *n) { v->left = n; }
    Position right() const { return Position(v->right); }
    void setRight(Node *n) { v->right = n; }
    Position parent() const // get parent
    {
      return Position(v->par);
    }
    bool isRoot() const // root of the tree?
    {
      return v->par == NULL;
    }
    bool isExternal() const // an external node?
    {
      return v->left == NULL && v->right == NULL;
    }
    friend class LinkedBinaryTree; // give tree access
  };
  typedef vector<Position> PositionList;

public:
  LinkedBinaryTree() : _root(NULL), score(0), steps(0), generation(0)
  {
    memory = vector<double>(4, 0.0);
  }

  // copy constructor
  LinkedBinaryTree(const LinkedBinaryTree &t)
  {
    _root = copyPreOrder(t.root());
    score = t.getScore();
    steps = t.getSteps();
    generation = t.getGeneration();
    memory = t.memory;
  }

  // copy assignment operator
  LinkedBinaryTree &operator=(const LinkedBinaryTree &t)
  {
    if (this != &t)
    {
      // if tree already contains data, delete it
      if (_root != NULL)
      {
        PositionList pl = positions();
        for (auto &p : pl)
          delete p.v;
      }
      _root = copyPreOrder(t.root());
      score = t.getScore();
      steps = t.getSteps();
      generation = t.getGeneration();
      memory = t.memory;
    }
    return *this;
  }

  // destructor
  ~LinkedBinaryTree()
  {
    if (_root != NULL)
    {
      PositionList pl = positions();
      for (auto &p : pl)
        delete p.v;
    }
  }

  int size() const { return size(_root); }
  int size(Node *root) const;
  int depth() const;
  bool empty() const { return size() == 0; };
  Node *root() const { return _root; }
  PositionList positions() const;
  void addRoot() { _root = new Node; }
  void addRoot(Elem e)
  {
    _root = new Node;
    _root->elt = e;
  }
  void nameRoot(string name) { _root->name = name; }
  void addLeftChild(const Position &p, const Node *n);
  void addLeftChild(const Position &p);
  void addRightChild(const Position &p, const Node *n);
  void addRightChild(const Position &p);
  void printExpression() { printExpression(_root); }
  void printExpression(Node *v);
  double evaluateExpression(double a, double b)
  {
    return evaluateExpression(Position(_root), a, b);
  };
  double evaluateExpression(const Position &p, double a, double b);
  long getGeneration() const { return generation; }
  void setGeneration(int g) { generation = g; }
  double getScore() const { return score; }
  void setScore(double s) { score = s; }
  double getSteps() const { return steps; }
  void setSteps(double s) { steps = s; }
  void randomExpressionTree(Node *p, const int &maxDepth, mt19937 &rng, bool PARTIALLY_OBSERVABLE);
  void randomExpressionTree(const int &maxDepth, mt19937 &rng, bool PARTIALLY_OBSERVABLE)
  {
    randomExpressionTree(_root, maxDepth, rng, PARTIALLY_OBSERVABLE);
  }
  void deleteSubtreeMutator(mt19937 &rng);
  void addSubtreeMutator(mt19937 &rng, const int maxDepth, bool PARTIALLY_OBSERVABLE);
  void clear(Node *v)
  {
    if (v == nullptr)
      return;
    clear(v->left);
    clear(v->right);
    delete v;
  }
  double getMemory() const
  {
    double sum = 0;
    for (double v : memory)
      sum += v;
    return sum / memory.size();
  }
  void setMemory(double m)
  {
    for (int i = 0; i < memory.size(); i++)
    {
      memory[i] = m;
    }
  }
  void updateMemory(double x)
  {
    for (int i = 0; i < memory.size() - 1; i++)
    {
      memory[i] = memory[i + 1];
    }
    memory[memory.size() - 1] = x;
  }

protected:                                        // local utilities
  void preorder(Node *v, PositionList &pl) const; // preorder utility
  Node *copyPreOrder(const Node *root);
  double score;    // mean reward over 20 episodes
  double steps;    // mean steps-per-episode over 20 episodes
  long generation; // which generation was tree "born"
private:
  Node *_root; // pointer to the root
  vector<double> memory;
};

// add the tree rooted at node child as this tree's left child
void LinkedBinaryTree::addLeftChild(const Position &p, const Node *child)
{
  Node *v = p.v;
  v->left = copyPreOrder(child); // deep copy child
  v->left->par = v;
}

// add the tree rooted at node child as this tree's right child
void LinkedBinaryTree::addRightChild(const Position &p, const Node *child)
{
  Node *v = p.v;
  v->right = copyPreOrder(child); // deep copy child
  v->right->par = v;
}

void LinkedBinaryTree::addLeftChild(const Position &p)
{
  Node *v = p.v;
  v->left = new Node;
  v->left->par = v;
}

void LinkedBinaryTree::addRightChild(const Position &p)
{
  Node *v = p.v;
  v->right = new Node;
  v->right->par = v;
}

// return a list of all nodes
LinkedBinaryTree::PositionList LinkedBinaryTree::positions() const
{
  PositionList pl;
  preorder(_root, pl);
  return PositionList(pl);
}

void LinkedBinaryTree::preorder(Node *v, PositionList &pl) const
{
  pl.push_back(Position(v));
  if (v->left != NULL)
    preorder(v->left, pl);
  if (v->right != NULL)
    preorder(v->right, pl);
}

int LinkedBinaryTree::size(Node *v) const
{
  int lsize = 0;
  int rsize = 0;
  if (v->left != NULL)
    lsize = size(v->left);
  if (v->right != NULL)
    rsize = size(v->right);
  return 1 + lsize + rsize;
}

int LinkedBinaryTree::depth() const
{
  PositionList pl = positions();
  int depth = 0;
  for (auto &p : pl)
    depth = std::max(depth, p.v->depth());
  return depth;
}

LinkedBinaryTree::Node *LinkedBinaryTree::copyPreOrder(const Node *root)
{
  if (root == NULL)
    return NULL;
  Node *nn = new Node;
  nn->elt = root->elt;
  nn->left = copyPreOrder(root->left);
  if (nn->left != NULL)
    nn->left->par = nn;
  nn->right = copyPreOrder(root->right);
  if (nn->right != NULL)
    nn->right->par = nn;
  return nn;
}

void LinkedBinaryTree::printExpression(Node *v)
{
  if (v == nullptr)
  {
    return;
  }
  if (v->left == nullptr && v->right == nullptr)
  {
    cout << v->elt;
    return;
  }
  int opArity = arity(v->elt);
  if (opArity == 0)
  {
    cout << v->elt;
    return;
  }
  else if (opArity == 1)
  {
    cout << v->elt << "(";
    if (v->left)
    {
      printExpression(v->left);
    }
    cout << ")";
  }
  else if (opArity == 2)
  {
    cout << "(";
    printExpression(v->left);
    cout << " " << v->elt << " ";
    printExpression(v->right);
    cout << ")";
  }
}

double evalOp(string op, LinkedBinaryTree &theTree, double x, double y = 0)
{
  double result;
  if (op == "+")
    result = x + y;
  else if (op == "-")
    result = x - y;
  else if (op == "*")
    result = x * y;
  else if (op == "/")
  {
    result = x / y;
  }
  else if (op == ">")
  {
    result = x > y ? 1 : -1;
  }
  else if (op == "abs")
  {
    result = abs(x);
  }
  else if (op == "read")
  {
    result = theTree.getMemory();
  }
  else if (op == "write")
  {
    theTree.setMemory(x);
    result = x;
  }
  else
    result = 0;
  return isnan(result) || !isfinite(result) ? 0 : result;
}

double LinkedBinaryTree::evaluateExpression(const Position &p, double a,
                                            double b)
{
  if (!p.isExternal())
  {
    auto x = evaluateExpression(p.left(), a, b);
    if (arity(p.v->elt) > 1)
    {
      auto y = evaluateExpression(p.right(), a, b);
      return evalOp(p.v->elt, *this, x, y);
    }
    else
    {
      return evalOp(p.v->elt, *this, x);
    }
  }
  else
  {
    if (p.v->elt == "a")
      return a;
    else if (p.v->elt == "b")
      return b;
    else if (p.v->elt == "read")
    {
      return evalOp("read", *this, 0.0, 0.0);
    }
    else if (p.v->elt == "write")
    {
      return evalOp("write", *this, 0.0, 0.0);
    }
    else
      return stod(p.v->elt);
  }
}

void LinkedBinaryTree::deleteSubtreeMutator(mt19937 &rng)
{
  if (_root == nullptr)
  {
    cout << "The tree is empty" << endl;
    return;
  }
  vector<Node *> candidates;
  stack<Node *> s;
  s.push(_root);
  while (!s.empty()) // traverse the tree in post-order
  {
    Node *cur = s.top();
    s.pop();
    if (cur->par != nullptr)
      candidates.push_back(cur);
    if (cur->left != nullptr)
      s.push(cur->left);
    if (cur->right != nullptr)
      s.push(cur->right);
  }
  if (candidates.empty())
  {
    return;
  }
  int index = randInt(rng, 0, candidates.size() - 1); // randomly select a node to delete
  Node *target = candidates[index];
  Node *parent = target->par;
  if (parent->left == target)
  {
    parent->left = nullptr;
  }
  else if (parent->right == target)
  {
    parent->right = nullptr;
  }
  clear(target);
  Node *newNode = new Node;
  newNode->elt = randChoice(rng) ? "a" : "b"; // increase the robustness of the tree
  newNode->par = parent;
  if (parent->left == nullptr)
  {
    parent->left = newNode;
  }
  else if (parent->right == nullptr)
  {
    parent->right = newNode;
  }
}

void LinkedBinaryTree::addSubtreeMutator(mt19937 &rng, const int maxDepth, bool PARTIALLY_OBSERVABLE)
{
  if (_root == nullptr)
  {
    addRoot();
    randomExpressionTree(_root, maxDepth, rng, PARTIALLY_OBSERVABLE);
    return;
  }
  vector<Node *> leaves;
  stack<Node *> nodeStack;
  nodeStack.push(_root); // initialize the stack with the root node

  while (!nodeStack.empty())
  {
    Node *current = nodeStack.top();
    nodeStack.pop();

    // check if the current node is a leaf node
    if (current->left == nullptr && current->right == nullptr)
    {
      leaves.push_back(current);
    }
    else
    {
      // push the children of the current node to the stack
      if (current->right != nullptr)
      {
        nodeStack.push(current->right);
      }
      if (current->left != nullptr)
      {
        nodeStack.push(current->left);
      }
    }
  }

  if (leaves.empty())
    return;

  // random select a leaf node
  int idx = randInt(rng, 0, leaves.size() - 1);
  Node *target = leaves[idx];
  int d = target->depth();
  int depthAllowance = maxDepth - d;
  if (depthAllowance < 0)
  {
    depthAllowance = 0;
  }

  // generate a random subtree
  randomExpressionTree(target, depthAllowance, rng, PARTIALLY_OBSERVABLE);
}

bool operator<(const LinkedBinaryTree &x, const LinkedBinaryTree &y)
{
  return x.getScore() < y.getScore();
}

LinkedBinaryTree createExpressionTree(string postfix)
{
  stack<LinkedBinaryTree> tree_stack;
  stringstream ss(postfix);
  // Split each line into words
  string token;
  while (getline(ss, token, ' '))
  {
    if (token.empty())
      continue;
    LinkedBinaryTree t;
    if (!isOp(token))
    {
      t.addRoot(token);
      tree_stack.push(t);
    }
    else
    {
      t.addRoot(token);
      if (arity(token) > 1)
      {
        LinkedBinaryTree r = tree_stack.top();
        tree_stack.pop();
        t.addRightChild(t.root(), r.root());
      }
      LinkedBinaryTree l = tree_stack.top();
      tree_stack.pop();
      t.addLeftChild(t.root(), l.root());
      tree_stack.push(t);
    }
  }
  return tree_stack.top();
}

void LinkedBinaryTree::randomExpressionTree(Node *p, const int &maxDepth, mt19937 &rng, bool PARTIALLY_OBSERVABLE)
{
  double prob = randDouble(rng); // random number between 0 and 1
  static const vector<string> terminals = {"a", "b"};
  if (PARTIALLY_OBSERVABLE)
  {
    static const vector<string> ops = {"+", "-", "*", "/", ">", "abs", "read", "write"};

    // if get in the leaf from the recursion (base case), then create a leaf node
    if (maxDepth == 0 || prob < 0.3) // when reaches the max depth or with 30% probability, generate the terminal node (leaf)
    {
      int index = randInt(rng, 0, terminals.size() - 1); // randomly select a terminal, a or b
      p->elt = terminals[index];
      p->left = nullptr;
      p->right = nullptr;
      return;
    }
    // if not in the leaf, then create an internal node, which is the operator, and then do the recursion
    int op_index = randInt(rng, 0, ops.size() - 1);
    string op = ops[op_index];
    p->elt = op;
    int opArity = arity(op);
    if (opArity == 1)
    {
      p->left = new Node;
      p->left->par = p;
      randomExpressionTree(p->left, maxDepth - 1, rng, PARTIALLY_OBSERVABLE);
      p->right = nullptr;
    }
    else if (opArity == 2)
    {
      p->left = new Node;
      p->right = new Node;
      p->left->par = p;
      p->right->par = p;
      randomExpressionTree(p->left, maxDepth - 1, rng, PARTIALLY_OBSERVABLE);
      randomExpressionTree(p->right, maxDepth - 1, rng, PARTIALLY_OBSERVABLE);
    }
    else if (opArity == 0)
    {
      p->left = nullptr;
      p->right = nullptr;
    }
  }
  else
  {
    static const vector<string> ops = {"+", "-", "*", "/", ">", "abs"};

    // if get in the leaf from the recursion (base case), then create a leaf node
    if (maxDepth == 0 || prob < 0.3) // when reaches the max depth or with 30% probability, generate the terminal node (leaf)
    {
      int index = randInt(rng, 0, terminals.size() - 1); // randomly select a terminal, a or b
      p->elt = terminals[index];
      p->left = nullptr;
      p->right = nullptr;
      return;
    }
    // if not in the leaf, then create an internal node, which is the operator, and then do the recursion
    int op_index = randInt(rng, 0, ops.size() - 1);
    string op = ops[op_index];
    p->elt = op;
    int opArity = arity(op);
    if (opArity == 1)
    {
      p->left = new Node;
      p->left->par = p;
      randomExpressionTree(p->left, maxDepth - 1, rng, PARTIALLY_OBSERVABLE);
      p->right = nullptr;
    }
    else if (opArity == 2)
    {
      p->left = new Node;
      p->right = new Node;
      p->left->par = p;
      p->right->par = p;
      randomExpressionTree(p->left, maxDepth - 1, rng, PARTIALLY_OBSERVABLE);
      randomExpressionTree(p->right, maxDepth - 1, rng, PARTIALLY_OBSERVABLE);
    }
    else if (opArity == 0)
    {
      p->left = nullptr;
      p->right = nullptr;
    }
  }
}
LinkedBinaryTree createRandExpressionTree(int max_depth, mt19937 &rng, bool PARTIALLY_OBSERVABLE)
{
  // modify this function to create and return a random expression tree
  LinkedBinaryTree t;
  t.addRoot();
  t.randomExpressionTree(t.root(), max_depth, rng, PARTIALLY_OBSERVABLE);
  return t;
}

// evaluate tree t in the cart centering task
void evaluate(mt19937 &rng, LinkedBinaryTree &t, const int &num_episode,
              bool animate, bool partially_observable = false)
{
  cartCentering env;
  double mean_score = 0.0;
  double mean_steps = 0.0;
  for (int i = 0; i < num_episode; i++)
  {
    double episode_score = 0.0;
    int episode_steps = 0;
    env.reset(rng);
    if (partially_observable)
    { // for part 4
      t.setMemory(0.0);
    }
    while (!env.terminal())
    {
      int action;
      if (partially_observable)
      {
        action = t.evaluateExpression(env.getCartXPos(), 0.0);
      }
      else
      {
        action = t.evaluateExpression(env.getCartXPos(), env.getCartXVel());
      }
      episode_score += env.update(action, animate);
      episode_steps++;
    }
    mean_score += episode_score;
    mean_steps += episode_steps;
  }
  t.setScore(mean_score / num_episode);
  t.setSteps(mean_steps / num_episode);
}

class LexLessThan // use the class to achieve the operator
{
public:
  bool operator()(const LinkedBinaryTree &A, const LinkedBinaryTree &B) const
  {
    if (fabs(A.getScore() - B.getScore()) < 0.01)
    {
      return A.size() > B.size();
    }
    else
    {
      return A.getScore() < B.getScore();
    }
  }
};

void crossover(LinkedBinaryTree &treeA, LinkedBinaryTree &treeB, mt19937 &rng, int maxAllowedDepth)
{
  vector<LinkedBinaryTree::Node *> candidatesA;
  {
    stack<LinkedBinaryTree::Node *> sA;
    sA.push(treeA.root());
    while (!sA.empty())
    {
      LinkedBinaryTree::Node *cur = sA.top();
      sA.pop();
      if (cur->par != nullptr)
        candidatesA.push_back(cur);
      if (cur->left != nullptr)
        sA.push(cur->left);
      if (cur->right != nullptr)
        sA.push(cur->right);
    }
  }

  vector<LinkedBinaryTree::Node *> candidatesB;
  {
    stack<LinkedBinaryTree::Node *> sB;
    sB.push(treeB.root());
    while (!sB.empty())
    {
      LinkedBinaryTree::Node *cur = sB.top();
      sB.pop();
      if (cur->par != nullptr)
        candidatesB.push_back(cur);
      if (cur->left != nullptr)
        sB.push(cur->left);
      if (cur->right != nullptr)
        sB.push(cur->right);
    }
  }

  if (candidatesA.empty() || candidatesB.empty())
    return;

  int indexA = randInt(rng, 0, candidatesA.size() - 1);
  int indexB = randInt(rng, 0, candidatesB.size() - 1);
  LinkedBinaryTree::Node *nodeA = candidatesA[indexA];
  LinkedBinaryTree::Node *nodeB = candidatesB[indexB];
  LinkedBinaryTree::Node *parentA = nodeA->par;
  LinkedBinaryTree::Node *parentB = nodeB->par;

  // record the old pointers
  LinkedBinaryTree::Node *oldA = nodeA;
  LinkedBinaryTree::Node *oldB = nodeB;

  // exchange the subtrees
  if (parentA->left == nodeA)
    parentA->left = nodeB;
  if (parentA->right == nodeA)
    parentA->right = nodeB;
  if (parentB->left == nodeB)
    parentB->left = nodeA;
  if (parentB->right == nodeB)
    parentB->right = nodeA;
  nodeB->par = parentA;
  nodeA->par = parentB;

  // check for the max depth
  int depthA = treeA.depth();
  int depthB = treeB.depth();
  if (depthA > maxAllowedDepth || depthB > maxAllowedDepth)
  {
    // if the depth is too large, then revert the changes
    if (parentA->left == nodeB)
      parentA->left = nodeA;
    if (parentA->right == nodeB)
      parentA->right = nodeA;
    if (parentB->left == nodeA)
      parentB->left = nodeB;
    if (parentB->right == nodeA)
      parentB->right = nodeB;
    nodeA->par = parentA;
    nodeB->par = parentB;
  }
}

int main()
{
  mt19937 rng(42);
  // Experiment parameters
  const int NUM_TREE = 50;
  const int MAX_DEPTH_INITIAL = 1;
  const int MAX_DEPTH = 10;
  const int NUM_EPISODE = 20;
  const int MAX_GENERATIONS = 100;
  // note: if want to test the part 3, please ensure to close the partially_observableand open the use_crossover
  const bool PARTIALLY_OBSERVABLE = true;

  // set ture to use the part 3 result, set false to use part 1 and 2
  // note: if you want to test part 3 or 4, please open the use_crossover
  const bool USE_CROSSOVER = true;

  // Create an initial "population" of expression trees
  vector<LinkedBinaryTree> trees;
  for (int i = 0; i < NUM_TREE; i++)
  {
    LinkedBinaryTree t = createRandExpressionTree(MAX_DEPTH_INITIAL, rng, PARTIALLY_OBSERVABLE);
    trees.push_back(t);
  }

  // Genetic Algorithm loop
  LinkedBinaryTree best_tree;
  std::cout << "generation,fitness,steps,size,depth" << std::endl;
  for (int g = 1; g <= MAX_GENERATIONS; g++)
  {

    // Fitness evaluation
    for (auto &t : trees)
    {
      if (t.getGeneration() < g - 1)
        continue; // skip if not new
      evaluate(rng, t, NUM_EPISODE, false, PARTIALLY_OBSERVABLE);
    }

    // sort trees using overloaded "<" op (worst->best)
    std::sort(trees.begin(), trees.end());

    // // sort trees using comparaor class (worst->best)
    // std::sort(trees.begin(), trees.end(), LexLessThan());

    // erase worst 50% of trees (first half of vector)
    trees.erase(trees.begin(), trees.begin() + NUM_TREE / 2);

    // Print stats for best tree
    best_tree = trees[trees.size() - 1];
    std::cout << g << ",";
    std::cout << best_tree.getScore() << ",";
    std::cout << best_tree.getSteps() << ",";
    std::cout << best_tree.size() << ",";
    std::cout << best_tree.depth() << std::endl;

    if (USE_CROSSOVER)
    {
      if (randDouble(rng) < 0.5) // set 0.5 for the possibility of crossover
      {
        int idx1 = randInt(rng, 0, trees.size() - 1);
        int idx2 = randInt(rng, 0, trees.size() - 1);
        if (idx1 != idx2)
          crossover(trees[idx1], trees[idx2], rng, MAX_DEPTH);
      }
      // Selection and mutation
      while (trees.size() < NUM_TREE)
      {
        // Selected random "parent" tree from survivors
        LinkedBinaryTree parent = trees[randInt(rng, 0, (NUM_TREE / 2) - 1)];

        // Create child tree with copy constructor
        LinkedBinaryTree child(parent);
        child.setGeneration(g);

        // Mutation
        // Delete a randomly selected part of the child's tree
        child.deleteSubtreeMutator(rng);
        // Add a random subtree to the child
        child.addSubtreeMutator(rng, MAX_DEPTH, PARTIALLY_OBSERVABLE);

        trees.push_back(child);
      }
    }
    else
    {
      // Selection and mutation
      while (trees.size() < NUM_TREE)
      {
        // Selected random "parent" tree from survivors
        LinkedBinaryTree parent = trees[randInt(rng, 0, (NUM_TREE / 2) - 1)];

        // Create child tree with copy constructor
        LinkedBinaryTree child(parent);
        child.setGeneration(g);

        // Mutation
        // Delete a randomly selected part of the child's tree
        child.deleteSubtreeMutator(rng);
        // Add a random subtree to the child
        child.addSubtreeMutator(rng, MAX_DEPTH, PARTIALLY_OBSERVABLE);

        trees.push_back(child);
      }
    }
  }

  // // Evaluate best tree with animation
  // const int num_episode = 3;
  // evaluate(rng, best_tree, num_episode, true, PARTIALLY_OBSERVABLE);

  // Print best tree info
  std::cout << std::endl
            << "Best tree:" << std::endl;
  best_tree.printExpression();
  std::cout << endl;
  std::cout << "Generation: " << best_tree.getGeneration() << endl;
  std::cout << "Size: " << best_tree.size() << std::endl;
  std::cout << "Depth: " << best_tree.depth() << std::endl;
  std::cout << "Fitness: " << best_tree.getScore() << std::endl
            << std::endl;
}