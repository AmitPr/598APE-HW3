#ifndef QUADTREE_H
#define QUADTREE_H
#include <math.h>

#include "vec.h"

class Node {
 private:
  int childIndex(const Vec2& p) const {
    int index = 0;
    if (p.x >= center.x) index += 1;
    if (p.y >= center.y) index += 2;
    return index;
  }

  Node* createChild(int index) {
    double halfSize = size / 2.0;
    Vec2 offset{(index & 1) ? halfSize : -halfSize,
                (index & 2) ? halfSize : -halfSize};
    Node* child = new Node(center + offset, halfSize);
    children[index] = child;
    return child;
  }

 public:
  double mass, size;
  Vec2 center, com;
  int idx;
  Node* children[4];

  Node(Vec2 center_, double size_)
      : center(center_), size(size_), mass(0), com(Vec2()), idx(-1) {
    for (int i = 0; i < 4; i++) children[i] = nullptr;
  }

  ~Node() {
    for (int i = 0; i < 4; i++) {
      if (children[i]) delete children[i];
    }
  }

  bool isLeaf() const { return idx >= 0; }
  bool isEmpty() const { return mass == 0 && idx < 0; }

  void insert(int idx, Vec2 p, double m) {
    if (isEmpty()) {
      this->idx = idx;
      mass = m;
      com = p;
      return;
    }
    if (isLeaf()) {
      Vec2 oldP = com;
      int oldIdx = this->idx;
      double oldMass = mass;
      int oldChildIdx = childIndex(oldP);
      createChild(oldChildIdx)->insert(oldIdx, oldP, oldMass);

      this->idx = -1;
    }
    com = (com * mass + p * m) / (mass + m);
    mass += m;
    int childIdx = childIndex(p);
    if (children[childIdx] == nullptr)
      createChild(childIdx)->insert(idx, p, m);
    else
      children[childIdx]->insert(idx, p, m);
  }

  Vec2 calculateForce(const Vec2& position, double mass, double theta) const {
    Vec2 direction = com - position;
    double distSqr = direction.mag2() + 0.0001;
    double dist = sqrt(distSqr);
    if (isLeaf() || (size / dist < theta)) {
      if (isLeaf() && idx >= 0 && position == com) return Vec2(0, 0);
      double invDist = this->mass * mass / dist;
      double invDist3 = invDist * invDist * invDist;
      return (direction * 0.01) * invDist3;
    } else {
      Vec2 totalForce(0, 0);
      for (int i = 0; i < 4; i++) {
        if (children[i] && children[i]->mass > 0)
          totalForce =
              totalForce + children[i]->calculateForce(position, mass, theta);
      }
      return totalForce;
    }
  }
};

#endif
