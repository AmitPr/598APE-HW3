#ifndef QUADTREE_H
#define QUADTREE_H
#include <math.h>

#include "vec.h"

class Node {
 private:
  // Which quadrant
  int childIndex(const Vec2& p) const {
    int index = 0;
    if (p.x >= center.x) {
      index += 1;  // right
    }
    if (p.y >= center.y) {
      index += 2;  // bottom
    }
    return index;
  }

  Node* createChild(int index) {
    double halfSize = size / 2.0;
    Vec2 offset = Vec2{(index & 1) ? halfSize : -halfSize,
                       (index & 2) ? halfSize : -halfSize};
    Node* child = new Node(center + offset, halfSize);
    children[index] = child;
    return child;
  }

 public:
  double mass;
  double size;
  Vec2 center;
  Vec2 com;

  int idx;  // -1 = not leaf
  Node* children[4];

  Node(Vec2 center_, double size_)
      : center(center_), size(size_), mass(0), com(Vec2()), idx(-1) {
    for (int i = 0; i < 4; i++) {
      children[i] = nullptr;
    }
  }

  ~Node() {
    for (int i = 0; i < 4; i++) {
      if (children[i]) {
        delete children[i];
      }
    }
  }

  bool isLeaf() const { return idx < 0; }
  bool isEmpty() const { return mass == 0 && idx < 0; }

  void insert(int idx, Vec2 p, double m) {
    if (isEmpty()) {
      this->idx = idx;
      this->mass = m;
      this->com = p;
    } else {
      // Update mass and center of mass
      com = (com * mass + p * m) / (mass + m);
      mass += m;

      double halfSize = size / 2.0;
      int childIdx = childIndex(p);
      if (isLeaf()) {
        // We have to move the current node's data to a child
        int curChildIdx = childIndex(com);
        Node* child = createChild(curChildIdx);
        child->idx = this->idx;
        child->mass = mass;
        child->com = com;

        this->idx = -1;
      }

      if (children[childIdx] == nullptr) {
        Node* child = createChild(childIdx);
        child->idx = idx;
        child->mass = m;
        child->com = p;
      } else {
        children[childIdx]->insert(idx, p, m);
      }
    }
  }

  // Calculate force on a body using Barnes-Hut approximation
  Vec2 calculateForce(const Vec2& position, double mass, double theta) const {
    Vec2 direction = com - position;
    double distSqr = direction.mag2() + 0.0001;
    double dist = sqrt(distSqr);

    // If this is a leaf node or the node is sufficiently far away
    // (distance/size > 1/theta), treat this node as a single body
    if (isLeaf() || (size / dist < theta)) {
      // Skip self-interaction
      if (isLeaf() && idx >= 0 && position == com) {
        return Vec2(0, 0);
      }

      const double invDist = this->mass * mass / dist;
      const double invDist3 = invDist * invDist * invDist;
      return (direction * 0.01) * invDist3;
    } else {
      // Otherwise, recursively calculate forces from each child
      Vec2 totalForce(0, 0);
      for (int i = 0; i < 4; i++) {
        if (children[i] && children[i]->mass > 0) {
          totalForce =
              totalForce + children[i]->calculateForce(position, mass, theta);
        }
      }
      return totalForce;
    }
  }
};

#endif