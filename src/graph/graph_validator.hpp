/**
 * Graph IR validator — validates an AneGraph before compilation.
 *
 * Runs entirely at compile time; touches no ANE hardware.
 *
 * Checks performed:
 *  1. Graph has at least one input and one output.
 *  2. All tensor shapes pass ANE constraints (batch==1, height==1,
 *     S%8==0, S≤65536, C in [1,16384]).
 *  3. No dangling edges: every node input is produced by a prior node
 *     or is a declared graph input.
 *  4. No cycles: topological sort (Kahn's algorithm) succeeds.
 *  5. Every marked output is reachable from a graph input.
 *  6. Weight sizes match op expectations:
 *       matmul:    IC × OC × 2 bytes (IC from input tensor, OC from output tensor)
 *       rmsnorm:   C × 2 bytes (scale)
 *       layernorm: C × 2 × 2 bytes (gamma packed before beta)
 *       weight-free ops: weights must be empty
 *       conv2d / cast: unsupported in graph API
 *  7. Binary ops (ADD, MUL) and any other multi-input ops: all input
 *     tensors must have identical shapes (ANE constraint #18).
 *
 * All errors are collected before returning — the result contains every
 * violation found, not just the first one.
 */
#pragma once

#include "ane_graph.hpp"
#include <string>
#include <vector>

namespace libane {
namespace graph {

struct ValidationResult {
    /** True iff no errors were found. */
    bool ok() const { return errors.empty(); }

    /** Human-readable error messages, one per violation. Empty on success. */
    std::vector<std::string> errors;
};

class GraphValidator {
public:
    static ValidationResult validate(const AneGraph& graph);

private:
    static void check_structure   (const AneGraph& g, ValidationResult& r);
    static void check_shapes      (const AneGraph& g, ValidationResult& r);
    static void check_edges       (const AneGraph& g, ValidationResult& r);
    static void check_topo        (const AneGraph& g, ValidationResult& r);
    static void check_reachability(const AneGraph& g, ValidationResult& r);
    static void check_weights     (const AneGraph& g, ValidationResult& r);
    static void check_binary_shapes(const AneGraph& g, ValidationResult& r);
};

} // namespace graph
} // namespace libane
