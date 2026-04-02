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
 *       reshape:   no weights, exactly one input, input/output numel equal
 *       concat:    no weights, exactly two inputs, output C = C0 + C1, same S
 *       slice_by_index: no weights, exactly one input, output dims <= input dims
 *       reduce_sum: no weights, exactly one input, axis=1 keep_dims=true -> output C=1, same S
 *       reduce_mean: same constraints as reduce_sum
 *       reduce_max: same constraints as reduce_sum
 *       sqrt:      no weights, exactly one input
 *       log:       no weights, exactly one input
 *       rsqrt:     no weights, exactly one input
 *       weight-free ops: weights must be empty
 *       conv2d / cast: unsupported in graph API
 *  7. Binary arithmetic ops (ADD, MUL, SUB, REAL_DIV): all input
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
