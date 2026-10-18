#ifndef __SD_CORE_LAYER_SPLIT_PARTITION_H__
#define __SD_CORE_LAYER_SPLIT_PARTITION_H__

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

#include "core/ggml_graph_cut.h"

namespace sd {

    struct GraphCutLayerSplitAssignment {
        std::vector<std::vector<ggml_tensor*>> tensors_by_backend;
        std::vector<int64_t> bytes_by_backend;
        std::vector<size_t> first_segment_by_backend;
        std::vector<size_t> last_segment_by_backend;
        std::unordered_map<const ggml_tensor*, ggml_backend_t> node_assignments;
        size_t segment_count          = 0;
        bool has_new_param_assignment = false;
    };

    inline bool balanced_layer_split_backend_indices(const std::vector<int64_t>& segment_bytes,
                                                     const std::vector<int64_t>& backend_capacities,
                                                     std::vector<size_t>* backend_by_segment,
                                                     std::vector<int64_t>* bytes_by_backend = nullptr) {
        if (backend_by_segment == nullptr || backend_capacities.empty()) {
            return false;
        }

        int64_t total_bytes    = 0;
        int64_t total_capacity = 0;
        for (int64_t bytes : segment_bytes) {
            if (bytes < 0) {
                return false;
            }
            total_bytes += bytes;
        }
        for (int64_t capacity : backend_capacities) {
            if (capacity < 0) {
                return false;
            }
            total_capacity += capacity;
        }
        if (total_bytes <= 0 || total_bytes > total_capacity) {
            return false;
        }

        backend_by_segment->assign(segment_bytes.size(), 0);
        std::vector<int64_t> assigned(backend_capacities.size(), 0);
        size_t segment_index       = 0;
        int64_t remaining_bytes    = total_bytes;
        int64_t remaining_capacity = total_capacity;

        // 均衡目标可能错过可行的连续边界；失败时保留原容量优先分配能力。
        auto capacity_plan = [&]() {
            backend_by_segment->assign(segment_bytes.size(), 0);
            std::fill(assigned.begin(), assigned.end(), 0);
            size_t backend_index = 0;
            for (size_t index = 0; index < segment_bytes.size(); ++index) {
                const int64_t bytes = segment_bytes[index];
                while (backend_index + 1 < backend_capacities.size() && bytes > 0 &&
                       assigned[backend_index] + bytes > backend_capacities[backend_index]) {
                    ++backend_index;
                }
                if (assigned[backend_index] + bytes > backend_capacities[backend_index]) {
                    return false;
                }
                (*backend_by_segment)[index] = backend_index;
                assigned[backend_index] += bytes;
            }
            if (bytes_by_backend != nullptr) {
                *bytes_by_backend = std::move(assigned);
            }
            return true;
        };

        for (size_t backend_index = 0;
             backend_index < backend_capacities.size() && segment_index < segment_bytes.size();
             backend_index++) {
            const int64_t capacity  = backend_capacities[backend_index];
            const bool last_backend = backend_index + 1 == backend_capacities.size();
            int64_t target          = remaining_bytes;
            if (!last_backend) {
                target = static_cast<int64_t>((static_cast<long double>(remaining_bytes) * capacity /
                                               remaining_capacity) +
                                              0.5L);
                target = std::min(target, capacity);
            }

            int64_t used = 0;
            while (segment_index < segment_bytes.size()) {
                const int64_t bytes = segment_bytes[segment_index];
                if (bytes == 0) {
                    (*backend_by_segment)[segment_index++] = backend_index;
                    continue;
                }

                if (last_backend) {
                    if (used + bytes > capacity) {
                        return capacity_plan();
                    }
                } else {
                    const int64_t future_capacity = remaining_capacity - capacity;
                    const bool must_take          = remaining_bytes - used > future_capacity;
                    if (!must_take && used > 0) {
                        if (used >= target || used + bytes > capacity) {
                            break;
                        }
                        if (used + bytes > target && target - used <= used + bytes - target) {
                            break;
                        }
                    }
                    if (used + bytes > capacity) {
                        if (must_take) {
                            return capacity_plan();
                        }
                        break;
                    }
                }

                (*backend_by_segment)[segment_index++] = backend_index;
                used += bytes;
            }

            assigned[backend_index] = used;
            remaining_bytes -= used;
            remaining_capacity -= capacity;
        }

        if (segment_index != segment_bytes.size() || remaining_bytes != 0) {
            return capacity_plan();
        }
        if (bytes_by_backend != nullptr) {
            *bytes_by_backend = std::move(assigned);
        }
        return true;
    }

    std::string layer_split_backend_device_display_name(ggml_backend_t backend);
    int layer_split_tensor_block_index(const std::string& name);
    bool partition_graph_cut_layer_split(const char* desc,
                                         ggml_cgraph* gf,
                                         const sd::ggml_graph_cut::Plan& plan,
                                         const std::vector<ggml_backend_t>& split_backends,
                                         const std::vector<size_t>& backend_vram_limits,
                                         size_t primary_backend_vram_limit,
                                         std::unordered_map<const ggml_tensor*, ggml_backend_t>& param_assignments,
                                         const std::function<ggml_tensor*(ggml_tensor*)>& canonical_param_tensor,
                                         GraphCutLayerSplitAssignment* assignment_out);
    void log_graph_cut_layer_split_assignment(const char* desc,
                                              const std::vector<ggml_backend_t>& split_backends,
                                              const GraphCutLayerSplitAssignment& assignment);

}  // namespace sd

#endif  // __SD_CORE_LAYER_SPLIT_PARTITION_H__
