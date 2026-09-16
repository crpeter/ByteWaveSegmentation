import CoreML
import CoreMedia
import CoreVideo
import Foundation

enum OwnedTemporalError: LocalizedError {
    case invalid(String)
    var errorDescription: String? {
        switch self { case .invalid(let message): return message }
    }
}

enum OwnedTemporalContract {
    static let id = "bytewave.edgetam-temporal-owned.v2"
    static let precision = "mixed-encoder-late-conv33-fanout-and-fp32-attention-iou.v1"
    static let graphRevision = "dense-points-encoder-fanout.v1"
    static let upstream = "7711e012a30a2402c4eaab637bdb00a521302c91"
    static let checkpoint = "ed2d4850b8792c239689b043c47046ec239b6e808a3d9b6ae676c803fd8780df"
    static let attentionDiagnosticContract = "bytewave.propagator-attention.diagnostic.v1"
    static let memoryFP16Precision = "memory-sdpa-fp16-only.diagnostic.v1"
    static let components = ["ImageEncoder", "Initializer", "InitialMemoryEncoder", "Propagator"]
    static let imageOutputs = ["raw_vision_features", "initial_vision_features", "high_res_feature_0", "high_res_feature_1"]
    static let maskOutputs = ["low_res_mask", "high_res_mask", "best_iou", "object_pointer", "object_score"]
    static let memoryOutputs = ["memory_features", "memory_positions"]
    static let inputs: [String: [String]] = [
        "ImageEncoder": ["image"],
        "Initializer": ["initial_vision_features", "high_res_feature_0", "high_res_feature_1", "point_coords", "point_labels"],
        "InitialMemoryEncoder": ["raw_vision_features", "high_res_mask", "object_score"],
        "Propagator": ["raw_vision_features", "high_res_feature_0", "high_res_feature_1",
                       "spatial_bank", "spatial_positions", "pointer_bank", "valid_slots"]
    ]
    static let shapes: [String: [Int]] = [
        "raw_vision_features": [1, 256, 64, 64], "initial_vision_features": [1, 256, 64, 64],
        "high_res_feature_0": [1, 32, 256, 256], "high_res_feature_1": [1, 64, 128, 128],
        "point_coords": [1, 1, 2], "point_labels": [1, 1],
        "low_res_mask": [1, 1, 256, 256], "high_res_mask": [1, 1, 1024, 1024],
        "best_iou": [1], "object_pointer": [1, 256], "object_score": [1, 1],
        "memory_features": [1, 512, 64], "memory_positions": [1, 512, 64],
        "spatial_bank": [1, 7, 512, 64], "spatial_positions": [1, 7, 512, 64],
        "pointer_bank": [1, 16, 256], "valid_slots": [1, 23]
    ]

    static func outputs(_ component: String) -> [String] {
        switch component {
        case "ImageEncoder": return imageOutputs
        case "Initializer": return maskOutputs
        case "InitialMemoryEncoder": return memoryOutputs
        default: return maskOutputs + memoryOutputs
        }
    }

    static func validate(_ model: MLModel, component: String, propagatorVariant: String? = nil) throws {
        let description = model.modelDescription
        let metadata = description.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        guard propagatorVariant == nil || propagatorVariant == "memoryfp16" else {
            throw OwnedTemporalError.invalid("Unknown diagnostic propagator variant.")
        }
        let diagnostic = component == "Propagator" && propagatorVariant == "memoryfp16"
        // Diagnostic packages retain source precision metadata; the explicit
        // contract/variant identifies the four attention operations overridden.
        guard metadata["bytewave.contract"] == (diagnostic ? attentionDiagnosticContract : id),
              (!diagnostic || metadata["bytewave.diagnostic.variant"] == "memoryfp16"),
              metadata["bytewave.precision"] == precision,
              metadata["bytewave.graph"] == graphRevision,
              metadata["bytewave.upstream"] == upstream,
              metadata["bytewave.checkpoint.sha256"] == checkpoint,
              Set(description.inputDescriptionsByName.keys) == Set(inputs[component] ?? []),
              Set(description.outputDescriptionsByName.keys) == Set(outputs(component)) else {
            throw OwnedTemporalError.invalid("\(component) metadata or feature names do not match the owned contract.")
        }
        for (name, feature) in description.inputDescriptionsByName.merging(description.outputDescriptionsByName,
                                                                          uniquingKeysWith: { first, _ in first }) {
            if name == "image" {
                guard let image = feature.imageConstraint, image.pixelsWide == 1024, image.pixelsHigh == 1024 else {
                    throw OwnedTemporalError.invalid("Expected a 1024 × 1024 image input.")
                }
            } else {
                guard let array = feature.multiArrayConstraint,
                      array.shape.map(\.intValue) == shapes[name],
                      array.dataType == (name == "point_labels" ? .int32 : .float32) else {
                    throw OwnedTemporalError.invalid("Unexpected model tensor schema: \(name).")
                }
            }
        }
    }
}

// Owned copies prevent subsequent Core ML calls from changing retained state.
// Tensor storage is in logical row-major order even when model outputs are strided.
struct OwnedTensor {
    let shape: [Int]
    let values: [Float]

    init(shape: [Int], values: [Float]) throws {
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }),
              shape.reduce(1, *) == values.count, values.allSatisfy(\.isFinite) else {
            throw OwnedTemporalError.invalid("Invalid shape or non-finite tensor.")
        }
        self.shape = shape
        self.values = values
    }

    init(_ array: MLMultiArray, name: String) throws {
        let shape = array.shape.map(\.intValue)
        guard shape == OwnedTemporalContract.shapes[name], array.dataType == .float32 else {
            throw OwnedTemporalError.invalid("Unexpected output type/shape: \(name).")
        }
        let strides = array.strides.map(\.intValue)
        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        var values = [Float](repeating: 0, count: array.count)
        var expectedStride = 1
        var contiguous = true
        for dimension in shape.indices.reversed() {
            if shape[dimension] > 1 && strides[dimension] != expectedStride { contiguous = false }
            expectedStride *= shape[dimension]
        }
        for index in values.indices {
            if index % 16384 == 0 { try Task.checkCancellation() }
            var offset = index
            if !contiguous {
                var remainder = index
                offset = 0
                for dimension in shape.indices.reversed() {
                    offset += (remainder % shape[dimension]) * strides[dimension]
                    remainder /= shape[dimension]
                }
            }
            let value = pointer[offset]
            guard value.isFinite else { throw OwnedTemporalError.invalid("Non-finite output: \(name).") }
            values[index] = value
        }
        self.shape = shape
        self.values = values
    }

    func multiArray() throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        // Fresh MLMultiArray allocations are normally contiguous. Check instead
        // of assuming so; the fallback honors the actual strides.
        let strides = array.strides.map(\.intValue)
        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        var expectedStride = 1
        var contiguous = true
        for dimension in shape.indices.reversed() {
            if shape[dimension] > 1 && strides[dimension] != expectedStride { contiguous = false }
            expectedStride *= shape[dimension]
        }
        if contiguous {
            values.withUnsafeBufferPointer { source in
                if let base = source.baseAddress { pointer.update(from: base, count: values.count) }
            }
            return array
        }
        for index in values.indices {
            var remainder = index
            var offset = 0
            for dimension in shape.indices.reversed() {
                offset += (remainder % shape[dimension]) * strides[dimension]
                remainder /= shape[dimension]
            }
            pointer[offset] = values[index]
        }
        return array
    }
}

struct OwnedStateToken: Equatable {
    let generation: UInt64
    let index: Int
}

struct OwnedStateSummary: Codable, Sendable {
    let acceptedFrames: Int
    let spatialEntries: Int
    let pointerEntries: Int
}

// This session is owned exclusively by the probe actor. No full-video masks or
// future frames are retained. New selection/video/seek must start with reset().
final class OwnedTemporalState {
    private struct Spatial {
        let index: Int
        let features: OwnedTensor
        let positions: OwnedTensor
    }
    private var generation: UInt64 = 0
    private var nextIndex = 0
    private var lastTime: CMTime?
    private var condition: Spatial?
    private var conditionPointer: OwnedTensor?
    private var recent: [Spatial] = []
    private var pointers: [(index: Int, tensor: OwnedTensor)] = []

    init() { reset() }
    var token: OwnedStateToken { OwnedStateToken(generation: generation, index: nextIndex) }
    var summary: OwnedStateSummary {
        OwnedStateSummary(acceptedFrames: nextIndex, spatialEntries: (condition == nil ? 0 : 1) + recent.count,
                          pointerEntries: (conditionPointer == nil ? 0 : 1) + pointers.count)
    }

    func reset() {
        generation &+= 1
        nextIndex = 0
        lastTime = nil
        condition = nil
        conditionPointer = nil
        recent.removeAll()
        pointers.removeAll()
    }

    func validateTime(_ time: CMTime) throws {
        guard time.isNumeric, time.timescale > 0, time.epoch == 0,
              lastTime.map({ CMTimeCompare(time, $0) > 0 }) ?? true else {
            throw OwnedTemporalError.invalid("Timestamp did not advance. Reset and reselect after seeking.")
        }
    }

    func pack(at time: CMTime) throws -> [String: OwnedTensor] {
        try validateTime(time)
        guard let condition, let conditionPointer else {
            throw OwnedTemporalError.invalid("Initialize a subject before propagation.")
        }
        let spatialSize = 512 * 64
        var features = [Float](repeating: 0, count: 7 * spatialSize)
        var positions = features
        var pointerBank = [Float](repeating: 0, count: 16 * 256)
        var valid = [Float](repeating: 0, count: 23)
        features.replaceSubrange(0..<spatialSize, with: condition.features.values)
        positions.replaceSubrange(0..<spatialSize, with: condition.positions.values)
        pointerBank.replaceSubrange(0..<256, with: conditionPointer.values)
        valid[0] = 1
        valid[7] = 1
        for slot in 1...6 {
            let lag = 7 - slot
            if let entry = recent.first(where: { $0.index == nextIndex - lag }) {
                let range = (slot * spatialSize)..<((slot + 1) * spatialSize)
                features.replaceSubrange(range, with: entry.features.values)
                positions.replaceSubrange(range, with: entry.positions.values)
                valid[slot] = 1
            }
        }
        for lag in 1...15 {
            if let entry = pointers.first(where: { $0.index == nextIndex - lag }) {
                pointerBank.replaceSubrange((lag * 256)..<((lag + 1) * 256), with: entry.tensor.values)
                valid[7 + lag] = 1
            }
        }
        return ["spatial_bank": try OwnedTensor(shape: [1, 7, 512, 64], values: features),
                "spatial_positions": try OwnedTensor(shape: [1, 7, 512, 64], values: positions),
                "pointer_bank": try OwnedTensor(shape: [1, 16, 256], values: pointerBank),
                "valid_slots": try OwnedTensor(shape: [1, 23], values: valid)]
    }

    func commit(_ output: [String: OwnedTensor], at time: CMTime, token expected: OwnedStateToken) throws {
        try validateTime(time)
        guard expected == token else { throw OwnedTemporalError.invalid("Stale prediction after reset or state advance.") }
        for name in OwnedTemporalContract.maskOutputs + OwnedTemporalContract.memoryOutputs {
            guard let tensor = output[name], tensor.shape == OwnedTemporalContract.shapes[name],
                  tensor.values.allSatisfy(\.isFinite) else {
                throw OwnedTemporalError.invalid("Incomplete or invalid prediction: \(name).")
            }
        }
        guard let features = output["memory_features"], let positions = output["memory_positions"],
              let pointer = output["object_pointer"] else { throw OwnedTemporalError.invalid("Missing memory output.") }
        let entry = Spatial(index: nextIndex, features: features, positions: positions)
        // All validation precedes mutation, including the absence path.
        if nextIndex == 0 {
            condition = entry
            conditionPointer = pointer
        } else {
            if recent.count == 6 { recent.removeFirst() }
            if pointers.count == 15 { pointers.removeFirst() }
            recent.append(entry)
            pointers.append((nextIndex, pointer))
        }
        nextIndex += 1
        lastTime = time
    }
}

struct OwnedPrediction {
    let tensors: [String: OwnedTensor]
    let modelMilliseconds: [String: Double]
    let sessionStageMilliseconds: [String: Double]
    let state: OwnedStateSummary
}

final class OwnedTemporalSession {
    private let models: [String: MLModel]
    private let state = OwnedTemporalState()

    init(models: [String: MLModel]) { self.models = models }
    func reset() { state.reset() }

    func predict(image: CVPixelBuffer, at time: CMTime, initialPoint: [Double],
                 beforePrediction: (String) throws -> Void = { _ in }) throws -> OwnedPrediction {
        try Task.checkCancellation()
        try state.validateTime(time)
        let token = state.token
        var timings: [String: Double] = [:]
        var stages: [String: Double] = [:]
        func call(_ component: String, _ tensors: [String: OwnedTensor]) throws -> [String: OwnedTensor] {
            guard let model = models[component] else { throw OwnedTemporalError.invalid("Missing model \(component).") }
            let inputStart = ProcessInfo.processInfo.systemUptime
            var values: [String: MLFeatureValue] = [:]
            for name in OwnedTemporalContract.inputs[component] ?? [] {
                if name == "image" {
                    values[name] = MLFeatureValue(pixelBuffer: image)
                } else if name == "point_labels" {
                    let label = try MLMultiArray(shape: [1, 1], dataType: .int32)
                    label[[0, 0]] = 1
                    values[name] = MLFeatureValue(multiArray: label)
                } else {
                    guard let tensor = tensors[name] else { throw OwnedTemporalError.invalid("Missing input \(name).") }
                    values[name] = MLFeatureValue(multiArray: try tensor.multiArray())
                }
            }
            let input = try MLDictionaryFeatureProvider(dictionary: values)
            stages["\(component).inputPreparation"] = (ProcessInfo.processInfo.systemUptime - inputStart) * 1000
            // Persist the impending component outside the prediction timing.
            // A native Metal assertion terminates the process without throwing.
            let checkpointStart = ProcessInfo.processInfo.systemUptime
            try beforePrediction(component)
            stages["\(component).beforePredictionHook"] = (ProcessInfo.processInfo.systemUptime - checkpointStart) * 1000
            let start = ProcessInfo.processInfo.systemUptime
            let prediction = try model.prediction(from: input)
            timings[component] = (ProcessInfo.processInfo.systemUptime - start) * 1000
            let outputStart = ProcessInfo.processInfo.systemUptime
            try Task.checkCancellation()
            var result: [String: OwnedTensor] = [:]
            for name in OwnedTemporalContract.outputs(component) {
                guard let array = prediction.featureValue(for: name)?.multiArrayValue else {
                    throw OwnedTemporalError.invalid("Missing output \(name).")
                }
                result[name] = try OwnedTensor(array, name: name)
            }
            stages["\(component).outputCopyAndValidation"] = (ProcessInfo.processInfo.systemUptime - outputStart) * 1000
            return result
        }
        var features = try call("ImageEncoder", [:])
        var output: [String: OwnedTensor]
        if token.index == 0 {
            guard initialPoint.count == 2, initialPoint.allSatisfy({ $0.isFinite && (0...1).contains($0) }) else {
                throw OwnedTemporalError.invalid("Invalid initial point.")
            }
            features["point_coords"] = try OwnedTensor(shape: [1, 1, 2],
                values: initialPoint.map { Float(min($0 * 1024, 1023)) })
            output = try call("Initializer", features)
            let memory = try call("InitialMemoryEncoder", features.merging(output, uniquingKeysWith: { _, new in new }))
            output.merge(memory, uniquingKeysWith: { _, new in new })
        } else {
            let packStart = ProcessInfo.processInfo.systemUptime
            let bank = try state.pack(at: time)
            stages["state.pack"] = (ProcessInfo.processInfo.systemUptime - packStart) * 1000
            output = try call("Propagator", features.merging(bank, uniquingKeysWith: { _, new in new }))
        }
        try Task.checkCancellation()
        let commitStart = ProcessInfo.processInfo.systemUptime
        try state.commit(output, at: time, token: token)
        stages["state.commit"] = (ProcessInfo.processInfo.systemUptime - commitStart) * 1000
        return OwnedPrediction(tensors: output, modelMilliseconds: timings,
                               sessionStageMilliseconds: stages, state: state.summary)
    }
}
