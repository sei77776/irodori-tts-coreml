import CoreML
import Foundation

public enum BeatriceComputeUnits: String, CaseIterable, Identifiable, Sendable {
    case cpuOnly = "CPU"
    case cpuAndGPU = "CPU+GPU"
    case cpuAndNeuralEngine = "CPU+NE"
    case all = "ALL"

    public var id: String { rawValue }

    public var mlComputeUnits: MLComputeUnits {
        switch self {
        case .cpuOnly: return .cpuOnly
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        case .all: return .all
        }
    }
}

/// One function (fixed window of `window` frames) of a Beatrice multifunction .mlpackage.
///
/// Inputs: wav [1,1,W·160] (16 kHz), spk_embed [1,256], kv [1,384,128], codebook [1,512,128],
/// pitch_shift_bins [1]. Outputs: ir_amp / ir_phase [1,257,W], aperiodicity [1,240,W],
/// post_filter [1,512,W], pitch_hz / qp [1,W].
public final class BeatriceModel {
    public let window: Int
    public let functionName: String
    private let model: MLModel
    private let wavInput: MLMultiArray
    private var speakerInputs: [String: MLMultiArray] = [:]
    private let pitchShift: MLMultiArray

    @available(iOS 18.0, macOS 15.0, *)
    public init(package: URL, function: String, window: Int, computeUnits: BeatriceComputeUnits) throws {
        let compiled = try Self.compiledModel(for: package)
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits.mlComputeUnits
        config.functionName = function
        model = try MLModel(contentsOf: compiled, configuration: config)
        self.window = window
        functionName = function
        wavInput = try MLMultiArray(shape: [1, 1, NSNumber(value: window * 160)], dataType: .float32)
        pitchShift = try MLMultiArray(shape: [1], dataType: .float32)
    }

    /// Compiles the .mlpackage once and caches the .mlmodelc under Caches/BeatriceCompiled.
    public static func compiledModel(for package: URL) throws -> URL {
        let fm = FileManager.default
        let caches = try fm.url(for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent("BeatriceCompiled", isDirectory: true)
        try fm.createDirectory(at: caches, withIntermediateDirectories: true)
        let key = "\(package.deletingPathExtension().lastPathComponent)-\(directorySize(package))"
        let target = caches.appendingPathComponent(key + ".mlmodelc", isDirectory: true)
        if fm.fileExists(atPath: target.path) { return target }
        let temp = try MLModel.compileModel(at: package)
        try? fm.removeItem(at: target)
        try fm.moveItem(at: temp, to: target)
        return target
    }

    static func directorySize(_ url: URL) -> Int {
        guard let e = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total = 0
        for case let file as URL in e {
            total += (try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        }
        return total
    }

    public func setSpeaker(embedding: [Float], keyValue: [Float], codebook: [Float]) throws {
        speakerInputs["spk_embed"] = try Self.array(embedding, shape: [1, 256])
        speakerInputs["kv"] = try Self.array(keyValue, shape: [1, 384, 128])
        speakerInputs["codebook"] = try Self.array(codebook, shape: [1, 512, 128])
    }

    static func array(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        precondition(values.count == a.count)
        let p = a.dataPointer.bindMemory(to: Float.self, capacity: a.count)
        values.withUnsafeBufferPointer { p.update(from: $0.baseAddress!, count: values.count) }
        return a
    }

    /// `wav` must hold exactly window·160 samples.
    public func predict(wav: [Float], pitchShiftBins: Float) throws -> BeatriceFrames {
        precondition(wav.count == window * 160)
        guard speakerInputs.count == 3 else { throw BeatriceError.model("speaker not set") }
        let p = wavInput.dataPointer.bindMemory(to: Float.self, capacity: wav.count)
        wav.withUnsafeBufferPointer { p.update(from: $0.baseAddress!, count: wav.count) }
        pitchShift[0] = NSNumber(value: pitchShiftBins)
        var features: [String: MLFeatureValue] = [
            "wav": MLFeatureValue(multiArray: wavInput),
            "pitch_shift_bins": MLFeatureValue(multiArray: pitchShift),
        ]
        for (k, v) in speakerInputs { features[k] = MLFeatureValue(multiArray: v) }
        let out = try model.prediction(from: try MLDictionaryFeatureProvider(dictionary: features))
        func get(_ name: String, _ rows: Int) throws -> [Float] {
            guard let a = out.featureValue(for: name)?.multiArrayValue else {
                throw BeatriceError.model("missing output \(name)")
            }
            return try Self.copy(a, rows: rows, cols: window)
        }
        return BeatriceFrames(frames: window,
                              irAmp: try get("ir_amp", 257),
                              irPhase: try get("ir_phase", 257),
                              aperiodicity: try get("aperiodicity", 240),
                              postFilter: try get("post_filter", 512),
                              pitchHz: try get("pitch_hz", 1))
    }

    /// Copies a [1, rows, cols] or [1, cols] float32 MLMultiArray honoring its strides
    /// (Core ML may pad the innermost dimension).
    static func copy(_ a: MLMultiArray, rows: Int, cols: Int) throws -> [Float] {
        guard a.dataType == .float32 else { throw BeatriceError.model("unexpected output type \(a.dataType.rawValue)") }
        let shape = a.shape.map { $0.intValue }
        let strides = a.strides.map { $0.intValue }
        let rowStride: Int, colStride: Int
        if shape.count == 3 {
            guard shape[1] == rows, shape[2] == cols else { throw BeatriceError.model("shape \(shape)") }
            rowStride = strides[1]; colStride = strides[2]
        } else if shape.count == 2 {
            guard rows == 1, shape[1] == cols else { throw BeatriceError.model("shape \(shape)") }
            rowStride = 0; colStride = strides[1]
        } else {
            throw BeatriceError.model("rank \(shape.count)")
        }
        let span = (rows - 1) * rowStride + (cols - 1) * colStride + 1
        let p = a.dataPointer.bindMemory(to: Float.self, capacity: span)
        var out = [Float](repeating: 0, count: rows * cols)
        for r in 0..<rows {
            for c in 0..<cols { out[r * cols + c] = p[r * rowStride + c * colStride] }
        }
        return out
    }
}
