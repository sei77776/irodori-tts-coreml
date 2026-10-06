import Foundation

public protocol BeatriceWindowPredictor: AnyObject {
    var window: Int { get }
    func predict(wav: [Float], pitchShiftBins: Float) throws -> BeatriceFrames
}

extension BeatriceModel: BeatriceWindowPredictor {}

/// Pseudo-streaming voice conversion: every chunk shifts a fixed window
/// [left context | chunk | look-ahead] of 16 kHz input, runs the whole model on it and emits the
/// chunk's 24 kHz output. The pulse phase is carried across calls and the first `crossfade` frames
/// are cross-faded into the previous call's look-ahead tail.
/// Mirrors `stream_reference` in Beatrice/tools/export_ios_assets.py.
public final class BeatriceStreamer {
    public static let inHop = 160
    public static let outHop = 240
    public let window: Int
    public let chunk: Int
    public let lookahead: Int
    public let crossfade: Int
    public var context: Int { window - chunk - lookahead }
    public var pitchShiftBins: Float = 0
    /// Optional input high-pass (16 kHz, applied before the model) against hum / rumble.
    public var inputHighPass: BeatriceHighPass?
    /// Optional output high-pass (24 kHz) against the vocoder's low-frequency drift (the pulses'
    /// DC component can only be positive in the trainer's vocoder).
    public var outputHighPass: BeatriceHighPass?
    /// Optional noise gate (nil = the trainer-equivalent output).
    public var gate: BeatriceNoiseGate?
    /// Linear gain applied to the 16 kHz input before everything else (1 = unchanged).
    public var inputGain: Float = 1
    /// Skip the pulse train in frames whose estimated pitch (before the user's pitch shift) is at or
    /// below `noisePitchMaxBin` (bin 24 ≈ 65.5 Hz). The pretrained pitch estimator never reports
    /// "unvoiced"; for silence and background noise it settles on a very low pitch (bin 18 ≈ 62.6 Hz),
    /// and the vocoder then emits a constant low pulse train — an audible buzz with real
    /// microphone noise. Human speech rarely goes that low. `nil` = trainer behaviour.
    public var noisePitchMaxBin: Int?
    /// Per-frame information about the most recent output chunk (for diagnostics).
    public struct FrameInfo: Codable, Sendable {
        public var pitchHz: Float
        public var inputLevelDB: Float
        public var gateOpen: Bool
        public var pulses: Bool
    }
    public private(set) var lastFrames: [FrameInfo] = []
    /// Input level (dBFS, high-passed) of the most recent 10 ms frames, newest last (≈ 4 s).
    public private(set) var recentInputLevelsDB: [Float] = []
    /// Input samples per call / output samples per call.
    public var inputChunkSamples: Int { chunk * Self.inHop }
    public var outputChunkSamples: Int { chunk * Self.outHop }

    private let predictor: BeatriceWindowPredictor
    private let dsp: BeatriceDSP
    private var history: [Float]
    private var phase = 0.0
    private var previousTail: [Float]?
    private var chunkIndex: Int64 = 0
    private let fade: [Float]

    public init(predictor: BeatriceWindowPredictor, dsp: BeatriceDSP, chunk: Int,
                lookahead: Int = 4, crossfade: Int = 2) {
        precondition(chunk > 0 && crossfade <= lookahead && predictor.window > chunk + lookahead)
        self.predictor = predictor
        self.dsp = dsp
        window = predictor.window
        self.chunk = chunk
        self.lookahead = lookahead
        self.crossfade = crossfade
        history = [Float](repeating: 0, count: predictor.window * Self.inHop)
        let n = crossfade * Self.outHop
        fade = (0..<n).map { n > 1 ? Float(Double($0) / Double(n - 1)) : 1 }
    }

    public func reset() {
        inputHighPass?.reset()
        outputHighPass?.reset()
        gate?.reset()
        recentInputLevelsDB = []
        history = [Float](repeating: 0, count: window * Self.inHop)
        phase = 0
        previousTail = nil
        chunkIndex = 0
    }

    /// `input`: `inputChunkSamples` samples at 16 kHz. Returns `outputChunkSamples` samples at 24 kHz,
    /// delayed by the look-ahead (`lookahead` frames).
    public func process(_ input: [Float]) throws -> [Float] {
        precondition(input.count == inputChunkSamples)
        var x = input
        if inputGain != 1 { for i in 0..<x.count { x[i] *= inputGain } }
        inputHighPass?.process(&x)
        history.removeFirst(x.count)
        history.append(contentsOf: x)
        chunkIndex += 1
        let hop = Self.outHop
        let windowStart = chunkIndex * Int64(chunk) - Int64(window) // global frame of window frame 0
        let frames = try predictor.predict(wav: history, pitchShiftBins: pitchShiftBins)
        let excitation = BeatriceNoise.block(start: windowStart * Int64(hop), count: (window + 1) * hop)
        let a = context * hop, b = (context + chunk) * hop
        // initial pulse phase such that the phase just before sample `a` equals the carried phase
        let nf = BeatriceDSP.normalizedFrequency(frames.pitchHz)
        var before = 0.0
        if a > 1 { for j in 1..<a { before += Double(nf[j]) } }
        var initPhase = phase - before
        initPhase -= initPhase.rounded(.down)
        let levels = BeatriceNoiseGate.frameLevelsDB(history)
        recentInputLevelsDB.append(contentsOf: levels[(window - chunk)...])
        if recentInputLevelsDB.count > 400 { recentInputLevelsDB.removeFirst(recentInputLevelsDB.count - 400) }
        let decision = gate?.decide(levels: levels, context: context, chunk: chunk, crossfade: crossfade,
                                    lookahead: lookahead, windowStart: windowStart)
        var pulseMask = decision?.pulseMask
        if let maxBin = noisePitchMaxBin {
            var mask = pulseMask ?? [Bool](repeating: true, count: window)
            for t in 0..<window {
                let bin = Int((96 * log2(max(frames.pitchHz[t], 1) / 55)).rounded()) - Int(pitchShiftBins.rounded())
                if bin <= maxBin { mask[t] = false }
            }
            pulseMask = mask
        }
        let (wave, cumulative) = dsp.synthesize(frames, initPhase: initPhase, excitation: excitation,
                                                fromFrame: context, pulseMask: pulseMask)
        lastFrames = (0..<chunk).map { j in
            FrameInfo(pitchHz: frames.pitchHz[context + j], inputLevelDB: levels[context + j],
                      gateOpen: decision?.open[j] ?? true, pulses: pulseMask?[context + j] ?? true)
        }
        phase = cumulative[b - 1] - cumulative[b - 1].rounded(.down)
        var segment = Array(wave[a..<b])
        if let tail = previousTail {
            for i in 0..<tail.count { segment[i] = tail[i] * (1 - fade[i]) + segment[i] * fade[i] }
        }
        previousTail = Array(wave[b..<(b + crossfade * hop)])
        if let gate, let decision { gate.apply(&segment, open: decision.open, hop: hop) }
        outputHighPass?.process(&segment)
        return segment
    }
}
