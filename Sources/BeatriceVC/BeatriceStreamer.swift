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
        history = [Float](repeating: 0, count: window * Self.inHop)
        phase = 0
        previousTail = nil
        chunkIndex = 0
    }

    /// `input`: `inputChunkSamples` samples at 16 kHz. Returns `outputChunkSamples` samples at 24 kHz,
    /// delayed by the look-ahead (`lookahead` frames).
    public func process(_ input: [Float]) throws -> [Float] {
        precondition(input.count == inputChunkSamples)
        history.removeFirst(input.count)
        history.append(contentsOf: input)
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
        let (wave, cumulative) = dsp.synthesize(frames, initPhase: initPhase, excitation: excitation,
                                                fromFrame: context)
        phase = cumulative[b - 1] - cumulative[b - 1].rounded(.down)
        var segment = Array(wave[a..<b])
        if let tail = previousTail {
            for i in 0..<tail.count { segment[i] = tail[i] * (1 - fade[i]) + segment[i] * fade[i] }
        }
        previousTail = Array(wave[b..<(b + crossfade * hop)])
        return segment
    }
}
