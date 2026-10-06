import Foundation

/// Second-order Butterworth high-pass (RBJ biquad), streaming (keeps state between calls).
public final class BeatriceHighPass {
    public let cutoff: Double
    public let sampleRate: Double
    private let b0, b1, b2, a1, a2: Double
    private var x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0

    public init(cutoff: Double, sampleRate: Double) {
        self.cutoff = cutoff
        self.sampleRate = sampleRate
        let w0 = 2 * Double.pi * cutoff / sampleRate
        let alpha = sin(w0) / (2 * (1 / 2.0.squareRoot()))
        let cosw = cos(w0)
        let a0 = 1 + alpha
        b0 = (1 + cosw) / 2 / a0
        b1 = -(1 + cosw) / a0
        b2 = (1 + cosw) / 2 / a0
        a1 = -2 * cosw / a0
        a2 = (1 - alpha) / a0
    }

    public func reset() { x1 = 0; x2 = 0; y1 = 0; y2 = 0 }

    public func process(_ x: inout [Float]) {
        for i in 0..<x.count {
            let v = Double(x[i])
            let y = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2 = x1; x1 = v; y2 = y1; y1 = y
            x[i] = Float(y)
        }
    }
}

/// Noise gate driven by the (high-passed) 16 kHz input level per 10 ms frame.
///
/// A frame is "open" when any input frame from it up to the look-ahead is above the threshold
/// (so the gate opens slightly before speech), and stays open for `holdMs` after the last loud
/// frame. The output gain follows the open/closed target with one-pole smoothing
/// (attack `attackMs`, release `releaseMs`) so switching never produces a step.
public final class BeatriceNoiseGate {
    public var thresholdDB: Float
    public var attackMs: Float = 5
    public var releaseMs: Float = 150
    public var holdMs: Float = 100
    /// Skip the vocoder's pulse train in closed frames (removes the low "buzz" the model emits
    /// for unvoiced/silent input while the gain is still releasing).
    public var suppressPulses = true
    public private(set) var gain: Float = 0
    private var holdLeft = 0
    private var decisions: [Int64: Bool] = [:]

    public init(thresholdDB: Float) { self.thresholdDB = thresholdDB }

    public func reset() {
        gain = 0; holdLeft = 0; decisions.removeAll()
    }

    /// Frame levels in dBFS (RMS of 160 samples).
    public static func frameLevelsDB(_ x: [Float], hop: Int = 160) -> [Float] {
        let n = x.count / hop
        var out = [Float](repeating: -120, count: n)
        for f in 0..<n {
            var s: Float = 0
            for i in (f * hop)..<((f + 1) * hop) { s += x[i] * x[i] }
            out[f] = 10 * log10(s / Float(hop) + 1e-12)
        }
        return out
    }

    /// Decides open/closed for the output frames [context, context + chunk) of the window and
    /// (provisionally) for the cross-fade tail; returns the decisions for the output frames and a
    /// per-window-frame pulse mask (or nil when pulses are not suppressed).
    func decide(levels: [Float], context: Int, chunk: Int, crossfade: Int, lookahead: Int,
                windowStart: Int64) -> (open: [Bool], pulseMask: [Bool]?) {
        let W = levels.count
        let holdFrames = Int((holdMs / 10).rounded())
        func loud(_ j: Int) -> Bool {
            var m: Float = -200
            for k in j...min(W - 1, j + lookahead) { m = max(m, levels[k]) }
            return m >= thresholdDB
        }
        var open = [Bool](repeating: false, count: chunk)
        var hold = holdLeft
        for j in context..<min(W, context + chunk + crossfade) {
            let isOutput = j < context + chunk
            let o: Bool
            if loud(j) { hold = holdFrames; o = true } else if hold > 0 { hold -= 1; o = true } else { o = false }
            decisions[windowStart + Int64(j)] = o
            if isOutput {
                open[j - context] = o
                if j == context + chunk - 1 { holdLeft = hold }
            }
        }
        for key in decisions.keys where key < windowStart { decisions.removeValue(forKey: key) }
        guard suppressPulses else { return (open, nil) }
        let mask = (0..<W).map { decisions[windowStart + Int64($0)] ?? true }
        return (open, mask)
    }

    /// Applies the smoothed gain to the output samples of `open.count` frames of `hop` samples.
    func apply(_ y: inout [Float], open: [Bool], hop: Int = 240, sampleRate: Float = 24_000) {
        let attack = 1 - exp(-1 / max(1, attackMs * sampleRate / 1000))
        let release = 1 - exp(-1 / max(1, releaseMs * sampleRate / 1000))
        for f in 0..<open.count {
            let target: Float = open[f] ? 1 : 0
            for i in (f * hop)..<min(y.count, (f + 1) * hop) {
                gain += (target - gain) * (target > gain ? attack : release)
                y[i] *= gain
            }
        }
    }
}
