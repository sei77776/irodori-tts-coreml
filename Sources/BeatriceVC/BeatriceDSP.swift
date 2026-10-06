import Foundation

/// Per-frame vocoder parameters for one window of `frames` 10 ms frames (Core ML model outputs).
/// Channel-major like the model's `[1, C, frames]` outputs: element (c, t) is at `c * frames + t`.
public struct BeatriceFrames {
    public var frames: Int
    public var irAmp: [Float]        // 257 × frames
    public var irPhase: [Float]      // 257 × frames
    public var aperiodicity: [Float] // 240 × frames
    public var postFilter: [Float]   // 512 × frames
    public var pitchHz: [Float]      // frames

    public init(frames: Int, irAmp: [Float], irPhase: [Float], aperiodicity: [Float],
                postFilter: [Float], pitchHz: [Float]) {
        self.frames = frames
        self.irAmp = irAmp
        self.irPhase = irPhase
        self.aperiodicity = aperiodicity
        self.postFilter = postFilter
        self.pitchHz = pitchHz
    }
}

/// Noise excitation as a pure function of the absolute 24 kHz sample index (SplitMix64),
/// so overlapping windows see the same noise. Mirrors `noise_at` in export_ios_assets.py.
public enum BeatriceNoise {
    @inline(__always)
    public static func value(at index: Int64) -> Float {
        var z = UInt64(bitPattern: index) &+ 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z = z ^ (z >> 31)
        return Float(Double(z >> 40) / Double(1 << 24) - 0.5)
    }

    public static func block(start: Int64, count: Int) -> [Float] {
        (0..<count).map { value(at: start + Int64($0)) }
    }
}

/// Beatrice v2 vocoder DSP (the part outside the neural network), ported from the trainer's
/// `Vocoder.forward`: pulse-train overlap-add of per-frame impulse responses, band-limited noise,
/// and a per-frame 768-point FFT post filter. Hop 240 samples at 24 kHz.
///
/// `fromFrame` skips work for earlier frames: the result is exact for samples ≥ fromFrame·240
/// (and undefined before that). The phase accumulator always covers the whole window.
public final class BeatriceDSP {
    public static let hop = 240
    public let irWindow: [Float]
    private let fft512: RealFFT
    private let fft480: RealFFT
    private let fft768: RealFFT
    private let hann480: [Float]

    public init(irWindow: [Float]) {
        precondition(irWindow.count == 512)
        self.irWindow = irWindow
        fft512 = RealFFT(n: 512)!
        fft480 = RealFFT(n: 480)!
        fft768 = RealFFT(n: 768)!
        // torch.hann_window(480) (periodic)
        hann480 = (0..<480).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / 480)) }
    }

    /// Per-sample normalized frequency `pitch / 24000` (float32, as in the trainer).
    public static func normalizedFrequency(_ pitchHz: [Float]) -> [Float] {
        var nf = [Float](repeating: 0, count: pitchHz.count * hop)
        for t in 0..<pitchHz.count {
            let v = pitchHz[t] / Float(24000)
            for j in 0..<hop { nf[t * hop + j] = v }
        }
        return nf
    }

    /// - Returns: the waveform (frames·240 samples) and the cumulative pulse phase per sample.
    /// - Parameter pulseMask: optional per-frame flag; frames marked `false` get no pulses
    ///   (periodic component), e.g. frames a noise gate considers silent. `nil` = trainer behaviour.
    public func synthesize(_ f: BeatriceFrames, initPhase: Double, excitation: [Float],
                           fromFrame: Int = 0, pulseMask: [Bool]? = nil) -> (wave: [Float], cumulativePhase: [Double]) {
        let hop = Self.hop
        let L = f.frames
        let N = L * hop
        precondition(excitation.count >= (L + 1) * hop)
        // --- pulse phase (float32 increments, float64 accumulation)
        var nf = Self.normalizedFrequency(f.pitchHz)
        nf[0] = Float(initPhase)
        var cum = [Double](repeating: 0, count: N)
        var phase = [Float](repeating: 0, count: N)
        var acc = 0.0
        for i in 0..<N {
            acc += Double(nf[i])
            cum[i] = acc
            phase[i] = Float(acc - acc.rounded(.down))
        }
        let s0 = max(0, fromFrame - 5)

        // --- periodic part: overlap-add of windowed impulse responses at pulse positions
        var periodic = [Float](repeating: 0, count: N + 512)
        var re = [Float](repeating: 0, count: 257), im = re
        var ir = [Float](repeating: 0, count: 512)
        let dTheta = Float(-2 * Double.pi / 512)
        var i = max(0, s0 * hop - 512)
        while i < N - 1 {
            if phase[i] > phase[i + 1], pulseMask?[i / hop] ?? true {
                let numer: Float = 1 - phase[i]
                let frac = numer / (numer + phase[i + 1])
                let t = i / hop
                for k in 0..<257 {
                    let a = f.irAmp[k * L + t]
                    let theta = f.irPhase[k * L + t] + Float(k) * dTheta * frac
                    re[k] = a * cos(theta)
                    im[k] = a * sin(theta)
                }
                fft512.inverse(re, im, &ir)
                for j in 0..<512 { periodic[i + j] += ir[j] * irWindow[j] }
            }
            i += 1
        }

        // --- aperiodic part: rectangular-window STFT of the excitation, shaped, Hann overlap-add
        var aperiodic = [Float](repeating: 0, count: (L + 1) * hop)
        var nre = [Float](repeating: 0, count: 241), nim = nre
        var buf480 = [Float](repeating: 0, count: 480)
        excitation.withUnsafeBufferPointer { ex in
            for t in max(0, s0 - 2)..<L {
                fft480.forward(ex.baseAddress! + t * hop, count: 480, &nre, &nim)
                nre[0] = 0; nim[0] = 0
                for k in 1...240 {
                    let g = f.aperiodicity[(k - 1) * L + t]
                    nre[k] *= g
                    nim[k] *= g
                }
                fft480.inverse(nre, nim, &buf480)
                for j in 0..<480 { aperiodic[t * hop + j] += buf480[j] * hann480[j] }
            }
        }

        // --- post filter on (periodic + aperiodic), per frame, 768-point FFT, overlap-add
        var excited = [Float](repeating: 0, count: N)
        for n in 0..<N { excited[n] = periodic[n] + aperiodic[n] }
        var out = [Float](repeating: 0, count: (L - 1) * hop + 768)
        var pre = [Float](repeating: 0, count: 385), pim = pre
        var xre = pre, xim = pre
        var taps = [Float](repeating: 0, count: 512)
        var y = [Float](repeating: 0, count: 768)
        excited.withUnsafeBufferPointer { ex in
            for t in s0..<L {
                for c in 0..<512 { taps[c] = f.postFilter[c * L + t] }
                fft768.forward(taps, &pre, &pim)
                fft768.forward(ex.baseAddress! + t * hop, count: hop, &xre, &xim)
                for k in 0..<385 {
                    let r = xre[k] * pre[k] - xim[k] * pim[k]
                    let m = xre[k] * pim[k] + xim[k] * pre[k]
                    xre[k] = r
                    xim[k] = m
                }
                fft768.inverse(xre, xim, &y)
                for j in 0..<768 { out[t * hop + j] += y[j] }
            }
        }
        let wave = Array(out[120..<(120 + N)])
        return (wave, cum)
    }
}
