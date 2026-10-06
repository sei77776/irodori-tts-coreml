import Foundation

/// Measured quality of a reference clip, with a three-level verdict and plain Japanese reasons.
///
/// Irodori imitates the reference including its room, noise and microphone character, so a quiet,
/// unclipped 3–10 second clip gives the cleanest speech.
public struct ReferenceQuality: Sendable, Equatable {
    public enum Grade: Int, Sendable, Comparable {
        case good, fair, retake
        public static func < (a: Grade, b: Grade) -> Bool { a.rawValue < b.rawValue }
        /// 「良い」「ふつう」「録り直しがおすすめ」
        public var label: String {
            switch self {
            case .good: return "良い"
            case .fair: return "ふつう"
            case .retake: return "録り直しがおすすめ"
            }
        }
    }

    public enum Issue: String, Sendable, Equatable {
        case noSpeech, noisy, slightlyNoisy, clipping, slightClipping, quiet, slightlyQuiet,
             tooShort, short, long, tooLong

        public var grade: Grade {
            switch self {
            case .noSpeech, .noisy, .clipping, .quiet, .tooShort, .tooLong: return .retake
            case .slightlyNoisy, .slightClipping, .slightlyQuiet, .short, .long: return .fair
            }
        }

        /// A concrete reason in plain Japanese.
        public var message: String {
            switch self {
            case .noSpeech: return "声が聞き取れません。マイクに向かって話してください。"
            case .noisy: return "雑音が大きいです。静かな場所で録り直してください。"
            case .slightlyNoisy: return "少し雑音があります。静かな場所だとさらに良くなります。"
            case .clipping: return "音が割れています。マイクから少し離れるか、声を少し小さくしてください。"
            case .slightClipping: return "ところどころ音が割れています。少しマイクから離れると安心です。"
            case .quiet: return "声が小さいです。マイクに近づくか、少し大きな声で話してください。"
            case .slightlyQuiet: return "声がやや小さめです。もう少しはっきり話すと良くなります。"
            case .tooShort: return "短すぎます。3〜10秒くらい話してください。"
            case .short: return "少し短めです。3〜10秒くらいがおすすめです。"
            case .long: return "少し長めです。3〜10秒くらいがおすすめです。"
            case .tooLong: return "長すぎます。3〜10秒くらいに収めてください。"
            }
        }
    }

    /// Background noise level (dBFS RMS, full-scale square wave = 0 dB).
    public var noiseFloorDB: Double
    /// Average level while speaking (dBFS RMS).
    public var speechLevelDB: Double
    /// `speechLevelDB - noiseFloorDB`.
    public var snrDB: Double
    /// Fraction of samples at (or within 0.1 % of) full scale.
    public var clippingRatio: Double
    /// Highest absolute sample (dBFS).
    public var peakDB: Double
    /// Length that remains after trimming leading/trailing silence (seconds, includes padding).
    public var usableSeconds: Double
    public var issues: [Issue]

    public var grade: Grade { issues.map(\.grade).max() ?? .good }
    public var reasons: [String] { issues.map(\.message) }
    /// Whether the clip can be registered at all (something that sounds like speech was found).
    public var hasSpeech: Bool { !issues.contains(.noSpeech) }
}

/// Gentle, offline clean-up for reference recordings: high-pass, optional mild noise reduction,
/// silence trimming and loudness normalisation. Pure Swift, deterministic, no allocation of audio
/// hardware, so it can be unit-tested anywhere.
///
/// Everything is deliberately mild: strong denoising leaves "musical noise" that Irodori would imitate.
public enum ReferenceCleanup {
    public struct Options: Sendable, Equatable {
        /// High-pass corner (Hz). Applied forward and backward (zero phase), i.e. a 4th-order magnitude response.
        public var highPassHz: Double = 80
        public var trimSilence = true
        /// Silence kept before the first and after the last speech frame.
        public var paddingSeconds: Double = 0.15
        public var fadeSeconds: Double = 0.01
        public var normalize = true
        /// Target RMS of the speech frames (dBFS).
        public var targetSpeechDB: Double = -20
        /// Normalisation never pushes the peak above this (dBFS).
        public var peakCeilingDB: Double = -1
        /// Normalisation never amplifies more than this (dB), so near-silent clips do not explode.
        public var maxGainDB: Double = 30
        /// Mild STFT Wiener noise reduction, applied before normalisation. Off by default.
        public var noiseReduction = false
        /// Lowest gain the noise reduction applies to any time–frequency bin (dB). −12 dB is mild.
        public var noiseReductionFloorDB: Double = -12
        public init() {}
    }

    public struct Result: Sendable {
        public var samples: [Float]
        public var sampleRate: Double
        public var quality: ReferenceQuality
        /// Normalisation gain that was applied (dB).
        public var gainDB: Double
        /// Kept range of the input (sample indices).
        public var keptRange: Range<Int>
    }

    /// Runs the full clean-up.
    /// - Parameters:
    ///   - noiseProfileRange: seconds of the input known to contain only background noise (e.g. the
    ///     "stay silent" lead-in of a recording). `nil` estimates noise from the quietest frames.
    public static func process(_ input: [Float], sampleRate: Double = 48_000,
                               noiseProfileRange: ClosedRange<Double>? = nil,
                               options: Options = Options()) -> Result {
        let raw = input.map { $0.isFinite ? $0 : 0 }
        let clipping = clippingRatio(raw)
        var signal = options.highPassHz > 0 ? highPass(raw, cutoff: options.highPassHz, sampleRate: sampleRate) : raw
        let analysis = FrameAnalysis(signal, sampleRate: sampleRate, noiseProfileRange: noiseProfileRange)

        if options.noiseReduction, !analysis.noiseFrames.isEmpty {
            let hop = analysis.hop
            let noiseRanges = analysis.noiseFrames.map { ($0 * hop)..<min($0 * hop + analysis.frame, signal.count) }
            signal = reduceNoise(signal, noiseRanges: noiseRanges, floorDB: options.noiseReductionFloorDB)
        }

        var kept = 0..<signal.count
        if options.trimSilence, let speech = analysis.speechRange {
            let pad = Int(options.paddingSeconds * sampleRate)
            kept = max(0, speech.lowerBound - pad)..<min(signal.count, speech.upperBound + pad)
        }

        var gainDB = 0.0
        if options.normalize, analysis.hasSpeech {
            let level = rmsDB(signal, frames: analysis.speechFrames, frame: analysis.frame, hop: analysis.hop)
            if level.isFinite { gainDB = min(options.targetSpeechDB - level, options.maxGainDB) }
        }

        var output = Array(signal[kept])
        if options.trimSilence, analysis.speechRange != nil { applyFades(&output, samples: Int(options.fadeSeconds * sampleRate)) }
        if options.normalize {
            let peak = output.reduce(Float(0)) { max($0, abs($1)) }
            if peak > 0 { gainDB = min(gainDB, options.peakCeilingDB - db(Double(peak))) }
            let gain = Float(pow(10, gainDB / 20))
            if gain != 1 { for i in output.indices { output[i] *= gain } }
        }

        var quality = analysis.quality(clippingRatio: clipping, peak: raw.reduce(Float(0)) { max($0, abs($1)) })
        quality.usableSeconds = Double(kept.count) / sampleRate
        quality.issues = classify(quality)
        return Result(samples: output, sampleRate: sampleRate, quality: quality, gainDB: gainDB, keptRange: kept)
    }

    /// Quality measurement only (no processing of the returned audio).
    public static func analyze(_ input: [Float], sampleRate: Double = 48_000,
                               noiseProfileRange: ClosedRange<Double>? = nil) -> ReferenceQuality {
        var options = Options()
        options.normalize = false
        return process(input, sampleRate: sampleRate, noiseProfileRange: noiseProfileRange, options: options).quality
    }

    // MARK: - Classification

    static func classify(_ q: ReferenceQuality) -> [ReferenceQuality.Issue] {
        guard q.speechLevelDB.isFinite, q.snrDB >= 6 else { return [.noSpeech] }
        var issues: [ReferenceQuality.Issue] = []
        if q.clippingRatio > 0.001 { issues.append(.clipping) }
        else if q.clippingRatio > 0.00005 { issues.append(.slightClipping) }
        if q.snrDB < 18 || q.noiseFloorDB > -40 { issues.append(.noisy) }
        else if q.snrDB < 28 || q.noiseFloorDB > -50 { issues.append(.slightlyNoisy) }
        if q.speechLevelDB < -45 { issues.append(.quiet) }
        else if q.speechLevelDB < -36 { issues.append(.slightlyQuiet) }
        if q.usableSeconds < 1.5 { issues.append(.tooShort) }
        else if q.usableSeconds < 3 { issues.append(.short) }
        else if q.usableSeconds > 30 { issues.append(.tooLong) }
        else if q.usableSeconds > 12 { issues.append(.long) }
        return issues
    }

    /// Fraction of samples at full scale (|x| ≥ 0.999).
    public static func clippingRatio(_ x: [Float]) -> Double {
        guard !x.isEmpty else { return 0 }
        return Double(x.reduce(0) { $0 + (abs($1) >= 0.999 ? 1 : 0) }) / Double(x.count)
    }

    // MARK: - High-pass

    /// 2nd-order Butterworth high-pass run forward and backward (zero phase, 4th-order magnitude,
    /// −6 dB at `cutoff`). Edges are extended by odd reflection to avoid start-up transients.
    public static func highPass(_ x: [Float], cutoff: Double, sampleRate: Double) -> [Float] {
        guard x.count > 2, cutoff > 0, cutoff < sampleRate / 2 else { return x }
        let w = 2 * Double.pi * cutoff / sampleRate
        let alpha = sin(w) / (2 * 0.5.squareRoot())
        let cw = cos(w), a0 = 1 + alpha
        let b0 = (1 + cw) / 2 / a0, b1 = -(1 + cw) / a0, b2 = b0
        let a1 = -2 * cw / a0, a2 = (1 - alpha) / a0
        let pad = min(x.count - 1, Int(sampleRate / cutoff * 3))
        var ext = [Double](repeating: 0, count: x.count + 2 * pad)
        let first = Double(x[0]), last = Double(x[x.count - 1])
        for i in 0..<pad { ext[i] = 2 * first - Double(x[pad - i]) }
        for i in 0..<x.count { ext[pad + i] = Double(x[i]) }
        for i in 0..<pad { ext[pad + x.count + i] = 2 * last - Double(x[x.count - 2 - i]) }
        func run(_ v: inout [Double], reversed: Bool) {
            var x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0
            let indices: [Int] = reversed ? Array(v.indices.reversed()) : Array(v.indices)
            for i in indices {
                let xi = v[i]
                let y = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
                x2 = x1; x1 = xi; y2 = y1; y1 = y
                v[i] = y
            }
        }
        run(&ext, reversed: false)
        run(&ext, reversed: true)
        return (0..<x.count).map { Float(ext[pad + $0]) }
    }

    // MARK: - Noise reduction

    /// Mild STFT Wiener noise reduction (1024-point Hann, hop 256) with a decision-directed a-priori
    /// SNR, a gain floor and frequency + time smoothing of the gain to avoid musical noise.
    /// - Parameter noiseRanges: sample ranges that contain only noise; their average spectrum is the profile.
    public static func reduceNoise(_ x: [Float], noiseRanges: [Range<Int>], floorDB: Double = -12,
                                   fftSize: Int = 1024, hop: Int = 256) -> [Float] {
        guard x.count >= fftSize, !noiseRanges.isEmpty else { return x }
        let fft = FFT(size: fftSize)
        let bins = fftSize / 2 + 1
        let window = (0..<fftSize).map { 0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(fftSize)) }

        // Noise power spectrum from noise-only frames.
        var noise = [Double](repeating: 0, count: bins)
        var noiseCount = 0
        var re = [Double](repeating: 0, count: fftSize), im = [Double](repeating: 0, count: fftSize)
        for range in noiseRanges {
            var start = range.lowerBound
            repeat {
                for i in 0..<fftSize {
                    let index = start + i
                    re[i] = index < x.count ? Double(x[index]) * window[i] : 0; im[i] = 0
                }
                fft.forward(&re, &im)
                for k in 0..<bins { noise[k] += re[k] * re[k] + im[k] * im[k] }
                noiseCount += 1
                start += hop
            } while start + fftSize <= range.upperBound
        }
        guard noiseCount > 0 else { return x }
        for k in 0..<bins { noise[k] = max(noise[k] / Double(noiseCount), 1e-20) }

        let floorGain = pow(10, floorDB / 20)
        let padded = fftSize
        let total = x.count + 2 * padded
        var output = [Double](repeating: 0, count: total)
        var norm = [Double](repeating: 0, count: total)
        var previousCleanPower = [Double](repeating: 0, count: bins)
        var previousGain = [Double](repeating: 1, count: bins)
        var gain = [Double](repeating: 1, count: bins)
        var smoothed = [Double](repeating: 1, count: bins)
        let alpha = 0.98, timeSmoothing = 0.5
        var start = 0
        while start + fftSize <= total {
            for i in 0..<fftSize {
                let index = start + i - padded
                re[i] = (index >= 0 && index < x.count) ? Double(x[index]) * window[i] : 0; im[i] = 0
            }
            fft.forward(&re, &im)
            for k in 0..<bins {
                let power = re[k] * re[k] + im[k] * im[k]
                let posterior = power / noise[k]
                let prior = alpha * previousCleanPower[k] / noise[k] + (1 - alpha) * max(posterior - 1, 0)
                gain[k] = max(prior / (1 + prior), floorGain)
            }
            // Frequency smoothing (5-bin average), then first-order temporal smoothing.
            for k in 0..<bins {
                var sum = 0.0, count = 0.0
                for j in max(0, k - 2)...min(bins - 1, k + 2) { sum += gain[j]; count += 1 }
                let target = sum / count
                smoothed[k] = timeSmoothing * previousGain[k] + (1 - timeSmoothing) * target
                smoothed[k] = min(1, max(floorGain, smoothed[k]))
            }
            for k in 0..<bins {
                let power = re[k] * re[k] + im[k] * im[k]
                previousCleanPower[k] = smoothed[k] * smoothed[k] * power
                previousGain[k] = smoothed[k]
                re[k] *= smoothed[k]; im[k] *= smoothed[k]
                if k > 0 && k < fftSize / 2 {
                    re[fftSize - k] = re[k]; im[fftSize - k] = -im[k]
                }
            }
            fft.inverse(&re, &im)
            for i in 0..<fftSize {
                output[start + i] += re[i]
                norm[start + i] += window[i]
            }
            start += hop
        }
        return (0..<x.count).map { i in
            let n = norm[i + padded]
            return n > 1e-6 ? Float(output[i + padded] / n) : x[i]
        }
    }

    // MARK: - Helpers

    static func db(_ amplitude: Double) -> Double { 20 * log10(max(amplitude, 1e-12)) }

    static func rmsDB(_ x: [Float], frames: [Int], frame: Int, hop: Int) -> Double {
        var sum = 0.0, count = 0
        for f in frames {
            let lower = f * hop, upper = min(lower + frame, x.count)
            guard lower < upper else { continue }
            for i in lower..<upper { sum += Double(x[i]) * Double(x[i]) }
            count += upper - lower
        }
        guard count > 0, sum > 0 else { return -.infinity }
        return 10 * log10(sum / Double(count))
    }

    static func applyFades(_ x: inout [Float], samples: Int) {
        let n = min(samples, x.count / 2)
        guard n > 1 else { return }
        for i in 0..<n {
            let g = Float(i) / Float(n)
            x[i] *= g
            x[x.count - 1 - i] *= g
        }
    }
}

/// 20 ms frames every 10 ms: noise floor, speech activity and levels.
struct FrameAnalysis {
    let frame: Int
    let hop: Int
    let levels: [Double]          // dBFS per frame
    let noiseFloorDB: Double
    let speechLevelDB: Double
    let speechFrames: [Int]
    let noiseFrames: [Int]
    let speechRange: Range<Int>?  // samples
    var hasSpeech: Bool { speechRange != nil }

    init(_ x: [Float], sampleRate: Double, noiseProfileRange: ClosedRange<Double>?) {
        let frame = max(1, Int(sampleRate * 0.02))
        let hop = max(1, Int(sampleRate * 0.01))
        self.frame = frame
        self.hop = hop
        let count = x.count < frame ? (x.isEmpty ? 0 : 1) : (x.count - frame) / hop + 1
        var levels = [Double](repeating: -.infinity, count: count)
        var powers = [Double](repeating: 0, count: count)
        for f in 0..<count {
            let lower = f * hop, upper = min(lower + frame, x.count)
            var sum = 0.0
            for i in lower..<upper { sum += Double(x[i]) * Double(x[i]) }
            powers[f] = sum / Double(upper - lower)
            levels[f] = powers[f] > 1e-14 ? 10 * log10(powers[f]) : -140
        }
        self.levels = levels
        // Digital silence (recorder start-up) says nothing about the room; skip it when possible.
        let audible = levels.indices.filter { levels[$0] > -120 }
        var candidates = audible
        if let range = noiseProfileRange {
            let lead = audible.filter { index in
                let t = Double(index * hop) / sampleRate
                return t >= range.lowerBound && t + Double(frame) / sampleRate <= range.upperBound
            }
            if lead.count >= 3 { candidates = lead }
        }
        let floor: Double
        var noise: [Int] = []
        if candidates.isEmpty {
            floor = -140
        } else {
            let sorted = candidates.map { levels[$0] }.sorted()
            if noiseProfileRange != nil && candidates.count < audible.count {
                // Use the median of the silent lead-in, but never above the quietest tenth of the clip
                // (in case the user started talking early).
                let all = audible.map { levels[$0] }.sorted()
                floor = min(sorted[sorted.count / 2], all[all.count / 10])
            } else {
                floor = sorted[max(0, sorted.count / 10)]
            }
            noise = candidates.filter { levels[$0] <= floor + 3 }
        }
        noiseFloorDB = floor
        noiseFrames = noise

        // Speech: clearly above the noise floor, and not more than 45 dB below the loudest frame.
        let loudest = levels.max() ?? -140
        let threshold = max(floor + min(10, max(3, (loudest - floor) / 2)), loudest - 45)
        let active = levels.map { $0 > threshold && loudest - floor >= 6 }
        // Ignore isolated single frames (clicks): require two consecutive active frames.
        var speech: [Int] = []
        for f in active.indices where active[f] {
            if (f > 0 && active[f - 1]) || (f + 1 < active.count && active[f + 1]) { speech.append(f) }
        }
        speechFrames = speech
        if let first = speech.first, let last = speech.last {
            speechRange = (first * hop)..<min(x.count, last * hop + frame)
            let mean = speech.reduce(0.0) { $0 + powers[$1] } / Double(speech.count)
            speechLevelDB = 10 * log10(max(mean, 1e-14))
        } else {
            speechRange = nil
            speechLevelDB = -.infinity
        }
    }

    func quality(clippingRatio: Double, peak: Float) -> ReferenceQuality {
        ReferenceQuality(noiseFloorDB: noiseFloorDB, speechLevelDB: speechLevelDB,
                         snrDB: speechLevelDB.isFinite ? speechLevelDB - noiseFloorDB : 0,
                         clippingRatio: clippingRatio, peakDB: ReferenceCleanup.db(Double(peak)),
                         usableSeconds: 0, issues: [])
    }
}

/// Small iterative radix-2 complex FFT (power-of-two sizes), double precision.
struct FFT {
    let size: Int
    private let cosTable: [Double]
    private let sinTable: [Double]
    private let bitReverse: [Int]

    init(size: Int) {
        precondition(size > 1 && size & (size - 1) == 0, "FFT size must be a power of two")
        self.size = size
        cosTable = (0..<size / 2).map { cos(2 * Double.pi * Double($0) / Double(size)) }
        sinTable = (0..<size / 2).map { sin(2 * Double.pi * Double($0) / Double(size)) }
        var bits = 0
        while (1 << bits) < size { bits += 1 }
        bitReverse = (0..<size).map { i in
            var r = 0, v = i
            for _ in 0..<bits { r = (r << 1) | (v & 1); v >>= 1 }
            return r
        }
    }

    func forward(_ re: inout [Double], _ im: inout [Double]) { transform(&re, &im, sign: -1) }

    /// Inverse transform including the 1/N scale.
    func inverse(_ re: inout [Double], _ im: inout [Double]) {
        transform(&re, &im, sign: 1)
        let scale = 1 / Double(size)
        for i in 0..<size { re[i] *= scale; im[i] *= scale }
    }

    private func transform(_ re: inout [Double], _ im: inout [Double], sign: Double) {
        for i in 0..<size {
            let j = bitReverse[i]
            if j > i { re.swapAt(i, j); im.swapAt(i, j) }
        }
        var length = 2
        while length <= size {
            let half = length / 2, step = size / length
            var start = 0
            while start < size {
                for k in 0..<half {
                    let wr = cosTable[k * step], wi = sign * sinTable[k * step]
                    let a = start + k, b = a + half
                    let tr = re[b] * wr - im[b] * wi
                    let ti = re[b] * wi + im[b] * wr
                    re[b] = re[a] - tr; im[b] = im[a] - ti
                    re[a] += tr; im[a] += ti
                }
                start += length
            }
            length <<= 1
        }
    }
}
