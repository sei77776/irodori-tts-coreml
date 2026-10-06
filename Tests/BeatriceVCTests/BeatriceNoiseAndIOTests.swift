import AVFoundation
import XCTest
@testable import BeatriceVC

/// Noise gate / high-pass behaviour and the audio I/O data path (sample-rate conversion, ring
/// buffers, render callbacks) at realistic callback sizes. Numbers are printed as BEATRICE_RESULT.
final class BeatriceNoiseAndIOTests: BeatriceVCTests {
    // Keep the inherited tests from running twice.
    override func testRealFFTMatchesNaiveDFT() throws {}
    override func testNoiseMatchesPython() throws {}
    override func testDSPMatchesPython() throws {}
    override func testFP32ModelMatchesPython() throws {}
    override func testMixedModelRunsAndRoughlyMatches() throws {}
    override func testStreamingMatchesPython() throws {}
    override func testRealtimeEnginePipelineWithoutAudioHardware() throws {}

    func rms(_ x: ArraySlice<Float>) -> Double {
        guard !x.isEmpty else { return 0 }
        return (x.reduce(0.0) { $0 + Double($1 * $1) } / Double(x.count)).squareRoot()
    }

    func db(_ v: Double) -> Double { 20 * log10(max(v, 1e-12)) }

    func maxStep(_ x: [Float]) -> Float {
        var m: Float = 0
        for i in 1..<x.count { m = max(m, abs(x[i] - x[i - 1])) }
        return m
    }

    func sine(_ hz: Double, rate: Double, count: Int, amp: Float = 0.5) -> [Float] {
        (0..<count).map { amp * Float(sin(2 * Double.pi * hz * Double($0) / rate)) }
    }

    /// Fraction of energy below `hz` (direct DFT on a decimated grid is too slow; use RealFFT).
    func lowBandFraction(_ x: [Float], below hz: Double, rate: Double) -> Double {
        var n = 1
        while n < x.count { n *= 2 }
        let fft = RealFFT(n: n)!
        var re = [Float](), im = [Float]()
        fft.forward(x, &re, &im)
        var low = 0.0, total = 0.0
        for k in 1..<(n / 2) {
            let p = Double(re[k] * re[k] + im[k] * im[k])
            total += p
            if Double(k) * rate / Double(n) < hz { low += p }
        }
        return low / max(total, 1e-30)
    }

    func makeStreamer() throws -> (BeatriceStreamer, BeatriceStreamer) {
        let (m1, assets) = try makeModel("fp32")
        let (m2, _) = try makeModel("fp32")
        let a = BeatriceStreamer(predictor: m1, dsp: BeatriceDSP(irWindow: assets.irWindow), chunk: 10)
        let b = BeatriceStreamer(predictor: m2, dsp: BeatriceDSP(irWindow: assets.irWindow), chunk: 10)
        return (a, b)
    }

    func run(_ s: BeatriceStreamer, _ input: [Float]) throws -> [Float] {
        let n = s.inputChunkSamples
        var out: [Float] = []
        for k in 0..<(input.count / n) { out += try s.process(Array(input[(k * n)..<((k + 1) * n)])) }
        return out
    }

    // MARK: - filters and gate

    func testHighPassFilters() {
        let hp = BeatriceHighPass(cutoff: 75, sampleRate: 16_000)
        var low = sine(30, rate: 16_000, count: 16_000)
        hp.process(&low)
        hp.reset()
        var mid = sine(300, rate: 16_000, count: 16_000)
        hp.process(&mid)
        let lowDB = db(rms(low[8000...]) / (0.5 / 2.0.squareRoot()))
        let midDB = db(rms(mid[8000...]) / (0.5 / 2.0.squareRoot()))
        report("hpf75_30hz_gain_db", lowDB)
        report("hpf75_300hz_gain_db", midDB)
        XCTAssertLessThan(lowDB, -14)
        XCTAssertGreaterThan(midDB, -0.5)
    }

    func testGateGainIsSmooth() {
        let gate = BeatriceNoiseGate(thresholdDB: -60)
        var y = [Float](repeating: 1, count: 240 * 40)
        let open = (0..<40).map { ($0 / 5) % 2 == 0 } // toggles every 50 ms
        gate.apply(&y, open: open)
        let step = maxStep(y)
        report("gate_max_gain_step_per_sample", Double(step))
        XCTAssertLessThan(step, 0.01) // 5 ms attack: < 1/120 per sample, no clicks
    }

    func testGateSilenceAndSpeech() throws {
        let (plain, gated) = try makeStreamer()
        gated.gate = BeatriceNoiseGate(thresholdDB: -60)
        let speech = try load("stream_in")
        // 1) digital silence and a -75 dBFS noise floor
        let silence = [Float](repeating: 0, count: speech.count)
        var rng = SystemRandomNumberGenerator()
        let floorNoise = (0..<speech.count).map { _ in Float.random(in: -1...1, using: &rng) * 0.0003 }
        for (name, x) in [("zeros", silence), ("noise75", floorNoise)] {
            plain.reset(); gated.reset()
            let a = try run(plain, x), b = try run(gated, x)
            report("ungated_\(name)_out_rms_db", db(rms(a[...])))
            report("gated_\(name)_out_rms_db", db(rms(b[...])))
            report("ungated_\(name)_out_below100hz_fraction", lowBandFraction(a, below: 100, rate: 24_000))
            XCTAssertLessThan(rms(b[...]), 1e-4, name)
        }
        // 2) speech: gated output ≈ ungated output (gate open during speech)
        plain.reset(); gated.reset()
        let a = try run(plain, speech), b = try run(gated, speech)
        var s = 0.0, e = 0.0
        for i in 0..<a.count { s += Double(a[i] * a[i]); e += Double((a[i] - b[i]) * (a[i] - b[i])) }
        let snr = 10 * log10(s / max(e, 1e-30))
        report("gated_vs_ungated_speech_snr_db", snr)
        XCTAssertGreaterThan(snr, 15)
        report("ungated_speech_max_step", Double(maxStep(a)))
        report("gated_speech_max_step", Double(maxStep(b)))
        XCTAssertLessThanOrEqual(maxStep(b), maxStep(a) + 0.01)
    }

    func testOutputHighPassRemovesVocoderRumble() throws {
        let (plain, filtered) = try makeStreamer()
        filtered.outputHighPass = BeatriceHighPass(cutoff: 50, sampleRate: 24_000)
        let speech = try load("stream_in")
        let a = try run(plain, speech), b = try run(filtered, speech)
        report("out_below50hz_fraction_plain", lowBandFraction(a, below: 50, rate: 24_000))
        report("out_below50hz_fraction_hpf50", lowBandFraction(b, below: 50, rate: 24_000))
        XCTAssertLessThan(lowBandFraction(b, below: 50, rate: 24_000), lowBandFraction(a, below: 50, rate: 24_000))
    }

    // MARK: - audio I/O data path

    func convertOneShot(_ x: [Float], from: Double, to: Double) -> [Float] {
        let inF = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: from, channels: 1, interleaved: false)!
        let outF = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: to, channels: 1, interleaved: false)!
        let conv = AVAudioConverter(from: inF, to: outF)!
        let inBuf = AVAudioPCMBuffer(pcmFormat: inF, frameCapacity: AVAudioFrameCount(x.count))!
        inBuf.frameLength = AVAudioFrameCount(x.count)
        x.withUnsafeBufferPointer { inBuf.floatChannelData![0].update(from: $0.baseAddress!, count: x.count) }
        let outBuf = AVAudioPCMBuffer(pcmFormat: outF, frameCapacity: AVAudioFrameCount(Double(x.count) * to / from + 4096))!
        var done = false
        var error: NSError?
        _ = conv.convert(to: outBuf, error: &error) { _, st in
            if done { st.pointee = .endOfStream; return nil }
            done = true; st.pointee = .haveData; return inBuf
        }
        return Array(UnsafeBufferPointer(start: outBuf.floatChannelData![0], count: Int(outBuf.frameLength)))
    }

    func snr(_ ref: ArraySlice<Float>, _ est: ArraySlice<Float>) -> Double {
        var s = 0.0, e = 0.0
        for (r, x) in zip(ref, est) { s += Double(r * r); e += Double((r - x) * (r - x)) }
        return e == 0 ? 999 : 10 * log10(s / e)
    }

    /// Microphone callbacks of `block` frames at `rate`, render callbacks at the matching 24 kHz
    /// size, inference run synchronously (deterministic). The engine's 16 kHz stream and rendered
    /// output must equal a one-shot conversion + direct streaming of the same input.
    func testIOCallbackSizesAndRates() throws {
        let speech = try load("stream_in")
        var worstIn = 999.0, worstOut = 999.0
        for rate in [48_000.0, 44_100.0] {
            let mic = convertOneShot(speech, from: 16_000, to: rate)
            let ref16 = convertOneShot(mic, from: rate, to: 16_000)
            for block in [240, 256, 480, 512] {
                let (engineStreamer, refStreamer) = try makeStreamer()
                let engine = BeatriceRealtimeEngine()
                try engine.prepareForTesting(streamer: engineStreamer, inputSampleRate: rate)
                engine.stopTimer()
                engine.recordConvertedInput = true
                var rendered: [Float] = []
                var fed = 0
                var renderCalls = 0
                while fed + block <= mic.count {
                    engine.captureForTesting(Array(mic[fed..<(fed + block)]))
                    fed += block
                    engine.waitUntilIdle()
                    let due = Int(Double(fed) / rate * 24_000)
                    let n = due - rendered.count
                    if n > 0 {
                        var buf = [Float](repeating: 0, count: n)
                        buf.withUnsafeMutableBufferPointer { engine.render($0.baseAddress!, n) }
                        rendered += buf
                        renderCalls += 1
                    }
                }
                let refOut = try run(refStreamer, Array(ref16.prefix(engine.convertedInput.count / 1600 * 1600)))
                let lead = engine.silentSamplesBeforePrimed
                let out = rendered[lead...]
                let m = min(out.count, refOut.count)
                let inM = min(engine.convertedInput.count, ref16.count) - 200
                let sIn = snr(ref16[0..<inM], engine.convertedInput[0..<inM])
                let sOut = snr(refOut[0..<m], Array(out)[0..<m])
                let stats = engine.snapshot()
                let tag = "io_\(Int(rate))_\(block)"
                report("\(tag)_input16k_snr_db", sIn)
                report("\(tag)_output_snr_db", sOut)
                report("\(tag)_compared_output_samples", Double(m))
                report("\(tag)_underruns", Double(stats.underruns))
                worstIn = min(worstIn, sIn); worstOut = min(worstOut, sOut)
                XCTAssertGreaterThan(m, 24_000, tag)
                XCTAssertEqual(stats.underruns, 0, tag)
            }
        }
        report("io_worst_input16k_snr_db", worstIn)
        report("io_worst_output_snr_db", worstOut)
        XCTAssertGreaterThan(worstIn, 60)
        XCTAssertGreaterThan(worstOut, 60)
    }
}
