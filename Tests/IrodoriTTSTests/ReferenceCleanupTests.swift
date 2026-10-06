import XCTest
@testable import IrodoriTTS

/// Synthetic-signal tests for the reference clean-up (high-pass, trim, normalisation, mild noise
/// reduction) and the quality check. Numbers are printed as REFERENCE_RESULT for the CI log.
final class ReferenceCleanupTests: XCTestCase {
    let rate = 48_000.0

    // MARK: - signal helpers

    struct Noise {
        var state: UInt64
        mutating func uniform() -> Double {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return (Double(state >> 11) + 0.5) / Double(1 << 53)
        }
        mutating func gaussian() -> Double {
            (-2 * log(uniform())).squareRoot() * cos(2 * Double.pi * uniform())
        }
    }

    func whiteNoise(_ count: Int, db level: Double, seed: UInt64 = 7) -> [Float] {
        var rng = Noise(state: seed)
        let sigma = pow(10, level / 20)
        return (0..<count).map { _ in Float(sigma * rng.gaussian()) }
    }

    /// Harmonic tone with a wandering pitch and a 4 Hz syllable-like envelope, scaled to `db` RMS.
    func speechLike(seconds: Double, db level: Double) -> [Float] {
        let count = Int(seconds * rate)
        var phase = 0.0
        var raw = [Double](repeating: 0, count: count)
        for i in 0..<count {
            let t = Double(i) / rate
            let f0 = 140 + 15 * sin(2 * Double.pi * 0.7 * t)
            phase += 2 * Double.pi * f0 / rate
            var v = 0.0
            for h in 1...20 { v += sin(Double(h) * phase) / Double(h) }
            raw[i] = v * (0.55 + 0.45 * sin(2 * Double.pi * 4 * t))
        }
        let rms = (raw.reduce(0) { $0 + $1 * $1 } / Double(count)).squareRoot()
        let scale = pow(10, level / 20) / rms
        return raw.map { Float($0 * scale) }
    }

    func sine(_ hz: Double, seconds: Double, amp: Float = 0.5) -> [Float] {
        (0..<Int(seconds * rate)).map { amp * Float(sin(2 * Double.pi * hz * Double($0) / rate)) }
    }

    func silence(_ seconds: Double) -> [Float] { [Float](repeating: 0, count: Int(seconds * rate)) }

    func rmsDB<C: Collection>(_ x: C) -> Double where C.Element == Float {
        guard !x.isEmpty else { return -.infinity }
        let power = x.reduce(0.0) { $0 + Double($1) * Double($1) } / Double(x.count)
        return 10 * log10(max(power, 1e-30))
    }

    func peakDB(_ x: [Float]) -> Double { 20 * log10(Double(x.map { abs($0) }.max() ?? 0)) }

    func add(_ a: [Float], _ b: [Float]) -> [Float] { zip(a, b).map { $0 + $1 } }

    func report(_ name: String, _ value: Double) {
        print("REFERENCE_RESULT \(name)=\(String(format: "%.3f", value))")
    }

    // MARK: - tests

    func testFFTRoundTripMatchesDFT() {
        let n = 64
        let fft = FFT(size: n)
        var re = (0..<n).map { sin(Double($0) * 0.37) + 0.2 * cos(Double($0) * 1.3) }
        var im = [Double](repeating: 0, count: n)
        let input = re
        fft.forward(&re, &im)
        for k in [0, 1, 5, 17, 32] {
            var dr = 0.0, di = 0.0
            for t in 0..<n {
                dr += input[t] * cos(-2 * Double.pi * Double(k * t) / Double(n))
                di += input[t] * sin(-2 * Double.pi * Double(k * t) / Double(n))
            }
            XCTAssertEqual(re[k], dr, accuracy: 1e-9)
            XCTAssertEqual(im[k], di, accuracy: 1e-9)
        }
        fft.inverse(&re, &im)
        for t in 0..<n { XCTAssertEqual(re[t], input[t], accuracy: 1e-12) }
    }

    func testHighPassCutsRumbleAndKeepsVoice() {
        let ref = rmsDB(sine(300, seconds: 2))
        let low = ReferenceCleanup.highPass(sine(30, seconds: 2), cutoff: 80, sampleRate: rate)
        let mid = ReferenceCleanup.highPass(sine(300, seconds: 2), cutoff: 80, sampleRate: rate)
        let lowDB = rmsDB(low[24_000..<72_000]) - ref
        let midDB = rmsDB(mid[24_000..<72_000]) - ref
        report("hpf80_30hz_db", lowDB)
        report("hpf80_300hz_db", midDB)
        XCTAssertLessThan(lowDB, -12)
        XCTAssertGreaterThan(midDB, -0.5)
        XCTAssertLessThan(midDB, 0.5)
        // Zero phase: the 300 Hz tone is not shifted.
        var lag0 = 0.0, lagQuarter = 0.0
        let source = sine(300, seconds: 2)
        for i in 24_000..<72_000 { lag0 += Double(mid[i] * source[i]); lagQuarter += Double(mid[i] * source[i - 40]) }
        XCTAssertGreaterThan(lag0, abs(lagQuarter) * 10)
    }

    func testTrimRemovesSilenceAndKeepsPadding() {
        let speech = speechLike(seconds: 2, db: -24)
        let input = silence(1) + speech + silence(1.5)
        var options = ReferenceCleanup.Options()
        options.normalize = false
        let result = ReferenceCleanup.process(input, sampleRate: rate, options: options)
        let seconds = Double(result.samples.count) / rate
        report("trim_output_seconds", seconds)
        XCTAssertEqual(seconds, 2.3, accuracy: 0.05)
        // About 150 ms of silence is kept in front of the speech.
        let firstLoud = result.samples.firstIndex { abs($0) > 0.005 } ?? 0
        report("trim_leading_padding_ms", Double(firstLoud) / rate * 1000)
        XCTAssertEqual(Double(firstLoud) / rate, 0.15, accuracy: 0.03)
        // 10 ms fades: the very first and last samples are silent.
        XCTAssertEqual(result.samples.first ?? 1, 0, accuracy: 1e-6)
        XCTAssertEqual(result.samples.last ?? 1, 0, accuracy: 1e-6)
        XCTAssertEqual(result.quality.usableSeconds, seconds, accuracy: 1e-9)
    }

    func testNormalizationHitsTargetAndRespectsPeakCeiling() {
        let quiet = add(silence(0.5) + speechLike(seconds: 4, db: -38), whiteNoise(Int(4.5 * rate), db: -90))
        let result = ReferenceCleanup.process(quiet, sampleRate: rate)
        let pad = Int(0.15 * rate)
        let speechPart = result.samples[pad..<(result.samples.count - pad)]
        let level = rmsDB(speechPart)
        report("normalized_speech_dbfs", level)
        report("normalized_peak_dbfs", peakDB(result.samples))
        XCTAssertEqual(level, -20, accuracy: 1)
        XCTAssertLessThanOrEqual(peakDB(result.samples), -1 + 1e-3)

        // A very peaky signal: the ceiling wins over the loudness target.
        var spiky = speechLike(seconds: 3, db: -30)
        for i in stride(from: 1000, to: spiky.count, by: 12_000) { spiky[i] = 0.9 }
        let limited = ReferenceCleanup.process(spiky, sampleRate: rate)
        report("spiky_peak_dbfs", peakDB(limited.samples))
        XCTAssertLessThanOrEqual(peakDB(limited.samples), -1 + 1e-3)
        XCTAssertLessThan(rmsDB(limited.samples), -20)
    }

    func testMildNoiseReductionImprovesSNRWithoutDamagingSpeech() {
        let lead = Int(0.5 * rate)
        let speech = speechLike(seconds: 4, db: -26)
        let clean = silence(0.5) + speech
        let noise = whiteNoise(clean.count, db: -40, seed: 11)
        let noisy = add(clean, noise)
        let output = ReferenceCleanup.reduceNoise(noisy, noiseRanges: [0..<lead], floorDB: -12)
        XCTAssertEqual(output.count, noisy.count)
        let region = (lead + 4800)..<(clean.count - 4800)
        func snr(_ y: [Float]) -> Double {
            var s = 0.0, e = 0.0
            for i in region { s += Double(clean[i] * clean[i]); e += Double((y[i] - clean[i]) * (y[i] - clean[i])) }
            return 10 * log10(s / e)
        }
        let before = snr(noisy), after = snr(output)
        report("nr_snr_in_db", before)
        report("nr_snr_out_db", after)
        report("nr_snr_gain_db", after - before)
        XCTAssertGreaterThan(after - before, 3)
        // Mild: the noise-only lead-in is attenuated by at most about the floor (−12 dB), not erased.
        let leadReduction = rmsDB(output[2400..<(lead - 2400)]) - rmsDB(noisy[2400..<(lead - 2400)])
        report("nr_noise_only_change_db", leadReduction)
        XCTAssertLessThan(leadReduction, -6)
        XCTAssertGreaterThan(leadReduction, -14)

        // Clean speech with a barely-there noise profile passes through almost unchanged.
        let quietNoise = whiteNoise(clean.count, db: -85, seed: 3)
        let nearlyClean = add(clean, quietNoise)
        let passed = ReferenceCleanup.reduceNoise(nearlyClean, noiseRanges: [0..<lead], floorDB: -12)
        var s = 0.0, e = 0.0
        for i in region { s += Double(clean[i] * clean[i]); e += Double((passed[i] - clean[i]) * (passed[i] - clean[i])) }
        let residual = 10 * log10(e / s)
        report("nr_clean_residual_db", residual)
        XCTAssertLessThan(residual, -25)
    }

    func testNoiseFloorFromSilentLeadIn() {
        let input = add(silence(0.5) + speechLike(seconds: 4, db: -22), whiteNoise(Int(4.5 * rate), db: -60))
        let quality = ReferenceCleanup.analyze(input, sampleRate: rate, noiseProfileRange: 0.05...0.5)
        report("leadin_noise_floor_db", quality.noiseFloorDB)
        report("leadin_speech_db", quality.speechLevelDB)
        // HPF removes ~0.3 dB of white noise; frame-level spread adds a little.
        XCTAssertEqual(quality.noiseFloorDB, -60, accuracy: 2)
        XCTAssertEqual(quality.speechLevelDB, -22, accuracy: 2)
        XCTAssertEqual(quality.grade, .good, "\(quality.issues)")
    }

    func testQualityClassification() {
        func check(_ input: [Float], lead: Bool = true) -> ReferenceQuality {
            ReferenceCleanup.analyze(input, sampleRate: rate, noiseProfileRange: lead ? 0.05...0.5 : nil)
        }
        func signal(seconds: Double, speech: Double, noise: Double) -> [Float] {
            add(silence(0.5) + speechLike(seconds: seconds, db: speech),
                whiteNoise(Int((seconds + 0.5) * rate), db: noise))
        }
        let good = check(signal(seconds: 5, speech: -22, noise: -70))
        XCTAssertEqual(good.grade, .good, "\(good.issues)")
        XCTAssertTrue(good.reasons.isEmpty)

        let noisy = check(signal(seconds: 5, speech: -22, noise: -36))
        report("noisy_snr_db", noisy.snrDB)
        XCTAssertEqual(noisy.grade, .retake)
        XCTAssertTrue(noisy.issues.contains(.noisy))
        XCTAssertTrue(noisy.reasons.contains { $0.contains("雑音") })

        let slightlyNoisy = check(signal(seconds: 5, speech: -22, noise: -46))
        XCTAssertEqual(slightlyNoisy.grade, .fair, "\(slightlyNoisy.issues)")

        let clipped = check(signal(seconds: 5, speech: -22, noise: -70).map { max(-1, min(1, $0 * 12)) })
        report("clipped_ratio", clipped.clippingRatio)
        XCTAssertGreaterThan(clipped.clippingRatio, 0.001)
        XCTAssertTrue(clipped.issues.contains(.clipping))
        XCTAssertEqual(clipped.grade, .retake)
        XCTAssertTrue(clipped.reasons.contains { $0.contains("割れ") })
        XCTAssertEqual(ReferenceCleanup.clippingRatio([0.1, 1.0, -1.0, 0.5]), 0.5)

        let quiet = check(signal(seconds: 5, speech: -50, noise: -100))
        XCTAssertTrue(quiet.issues.contains(.quiet), "\(quiet.issues)")
        XCTAssertEqual(quiet.grade, .retake)

        let short = check(signal(seconds: 1, speech: -22, noise: -70))
        XCTAssertTrue(short.issues.contains(.tooShort), "\(short.issues)")
        XCTAssertEqual(short.grade, .retake)

        let shortish = check(signal(seconds: 2.3, speech: -22, noise: -70))
        XCTAssertEqual(shortish.issues, [.short])
        XCTAssertEqual(shortish.grade, .fair)

        let long = check(signal(seconds: 15, speech: -22, noise: -70))
        XCTAssertEqual(long.issues, [.long])
        XCTAssertEqual(long.grade, .fair)

        // Imported file without a silent lead-in: the noise floor comes from the quietest frames.
        let imported = check(add(silence(0.3) + speechLike(seconds: 5, db: -22) + silence(0.3),
                                 whiteNoise(Int(5.6 * rate), db: -65)), lead: false)
        report("imported_noise_floor_db", imported.noiseFloorDB)
        XCTAssertEqual(imported.noiseFloorDB, -65, accuracy: 3)
        XCTAssertEqual(imported.grade, .good, "\(imported.issues)")
    }

    func testAllSilenceAndDegenerateInputsDoNotBlowUp() {
        var options = ReferenceCleanup.Options()
        options.noiseReduction = true
        for input in [silence(2), [], [Float](repeating: 0, count: 10), whiteNoise(Int(rate), db: -70),
                      [Float.nan, .infinity, 0, 0.5] + silence(0.1)] {
            let result = ReferenceCleanup.process(input, sampleRate: rate, noiseProfileRange: 0.05...0.5, options: options)
            XCTAssertTrue(result.samples.allSatisfy { $0.isFinite })
            XCTAssertLessThanOrEqual(result.samples.map { abs($0) }.max() ?? 0, 1)
            XCTAssertFalse(result.quality.hasSpeech)
            XCTAssertEqual(result.quality.grade, .retake)
            XCTAssertTrue(result.gainDB.isFinite)
        }
        let silent = ReferenceCleanup.process(silence(2), sampleRate: rate, options: options)
        XCTAssertEqual(silent.samples.count, Int(2 * rate), "nothing to trim against: the clip is kept")
        XCTAssertEqual(silent.samples.map { abs($0) }.max() ?? 1, 0)
        XCTAssertEqual(silent.quality.issues, [.noSpeech])
    }

    func testWholePipelineWithNoiseReduction() {
        let input = add(silence(0.5) + speechLike(seconds: 5, db: -30) + silence(0.8),
                        whiteNoise(Int(6.3 * rate), db: -62))
        var options = ReferenceCleanup.Options()
        options.noiseReduction = true
        let result = ReferenceCleanup.process(input, sampleRate: rate, noiseProfileRange: 0.05...0.5, options: options)
        let seconds = Double(result.samples.count) / rate
        report("pipeline_seconds", seconds)
        report("pipeline_gain_db", result.gainDB)
        XCTAssertEqual(seconds, 5.3, accuracy: 0.1)
        XCTAssertLessThanOrEqual(peakDB(result.samples), -1 + 1e-3)
        XCTAssertEqual(rmsDB(result.samples[7200..<(result.samples.count - 7200)]), -20, accuracy: 1.5)
        XCTAssertEqual(result.quality.grade, .good, "\(result.quality.issues)")
    }

    func testFloatWAVRoundTrip() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let samples: [Float] = [0, 0.5, -0.5, 1.5, -2, .nan] + sine(440, seconds: 0.3)
        try ReferenceAudio.writeWAV(samples: samples, to: url)
        let back = try ReferenceAudio.readSamples(url)
        XCTAssertEqual(back.count, samples.count)
        XCTAssertEqual(back[1], 0.5, accuracy: 1e-3)
        XCTAssertEqual(back[3], 1, accuracy: 1e-3)
        XCTAssertEqual(back[4], -1, accuracy: 1e-3)
        XCTAssertEqual(back[5], 0, accuracy: 1e-3)
    }
}
