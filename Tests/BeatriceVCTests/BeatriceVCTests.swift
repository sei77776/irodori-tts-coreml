import CoreML
import XCTest
@testable import BeatriceVC

/// Compares the Swift port against reference data written by
/// Beatrice/tools/export_ios_assets.py (PyTorch + the trainer's DSP). Results are printed as
/// `BEATRICE_RESULT key=value` lines so CI logs carry the numbers.
final class BeatriceVCTests: XCTestCase {
    struct Meta: Decodable {
        struct Stream: Decodable {
            var chunk: Int
            var lookahead: Int
            var crossfade: Int
            var chunks: Int
            var pitch_shift_bins: Float
            var formant_index: Int
        }
        var W: Int
        var speaker: Int
        var dsp_init_phase: Double
        var dsp_window_start_frame: Int64
        var noise_head: [Float]
        var stream: Stream
    }

    static let golden: URL = Bundle.module.url(forResource: "Golden", withExtension: nil)!
    static let assetsDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Examples/BeatriceAssets")

    func meta() throws -> Meta {
        try JSONDecoder().decode(Meta.self, from: Data(contentsOf: Self.golden.appendingPathComponent("meta.json")))
    }

    func load(_ name: String) throws -> [Float] {
        try BeatriceAssets.readFloats(Self.golden.appendingPathComponent(name + ".f32"))
    }

    func snr(_ ref: [Float], _ est: [Float]) -> Double {
        precondition(ref.count == est.count)
        var s = 0.0, e = 0.0
        for i in 0..<ref.count {
            s += Double(ref[i]) * Double(ref[i])
            let d = Double(ref[i]) - Double(est[i])
            e += d * d
        }
        return e == 0 ? 999 : 10 * log10(s / e)
    }

    func report(_ key: String, _ value: Double) {
        print(String(format: "BEATRICE_RESULT %@=%.3f", key, value))
    }

    // MARK: - FFT

    func testRealFFTMatchesNaiveDFT() throws {
        var rng = SystemRandomNumberGenerator()
        for n in [480, 512, 768] {
            let fft = try XCTUnwrap(RealFFT(n: n))
            let count = n == 768 ? 240 : n // zero padding path for 768
            let x = (0..<count).map { _ in Float.random(in: -1...1, using: &rng) }
            var re = [Float](), im = [Float]()
            fft.forward(x, &re, &im)
            var maxErr = 0.0, maxRef = 0.0
            for k in 0...(n / 2) {
                var r = 0.0, i = 0.0
                for t in 0..<count {
                    let a = -2 * Double.pi * Double(k * t) / Double(n)
                    r += Double(x[t]) * cos(a); i += Double(x[t]) * sin(a)
                }
                maxErr = max(maxErr, abs(r - Double(re[k])), abs(i - Double(im[k])))
                maxRef = max(maxRef, abs(r), abs(i))
            }
            report("fft\(n)_forward_rel_err", maxErr / maxRef)
            XCTAssertLessThan(maxErr / maxRef, 1e-5, "forward n=\(n)")
            // inverse of a random Hermitian spectrum vs naive irfft
            let sr = (0...(n / 2)).map { _ in Float.random(in: -1...1, using: &rng) }
            let si = (0...(n / 2)).map { _ in Float.random(in: -1...1, using: &rng) }
            var y = [Float]()
            fft.inverse(sr, si, &y)
            var ierr = 0.0, iref = 0.0
            for t in 0..<n {
                var v = Double(sr[0]) + Double(sr[n / 2]) * (t % 2 == 0 ? 1 : -1)
                for k in 1..<(n / 2) {
                    let a = 2 * Double.pi * Double(k * t) / Double(n)
                    v += 2 * (Double(sr[k]) * cos(a) - Double(si[k]) * sin(a))
                }
                v /= Double(n)
                ierr = max(ierr, abs(v - Double(y[t]))); iref = max(iref, abs(v))
            }
            report("fft\(n)_inverse_rel_err", ierr / iref)
            XCTAssertLessThan(ierr / iref, 1e-5, "inverse n=\(n)")
        }
    }

    func testNoiseMatchesPython() throws {
        let m = try meta()
        XCTAssertEqual(BeatriceNoise.block(start: 0, count: m.noise_head.count), m.noise_head)
    }

    // MARK: - DSP

    func goldenFrames(_ W: Int) throws -> BeatriceFrames {
        BeatriceFrames(frames: W, irAmp: try load("nn_ir_amp"), irPhase: try load("nn_ir_phase"),
                       aperiodicity: try load("nn_aperiodicity"), postFilter: try load("nn_post_filter"),
                       pitchHz: try load("nn_pitch_hz"))
    }

    func testDSPMatchesPython() throws {
        let m = try meta()
        let assets = try BeatriceAssets(directory: Self.assetsDir)
        let dsp = BeatriceDSP(irWindow: assets.irWindow)
        let frames = try goldenFrames(m.W)
        let excitation = try load("dsp_excitation")
        XCTAssertEqual(excitation, BeatriceNoise.block(start: m.dsp_window_start_frame * 240, count: (m.W + 1) * 240))
        let ref = try load("dsp_out")
        let t0 = Date()
        let full = dsp.synthesize(frames, initPhase: m.dsp_init_phase, excitation: excitation).wave
        report("dsp_full_ms", Date().timeIntervalSince(t0) * 1000)
        let s = snr(ref, full)
        report("dsp_vs_python_snr_db", s)
        XCTAssertGreaterThan(s, 60)
        for from in [10, 50] {
            let part = dsp.synthesize(frames, initPhase: m.dsp_init_phase, excitation: excitation, fromFrame: from).wave
            let ps = snr(Array(full[(from * 240)...]), Array(part[(from * 240)...]))
            report("dsp_partial\(from)_vs_full_snr_db", ps)
            XCTAssertGreaterThan(ps, 100)
        }
    }

    // MARK: - Core ML model and streaming (macOS 15+: multifunction models)

    func makeModel(_ key: String, units: BeatriceComputeUnits = .cpuOnly) throws -> (BeatriceModel, BeatriceAssets) {
        guard #available(macOS 15.0, iOS 18.0, *) else { throw XCTSkip("multifunction Core ML needs macOS 15") }
        let m = try meta()
        let assets = try BeatriceAssets(directory: Self.assetsDir)
        let url = try XCTUnwrap(assets.modelURL(key), "model \(key) missing")
        let t0 = Date()
        let model = try BeatriceModel(package: url, function: "w\(m.W)", window: m.W, computeUnits: units)
        report("\(key)_load_ms", Date().timeIntervalSince(t0) * 1000)
        let info = try XCTUnwrap(assets.manifest.voices.first { $0.id == m.speaker })
        let voice = try assets.loadVoice(info)
        try model.setSpeaker(embedding: assets.speakerEmbedding(voice, formantSemitones: 0),
                             keyValue: voice.keyValue, codebook: voice.codebook)
        return (model, assets)
    }

    func checkModel(_ key: String, minSNR: Double) throws {
        let m = try meta()
        let (model, _) = try makeModel(key)
        let wav = try load("nn_wav")
        var out = try model.predict(wav: wav, pitchShiftBins: 0)
        var times: [Double] = []
        for _ in 0..<10 {
            let t0 = Date()
            out = try model.predict(wav: wav, pitchShiftBins: 0)
            times.append(Date().timeIntervalSince(t0) * 1000)
        }
        report("\(key)_w\(m.W)_cpu_predict_median_ms", times.sorted()[times.count / 2])
        let ref = try goldenFrames(m.W)
        let pairs: [(String, [Float], [Float])] = [
            ("ir_amp", ref.irAmp, out.irAmp), ("ir_phase", ref.irPhase, out.irPhase),
            ("aperiodicity", ref.aperiodicity, out.aperiodicity), ("post_filter", ref.postFilter, out.postFilter),
            ("pitch_hz", ref.pitchHz, out.pitchHz),
        ]
        for (name, r, e) in pairs {
            let s = snr(r, e)
            report("\(key)_\(name)_snr_db", s)
            XCTAssertGreaterThan(s, minSNR, "\(key) \(name)")
        }
        let equal = zip(ref.pitchHz, out.pitchHz).filter { abs($0 - $1) < 1e-3 * max(1, abs($0)) }.count
        report("\(key)_pitch_equal_pct", 100 * Double(equal) / Double(ref.pitchHz.count))
    }

    func testFP32ModelMatchesPython() throws { try checkModel("fp32", minSNR: 60) }

    func testMixedModelRunsAndRoughlyMatches() throws { try checkModel("mixed", minSNR: 15) }

    func testStreamingMatchesPython() throws {
        let m = try meta()
        let (model, assets) = try makeModel("fp32")
        let streamer = BeatriceStreamer(predictor: model, dsp: BeatriceDSP(irWindow: assets.irWindow),
                                        chunk: m.stream.chunk, lookahead: m.stream.lookahead,
                                        crossfade: m.stream.crossfade)
        streamer.pitchShiftBins = m.stream.pitch_shift_bins
        let input = try load("stream_in")
        let ref = try load("stream_out")
        var out: [Float] = []
        var times: [Double] = []
        let n = streamer.inputChunkSamples
        for k in 0..<m.stream.chunks {
            let t0 = Date()
            out += try streamer.process(Array(input[(k * n)..<((k + 1) * n)]))
            times.append(Date().timeIntervalSince(t0) * 1000)
        }
        XCTAssertEqual(out.count, ref.count)
        let s = snr(ref, out)
        report("stream_vs_python_snr_db", s)
        report("stream_call_median_ms", times.sorted()[times.count / 2])
        report("stream_call_max_ms", times.max() ?? 0)
        XCTAssertGreaterThan(s, 30)
    }
}
