import AVFoundation
import Foundation
import os

/// Fixed-capacity single-producer / single-consumer float ring buffer. The audio threads only
/// take an unfair lock for a memcpy (no allocation, no dispatch).
final class FloatRing {
    let capacity: Int
    private let buffer: UnsafeMutablePointer<Float>
    private var readIndex = 0
    private var available = 0
    private let lock: os_unfair_lock_t

    init(capacity: Int) {
        self.capacity = capacity
        buffer = .allocate(capacity: capacity)
        buffer.initialize(repeating: 0, count: capacity)
        lock = .allocate(capacity: 1)
        lock.initialize(to: os_unfair_lock())
    }

    deinit {
        buffer.deallocate()
        lock.deinitialize(count: 1)
        lock.deallocate()
    }

    var count: Int {
        os_unfair_lock_lock(lock); defer { os_unfair_lock_unlock(lock) }
        return available
    }

    /// Appends; when full, the oldest samples are overwritten. Returns how many were overwritten.
    @discardableResult
    func write(_ p: UnsafePointer<Float>, _ n: Int, stride: Int = 1) -> Int {
        os_unfair_lock_lock(lock); defer { os_unfair_lock_unlock(lock) }
        var lost = 0
        for i in 0..<n {
            let w = (readIndex + available) % capacity
            buffer[w] = p[i * stride]
            if available == capacity {
                readIndex = (readIndex + 1) % capacity
                lost += 1
            } else {
                available += 1
            }
        }
        return lost
    }

    func write(_ values: [Float]) {
        values.withUnsafeBufferPointer { _ = write($0.baseAddress!, values.count) }
    }

    /// Copies up to `n` samples; returns how many were read.
    func read(into dst: UnsafeMutablePointer<Float>, _ n: Int) -> Int {
        os_unfair_lock_lock(lock); defer { os_unfair_lock_unlock(lock) }
        let m = min(n, available)
        for i in 0..<m { dst[i] = buffer[(readIndex + i) % capacity] }
        readIndex = (readIndex + m) % capacity
        available -= m
        return m
    }

    func read(_ n: Int) -> [Float]? {
        guard count >= n else { return nil }
        var out = [Float](repeating: 0, count: n)
        let got = out.withUnsafeMutableBufferPointer { read(into: $0.baseAddress!, n) }
        return got == n ? out : nil
    }

    /// Drops the oldest samples so at most `keep` remain; returns how many were dropped.
    @discardableResult
    func trim(keep: Int) -> Int {
        os_unfair_lock_lock(lock); defer { os_unfair_lock_unlock(lock) }
        let extra = available - keep
        guard extra > 0 else { return 0 }
        readIndex = (readIndex + extra) % capacity
        available -= extra
        return extra
    }

    func removeAll() {
        os_unfair_lock_lock(lock); defer { os_unfair_lock_unlock(lock) }
        readIndex = 0; available = 0
    }
}

/// Counters written by the audio threads (one unfair lock, no allocation).
final class AudioThreadCounters {
    struct Values {
        var inputCallbacks = 0
        var inputFrames = 0
        var lastInputFrames = 0
        /// gaps / overlaps in the input timestamps (samples lost or repeated by the system)
        var inputTimestampJumps = 0
        var renderCallbacks = 0
        var lastRenderFrames = 0
        var renderTimestampJumps = 0
        var underruns = 0
        var underrunSamples = 0
        var primed = false
        var silentSamplesBeforePrimed = 0
    }
    private var v = Values()
    private let lock: os_unfair_lock_t = {
        let l = os_unfair_lock_t.allocate(capacity: 1)
        l.initialize(to: os_unfair_lock())
        return l
    }()

    deinit { lock.deinitialize(count: 1); lock.deallocate() }

    func update(_ body: (inout Values) -> Void) {
        os_unfair_lock_lock(lock); body(&v); os_unfair_lock_unlock(lock)
    }

    var values: Values {
        os_unfair_lock_lock(lock); defer { os_unfair_lock_unlock(lock) }
        return v
    }
}

/// Microphone → 16 kHz → BeatriceStreamer (on a serial inference queue, polled by a timer) →
/// 24 kHz → speaker.
public final class BeatriceRealtimeEngine {
    public struct Stats: Sendable {
        public var running = false
        public var chunks = 0
        public var inferenceMedianMs = 0.0
        public var inferenceMaxMs = 0.0
        /// input chunks discarded because inference fell behind
        public var droppedChunks = 0
        /// render callbacks that found the output buffer empty after playback started
        public var underruns = 0
        public var underrunMs = 0.0
        /// discontinuities in the system's input / output timestamps (lost or repeated audio)
        public var inputTimestampJumps = 0
        public var renderTimestampJumps = 0
        public var inputCallbackFrames = 0
        public var renderCallbackFrames = 0
        public var inputSampleRate = 0.0
        public var inputChannels = 0
        public var outputSampleRate = 0.0
        public var voiceProcessing = false
        public var ioBufferDuration = 0.0
        public var hardwareSampleRate = 0.0
        public var inputLatency = 0.0
        public var outputLatency = 0.0
        public var chunkMs = 0.0
        public var contextMs = 0.0
        public var lookaheadMs = 0.0
        public var prebufferMs = 0.0
        /// latest input level (dBFS, after the input high-pass) and the gate state
        public var inputLevelDB: Float = -120
        public var gateThresholdDB: Float?
        public var gateOpen = false
        /// median pitch (Hz, after the pitch shift) of recent frames that produce pulses, and the
        /// share of recent frames that produce pulses (≈ "voiced")
        public var pitchMedianHz: Float = 0
        public var voicedPercent: Double = 0
        /// rough end-to-end estimate: I/O buffers + hardware latency + chunk + look-ahead
        /// + inference + output jitter buffer
        public var estimatedLatencyMs = 0.0
        public var lastError: String?
        public init() {}
    }

    private let engine = AVAudioEngine()
    private var sourceNode: AVAudioSourceNode?
    private var sinkNode: AVAudioSinkNode?
    private var converter: AVAudioConverter?
    private var rawFormat: AVAudioFormat?
    /// microphone samples at the hardware rate (mono), written on the audio thread
    private var rawRing = FloatRing(capacity: 48_000)
    private let inputRing = FloatRing(capacity: 16_000 * 2)
    private let outputRing = FloatRing(capacity: 24_000 * 2)
    private let counters = AudioThreadCounters()
    private let queue = DispatchQueue(label: "BeatriceRealtimeEngine.inference", qos: .userInteractive)
    private var timer: DispatchSourceTimer?
    private let stateLock = NSLock()
    private var stats = Stats()
    private var timings: [Double] = []
    private var streamer: BeatriceStreamer?
    /// output samples buffered before playback starts (and again after an underrun)
    private var prebufferSamples = 0
    private var recording: (input: [Float], output: [Float], frames: [BeatriceStreamer.FrameInfo], remaining: Int)?
    private var recentFrames: [BeatriceStreamer.FrameInfo] = []
    var recordConvertedInput = false
    private(set) var convertedInput: [Float] = []
    private static let inFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000,
                                                channels: 1, interleaved: false)!
    private static let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24_000,
                                                 channels: 1, interleaved: false)!

    public init() {}

    public func snapshot() -> Stats {
        let c = counters.values
        stateLock.lock(); defer { stateLock.unlock() }
        var s = stats
        s.underruns = c.underruns
        s.underrunMs = Double(c.underrunSamples) / 24
        s.inputTimestampJumps = c.inputTimestampJumps
        s.renderTimestampJumps = c.renderTimestampJumps
        s.inputCallbackFrames = c.lastInputFrames
        s.renderCallbackFrames = c.lastRenderFrames
        return s
    }

    /// - Parameters:
    ///   - useSpeaker: iOS only; route output to the built-in speaker (feedback risk).
    ///   - voiceProcessing: Apple's voice processing (echo cancellation / noise suppression) on the input.
    ///   - prebuffer: extra output buffering beyond one chunk, absorbing inference-time jitter.
    public func start(streamer: BeatriceStreamer, preferredIOBuffer: TimeInterval = 0.005,
                      useSpeaker: Bool = false, voiceProcessing: Bool = false,
                      prebuffer: TimeInterval = 0.03) throws {
        stop()
        streamer.reset()
        self.streamer = streamer
        inputRing.removeAll()
        outputRing.removeAll()
        counters.update { $0 = AudioThreadCounters.Values() }
        prebufferSamples = streamer.outputChunkSamples + Int(prebuffer * 24_000)
        recentFrames = []
        stateLock.lock()
        stats = Stats()
        timings = []
        stats.chunkMs = Double(streamer.chunk) * 10
        stats.contextMs = Double(streamer.context) * 10
        stats.lookaheadMs = Double(streamer.lookahead) * 10
        stats.prebufferMs = Double(prebufferSamples) / 24
        stats.gateThresholdDB = streamer.gate?.thresholdDB
        stats.voiceProcessing = voiceProcessing
        stateLock.unlock()

        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: voiceProcessing ? .voiceChat : .measurement,
                                options: useSpeaker ? [.defaultToSpeaker] : [])
        try? session.setPreferredSampleRate(48_000)
        try? session.setPreferredIOBufferDuration(preferredIOBuffer)
        try session.setActive(true)
        #endif

        let input = engine.inputNode
        if input.isVoiceProcessingEnabled != voiceProcessing {
            try input.setVoiceProcessingEnabled(voiceProcessing)
        }
        let hwFormat = input.outputFormat(forBus: 0)
        guard hwFormat.sampleRate > 0, hwFormat.channelCount > 0 else {
            throw BeatriceError.unsupported("マイク入力が見つかりません。")
        }
        try configureInput(sampleRate: hwFormat.sampleRate)
        // AVAudioSinkNode receives input at the I/O buffer size (installTap may deliver ~100 ms blocks)
        let channels = Int(hwFormat.channelCount)
        let stride = hwFormat.isInterleaved ? channels : 1
        let ring = self.rawRing, threadCounters = self.counters
        var expectedTime = -1.0
        let sink = AVAudioSinkNode { (timestamp, frameCount, abl) -> OSStatus in
            let n = Int(frameCount)
            Self.writeChannel0(abl, frames: n, stride: stride, to: ring)
            let t = timestamp.pointee
            let jump = t.mFlags.contains(.sampleTimeValid) && expectedTime >= 0
                && abs(t.mSampleTime - expectedTime) > 0.5
            expectedTime = t.mFlags.contains(.sampleTimeValid) ? t.mSampleTime + Double(n) : -1
            threadCounters.update {
                $0.inputCallbacks += 1; $0.inputFrames += n; $0.lastInputFrames = n
                if jump { $0.inputTimestampJumps += 1 }
            }
            return noErr
        }
        sinkNode = sink
        engine.attach(sink)
        engine.connect(input, to: sink, format: hwFormat)

        var expectedRender = -1.0
        let source = AVAudioSourceNode(format: Self.outFormat) { [weak self] (_, timestamp, frameCount, abl) -> OSStatus in
            guard let self else { return noErr }
            let buffers = UnsafeMutableAudioBufferListPointer(abl)
            guard let raw = buffers[0].mData else { return noErr }
            let n = Int(frameCount)
            let t = timestamp.pointee
            if t.mFlags.contains(.sampleTimeValid) {
                if expectedRender >= 0, abs(t.mSampleTime - expectedRender) > 0.5 {
                    self.counters.update { $0.renderTimestampJumps += 1 }
                }
                expectedRender = t.mSampleTime + Double(n)
            }
            self.render(raw.assumingMemoryBound(to: Float.self), n)
            return noErr
        }
        sourceNode = source
        engine.attach(source)
        engine.connect(source, to: engine.mainMixerNode, format: Self.outFormat)
        engine.prepare()
        try engine.start()
        startTimer()

        stateLock.lock()
        stats.running = true
        stats.inputSampleRate = hwFormat.sampleRate
        stats.inputChannels = channels
        stats.outputSampleRate = engine.outputNode.outputFormat(forBus: 0).sampleRate
        #if os(iOS)
        stats.ioBufferDuration = session.ioBufferDuration
        stats.hardwareSampleRate = session.sampleRate
        stats.inputLatency = session.inputLatency
        stats.outputLatency = session.outputLatency
        #else
        stats.ioBufferDuration = Double(512) / max(hwFormat.sampleRate, 1)
        stats.hardwareSampleRate = hwFormat.sampleRate
        #endif
        updateLatencyLocked()
        stateLock.unlock()
    }

    public func stop() {
        timer?.cancel(); timer = nil
        if engine.isRunning { engine.stop() }
        if let sink = sinkNode {
            engine.detach(sink)
            sinkNode = nil
        }
        if let source = sourceNode {
            engine.detach(source)
            sourceNode = nil
        }
        queue.sync {}
        stateLock.lock(); stats.running = false; stateLock.unlock()
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        #endif
    }

    /// Copies channel 0 of a float32 AudioBufferList (deinterleaved: first buffer; interleaved:
    /// every `stride`-th sample) into the ring. Runs on the audio thread.
    static func writeChannel0(_ abl: UnsafePointer<AudioBufferList>, frames n: Int, stride: Int, to ring: FloatRing) {
        let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: abl))
        guard buffers.count > 0, let raw = buffers[0].mData else { return }
        ring.write(raw.assumingMemoryBound(to: Float.self), n, stride: stride)
    }

    /// nil turns the gate off. Takes effect on the next chunk.
    public func setGateThreshold(_ db: Float?) {
        queue.async { [weak self] in
            guard let self, let streamer = self.streamer else { return }
            if let db {
                if let gate = streamer.gate { gate.thresholdDB = db } else { streamer.gate = BeatriceNoiseGate(thresholdDB: db) }
            } else {
                streamer.gate = nil
            }
            self.stateLock.lock(); self.stats.gateThresholdDB = db; self.stateLock.unlock()
        }
    }

    /// Waits `seconds` while running and returns the background level: the 80th percentile of the
    /// input frame levels (dBFS, after the input high-pass) recorded in that period.
    public func measureNoiseFloor(seconds: Double = 2) async -> Float? {
        try? await Task.sleep(nanoseconds: UInt64(seconds * 1e9))
        return await withCheckedContinuation { cont in
            queue.async { [weak self] in
                guard let levels = self?.streamer?.recentInputLevelsDB, !levels.isEmpty else {
                    cont.resume(returning: nil); return
                }
                let recent = levels.suffix(Int(seconds * 100)).sorted()
                cont.resume(returning: recent[min(recent.count - 1, Int(Double(recent.count) * 0.8))])
            }
        }
    }

    /// Records the next `seconds` of model input (16 kHz) and output (24 kHz) and writes
    /// `beatrice-input.wav` / `beatrice-output.wav` into `directory`.
    public func recordDiagnostics(seconds: Double, to directory: URL) async throws -> [URL] {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            queue.async { [weak self] in
                self?.recording = ([], [], [], Int(seconds * 100))
                cont.resume()
            }
        }
        try? await Task.sleep(nanoseconds: UInt64((seconds + 0.5) * 1e9))
        let data: (input: [Float], output: [Float], frames: [BeatriceStreamer.FrameInfo])? = await withCheckedContinuation { cont in
            queue.async { [weak self] in
                let r = self?.recording
                self?.recording = nil
                cont.resume(returning: r.map { (input: $0.input, output: $0.output, frames: $0.frames) })
            }
        }
        guard let data, !data.input.isEmpty else { throw BeatriceError.unsupported("録音できませんでした（変換中に実行してください）。") }
        let inURL = directory.appendingPathComponent("beatrice-input.wav")
        let outURL = directory.appendingPathComponent("beatrice-output.wav")
        try Self.writeWAV(data.input, sampleRate: 16_000, to: inURL)
        try Self.writeWAV(data.output, sampleRate: 24_000, to: outURL)
        let jsonURL = directory.appendingPathComponent("beatrice-frames.json")
        let s = snapshot()
        let settings: [String: Any] = ["chunkMs": s.chunkMs, "contextMs": s.contextMs, "lookaheadMs": s.lookaheadMs,
                         "inputSampleRate": s.inputSampleRate, "inputChannels": s.inputChannels,
                         "outputSampleRate": s.outputSampleRate, "ioBufferDuration": s.ioBufferDuration,
                         "voiceProcessing": s.voiceProcessing, "gateThresholdDB": s.gateThresholdDB.map { Double($0) as Any } ?? NSNull(),
                         "inferenceMedianMs": s.inferenceMedianMs, "underruns": s.underruns,
                         "inputTimestampJumps": s.inputTimestampJumps, "renderTimestampJumps": s.renderTimestampJumps]
        let frames: [[String: Any]] = data.frames.map { f -> [String: Any] in
            ["pitchHz": Double(f.pitchHz), "inputLevelDB": Double(f.inputLevelDB), "gateOpen": f.gateOpen, "pulses": f.pulses]
        }
        let doc: [String: Any] = [
            "note": "one entry per 10 ms output frame; pitchHz after the pitch shift; inputLevelDB = model input (after gain / high-pass)",
            "settings": settings,
            "frames": frames,
        ]
        let json = try JSONSerialization.data(withJSONObject: doc, options: [.prettyPrinted, .sortedKeys])
        try json.write(to: jsonURL)
        return [inURL, outURL, jsonURL]
    }

    static func writeWAV(_ x: [Float], sampleRate: Double, to url: URL) throws {
        try? FileManager.default.removeItem(at: url)
        let file = try AVAudioFile(forWriting: url, settings: [
            AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: sampleRate, AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false,
        ])
        guard let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false),
              let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(max(1, x.count))) else { return }
        buf.frameLength = AVAudioFrameCount(x.count)
        x.withUnsafeBufferPointer { buf.floatChannelData![0].update(from: $0.baseAddress!, count: x.count) }
        try file.write(from: buf)
    }

    /// Output render: plays once `prebufferSamples` are buffered; after an underrun it waits for the
    /// prebuffer again instead of stuttering through partially filled callbacks.
    func render(_ dst: UnsafeMutablePointer<Float>, _ n: Int) {
        var primed = counters.values.primed
        if !primed, outputRing.count >= prebufferSamples {
            primed = true
            counters.update { $0.primed = true }
        }
        var got = 0
        if primed { got = outputRing.read(into: dst, n) }
        if got < n {
            dst.advanced(by: got).initialize(repeating: 0, count: n - got)
            let missing = n - got
            counters.update {
                if primed {
                    $0.underruns += 1; $0.underrunSamples += missing; $0.primed = false
                } else {
                    $0.silentSamplesBeforePrimed += missing
                }
                $0.renderCallbacks += 1; $0.lastRenderFrames = n
            }
        } else {
            counters.update { $0.renderCallbacks += 1; $0.lastRenderFrames = n }
        }
    }

    // MARK: - test hooks (drive the pipeline without audio hardware)

    func prepareForTesting(streamer: BeatriceStreamer, inputSampleRate: Double, prebuffer: TimeInterval = 0.03) throws {
        streamer.reset()
        self.streamer = streamer
        inputRing.removeAll(); outputRing.removeAll()
        counters.update { $0 = AudioThreadCounters.Values() }
        prebufferSamples = streamer.outputChunkSamples + Int(prebuffer * 24_000)
        try configureInput(sampleRate: inputSampleRate)
        stateLock.lock()
        stats = Stats(); timings = []
        stateLock.unlock()
        startTimer()
    }

    /// Feeds microphone samples (at the configured input rate) as the sink node would.
    func captureForTesting(_ samples: [Float]) {
        rawRing.write(samples)
    }

    var silentSamplesBeforePrimed: Int { counters.values.silentSamplesBeforePrimed }

    func waitUntilIdle() {
        let n = streamer?.inputChunkSamples ?? 1
        while true {
            queue.sync { drain() }
            if rawRing.count == 0 && inputRing.count < n { return }
        }
    }

    func stopTimer() { timer?.cancel(); timer = nil; queue.sync {} }

    private func configureInput(sampleRate: Double) throws {
        guard let mono = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
                                       channels: 1, interleaved: false),
              let conv = AVAudioConverter(from: mono, to: Self.inFormat) else {
            throw BeatriceError.unsupported("入力 \(sampleRate) Hz を 16kHz に変換できません。")
        }
        rawFormat = mono
        converter = conv
        rawRing = FloatRing(capacity: Int(sampleRate))
        convertedInput = []
    }

    private func startTimer() {
        timer?.cancel()
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now(), repeating: .milliseconds(2), leeway: .milliseconds(1))
        t.setEventHandler { [weak self] in self?.drain() }
        timer = t
        t.resume()
    }

    /// Inference queue: hardware-rate samples -> 16 kHz (one persistent converter, so its filter
    /// state carries across calls).
    private func convertPending() {
        guard let converter, let rawFormat else { return }
        let n = rawRing.count
        guard n > 0, let inBuf = AVAudioPCMBuffer(pcmFormat: rawFormat, frameCapacity: AVAudioFrameCount(n)),
              let inData = inBuf.floatChannelData?[0] else { return }
        let got = rawRing.read(into: inData, n)
        inBuf.frameLength = AVAudioFrameCount(got)
        let ratio = Self.inFormat.sampleRate / rawFormat.sampleRate
        let capacity = AVAudioFrameCount(Double(got) * ratio + 64)
        guard let out = AVAudioPCMBuffer(pcmFormat: Self.inFormat, frameCapacity: capacity) else { return }
        var supplied = false
        var error: NSError?
        _ = converter.convert(to: out, error: &error) { _, status in
            if supplied { status.pointee = .noDataNow; return nil }
            supplied = true
            status.pointee = .haveData
            return inBuf
        }
        if let data = out.floatChannelData?[0], out.frameLength > 0 {
            _ = inputRing.write(data, Int(out.frameLength))
            if recordConvertedInput {
                convertedInput.append(contentsOf: UnsafeBufferPointer(start: data, count: Int(out.frameLength)))
            }
        }
    }

    private func drain() {
        guard let streamer else { return }
        let n = streamer.inputChunkSamples
        while true {
            convertPending()
            // bound the backlog: keep at most two chunks waiting
            let dropped = inputRing.count > 3 * n ? inputRing.trim(keep: 2 * n) / n : 0
            if dropped > 0 { stateLock.lock(); stats.droppedChunks += dropped; stateLock.unlock() }
            guard let chunk = inputRing.read(n) else { return }
            let t0 = DispatchTime.now().uptimeNanoseconds
            do {
                let y = try streamer.process(chunk)
                outputRing.write(y)
                // bound output latency against clock drift: keep at most three chunks queued
                outputRing.trim(keep: prebufferSamples + 2 * streamer.outputChunkSamples)
                if var r = recording, r.remaining > 0 {
                    r.input += chunk; r.output += y; r.frames += streamer.lastFrames; r.remaining -= streamer.chunk
                    recording = r
                }
                recentFrames += streamer.lastFrames
                if recentFrames.count > 100 { recentFrames.removeFirst(recentFrames.count - 100) }
            } catch {
                stateLock.lock(); stats.lastError = error.localizedDescription; stateLock.unlock()
            }
            let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
            let level = streamer.recentInputLevelsDB.suffix(streamer.chunk).max() ?? -120
            let open = streamer.gate.map { $0.gain > 0.5 } ?? true
            let voiced = recentFrames.filter { $0.pulses && $0.gateOpen }
            let pitches = voiced.map(\.pitchHz).sorted()
            stateLock.lock()
            stats.pitchMedianHz = pitches.isEmpty ? 0 : pitches[pitches.count / 2]
            stats.voicedPercent = recentFrames.isEmpty ? 0 : 100 * Double(voiced.count) / Double(recentFrames.count)
            timings.append(ms)
            if timings.count > 200 { timings.removeFirst(timings.count - 200) }
            let sorted = timings.sorted()
            stats.inferenceMedianMs = sorted[sorted.count / 2]
            stats.inferenceMaxMs = max(stats.inferenceMaxMs, ms)
            stats.chunks += 1
            stats.inputLevelDB = level
            stats.gateOpen = open
            updateLatencyLocked()
            stateLock.unlock()
        }
    }

    private func updateLatencyLocked() {
        stats.estimatedLatencyMs = (2 * stats.ioBufferDuration + stats.inputLatency + stats.outputLatency) * 1000
            + stats.chunkMs + stats.prebufferMs + stats.lookaheadMs + stats.inferenceMedianMs
    }
}
