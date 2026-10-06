import AVFoundation
import Foundation

/// Thread-safe float FIFO (prototype quality: a lock, no allocation-free guarantee).
final class FloatFIFO {
    private var storage: [Float] = []
    private var head = 0
    private let lock = NSLock()

    var count: Int { lock.lock(); defer { lock.unlock() }; return storage.count - head }

    func append(_ p: UnsafePointer<Float>, _ n: Int) {
        lock.lock(); defer { lock.unlock() }
        storage.append(contentsOf: UnsafeBufferPointer(start: p, count: n))
    }

    func append(_ values: [Float]) {
        lock.lock(); defer { lock.unlock() }
        storage.append(contentsOf: values)
    }

    func pop(_ n: Int) -> [Float]? {
        lock.lock(); defer { lock.unlock() }
        guard storage.count - head >= n else { return nil }
        let out = Array(storage[head..<(head + n)])
        head += n
        compact()
        return out
    }

    /// Drops the oldest samples so that at most `keep` remain; returns how many were dropped.
    @discardableResult func trim(keep: Int) -> Int {
        lock.lock(); defer { lock.unlock() }
        let extra = storage.count - head - keep
        guard extra > 0 else { return 0 }
        head += extra
        compact()
        return extra
    }

    /// Copies up to `n` samples into `dst`; returns how many were available.
    func read(into dst: UnsafeMutablePointer<Float>, _ n: Int) -> Int {
        lock.lock(); defer { lock.unlock() }
        let m = min(n, storage.count - head)
        storage.withUnsafeBufferPointer { src in
            if m > 0 { dst.update(from: src.baseAddress! + head, count: m) }
        }
        head += m
        compact()
        return m
    }

    func removeAll() {
        lock.lock(); defer { lock.unlock() }
        storage.removeAll(); head = 0
    }

    private func compact() {
        if head > 48_000 { storage.removeFirst(head); head = 0 }
    }
}

/// Microphone → 16 kHz → BeatriceStreamer (on a serial inference queue) → 24 kHz → speaker.
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
        public var ioBufferDuration = 0.0
        public var hardwareSampleRate = 0.0
        public var inputLatency = 0.0
        public var outputLatency = 0.0
        public var chunkMs = 0.0
        public var contextMs = 0.0
        public var lookaheadMs = 0.0
        /// rough end-to-end estimate: I/O buffers + hardware latency + chunk + look-ahead
        /// + inference + one chunk of output jitter buffer
        public var estimatedLatencyMs = 0.0
        public var lastError: String?
        public init() {}
    }

    private let engine = AVAudioEngine()
    private var sourceNode: AVAudioSourceNode?
    private var sinkNode: AVAudioSinkNode?
    private var converter: AVAudioConverter?
    private var rawFormat: AVAudioFormat?
    /// microphone samples at the hardware rate (mono), filled on the audio thread
    private let rawFIFO = FloatFIFO()
    private let inputFIFO = FloatFIFO()
    private let outputFIFO = FloatFIFO()
    private let queue = DispatchQueue(label: "BeatriceRealtimeEngine.inference", qos: .userInteractive)
    private let stateLock = NSLock()
    private var busy = false
    private var primed = false
    private var stats = Stats()
    private var timings: [Double] = []
    private var streamer: BeatriceStreamer?
    private static let inFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000,
                                                channels: 1, interleaved: false)!
    private static let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24_000,
                                                 channels: 1, interleaved: false)!

    public init() {}

    public func snapshot() -> Stats {
        stateLock.lock(); defer { stateLock.unlock() }
        return stats
    }

    /// - Parameter useSpeaker: iOS only; route output to the built-in speaker (feedback risk).
    public func start(streamer: BeatriceStreamer, preferredIOBuffer: TimeInterval = 0.005,
                      useSpeaker: Bool = false) throws {
        stop()
        streamer.reset()
        self.streamer = streamer
        rawFIFO.removeAll()
        inputFIFO.removeAll()
        outputFIFO.removeAll()
        stateLock.lock()
        stats = Stats()
        timings = []
        busy = false
        primed = false
        stats.chunkMs = Double(streamer.chunk) * 10
        stats.contextMs = Double(streamer.context) * 10
        stats.lookaheadMs = Double(streamer.lookahead) * 10
        stateLock.unlock()

        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement,
                                options: useSpeaker ? [.defaultToSpeaker] : [])
        try? session.setPreferredSampleRate(48_000)
        try? session.setPreferredIOBufferDuration(preferredIOBuffer)
        try session.setActive(true)
        #endif

        let input = engine.inputNode
        let hwFormat = input.outputFormat(forBus: 0)
        guard hwFormat.sampleRate > 0, hwFormat.channelCount > 0 else {
            throw BeatriceError.unsupported("マイク入力が見つかりません。")
        }
        try configureInput(sampleRate: hwFormat.sampleRate)
        // AVAudioSinkNode receives input at the I/O buffer size (installTap may deliver ~100 ms blocks)
        let channels = Int(hwFormat.channelCount)
        let interleaved = hwFormat.isInterleaved
        let sink = AVAudioSinkNode { [weak self] (_, frameCount, abl) -> OSStatus in
            guard let self else { return noErr }
            let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: abl))
            guard let raw = buffers[0].mData else { return noErr }
            let p = raw.assumingMemoryBound(to: Float.self)
            let n = Int(frameCount)
            if interleaved && channels > 1 {
                var mono = [Float](repeating: 0, count: n)
                for i in 0..<n { mono[i] = p[i * channels] }
                self.rawFIFO.append(mono)
            } else {
                self.rawFIFO.append(p, n)
            }
            self.kick()
            return noErr
        }
        sinkNode = sink
        engine.attach(sink)
        engine.connect(input, to: sink, format: hwFormat)

        let source = AVAudioSourceNode(format: Self.outFormat) { [weak self] (_, _, frameCount, abl) -> OSStatus in
            guard let self else { return noErr }
            let buffers = UnsafeMutableAudioBufferListPointer(abl)
            guard let raw = buffers[0].mData else { return noErr }
            self.render(raw.assumingMemoryBound(to: Float.self), Int(frameCount))
            return noErr
        }
        sourceNode = source
        engine.attach(source)
        engine.connect(source, to: engine.mainMixerNode, format: Self.outFormat)
        engine.prepare()
        try engine.start()

        stateLock.lock()
        stats.running = true
        #if os(iOS)
        stats.ioBufferDuration = session.ioBufferDuration
        stats.hardwareSampleRate = session.sampleRate
        stats.inputLatency = session.inputLatency
        stats.outputLatency = session.outputLatency
        #else
        stats.ioBufferDuration = Double(256) / max(hwFormat.sampleRate, 1)
        stats.hardwareSampleRate = hwFormat.sampleRate
        #endif
        updateLatencyLocked()
        stateLock.unlock()
    }

    public func stop() {
        if engine.isRunning { engine.stop() }
        if let sink = sinkNode {
            engine.detach(sink)
            sinkNode = nil
        }
        if let source = sourceNode {
            engine.detach(source)
            sourceNode = nil
        }
        stateLock.lock(); stats.running = false; stateLock.unlock()
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        #endif
    }

    /// Output render: waits until one chunk is buffered (jitter buffer), then plays continuously;
    /// an empty buffer after that counts as an underrun.
    func render(_ dst: UnsafeMutablePointer<Float>, _ n: Int) {
        let need = streamer?.outputChunkSamples ?? 0
        stateLock.lock()
        var primed = self.primed
        stateLock.unlock()
        if !primed, outputFIFO.count >= need {
            primed = true
            stateLock.lock(); self.primed = true; stateLock.unlock()
        }
        var got = 0
        if primed { got = outputFIFO.read(into: dst, n) }
        if got < n {
            dst.advanced(by: got).initialize(repeating: 0, count: n - got)
            if primed { stateLock.lock(); stats.underruns += 1; stateLock.unlock() }
        }
    }

    // MARK: - test hooks (drive the pipeline without audio hardware)

    func prepareForTesting(streamer: BeatriceStreamer, inputSampleRate: Double) throws {
        streamer.reset()
        self.streamer = streamer
        rawFIFO.removeAll(); inputFIFO.removeAll(); outputFIFO.removeAll()
        try configureInput(sampleRate: inputSampleRate)
        stateLock.lock()
        stats = Stats(); timings = []; busy = false; primed = false
        stateLock.unlock()
    }

    /// Feeds microphone samples (at the configured input rate) as the sink node would.
    func captureForTesting(_ samples: [Float]) {
        rawFIFO.append(samples)
        kick()
    }

    func waitUntilIdle() {
        while true {
            queue.sync {}
            stateLock.lock(); let idle = !busy; stateLock.unlock()
            if idle && rawFIFO.count == 0 && inputFIFO.count < (streamer?.inputChunkSamples ?? 1) { return }
            Thread.sleep(forTimeInterval: 0.005)
        }
    }

    private func configureInput(sampleRate: Double) throws {
        guard let mono = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
                                       channels: 1, interleaved: false),
              let conv = AVAudioConverter(from: mono, to: Self.inFormat) else {
            throw BeatriceError.unsupported("入力 \(sampleRate) Hz を 16kHz に変換できません。")
        }
        rawFormat = mono
        converter = conv
    }

    /// Runs on the inference queue: hardware-rate samples -> 16 kHz.
    private func convertPending() {
        guard let converter, let rawFormat else { return }
        let n = rawFIFO.count
        guard n > 0, let inBuf = AVAudioPCMBuffer(pcmFormat: rawFormat, frameCapacity: AVAudioFrameCount(n)),
              let inData = inBuf.floatChannelData?[0] else { return }
        let got = rawFIFO.read(into: inData, n)
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
            inputFIFO.append(data, Int(out.frameLength))
        }
    }

    private func kick() {
        stateLock.lock()
        if busy { stateLock.unlock(); return }
        busy = true
        stateLock.unlock()
        queue.async { [weak self] in self?.drain() }
    }

    private func drain() {
        defer { stateLock.lock(); busy = false; stateLock.unlock() }
        guard let streamer else { return }
        let n = streamer.inputChunkSamples
        while true {
            convertPending()
            // bound the backlog: keep at most two chunks waiting
            let dropped = inputFIFO.count > 3 * n ? inputFIFO.trim(keep: 2 * n) / n : 0
            if dropped > 0 { stateLock.lock(); stats.droppedChunks += dropped; stateLock.unlock() }
            guard let chunk = inputFIFO.pop(n) else { return }
            let t0 = DispatchTime.now().uptimeNanoseconds
            do {
                let y = try streamer.process(chunk)
                outputFIFO.append(y)
                // bound output latency against clock drift: keep at most three chunks queued
                outputFIFO.trim(keep: 3 * streamer.outputChunkSamples)
            } catch {
                stateLock.lock(); stats.lastError = error.localizedDescription; stateLock.unlock()
            }
            let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
            stateLock.lock()
            timings.append(ms)
            if timings.count > 200 { timings.removeFirst(timings.count - 200) }
            let sorted = timings.sorted()
            stats.inferenceMedianMs = sorted[sorted.count / 2]
            stats.inferenceMaxMs = max(stats.inferenceMaxMs, ms)
            stats.chunks += 1
            updateLatencyLocked()
            stateLock.unlock()
        }
    }

    private func updateLatencyLocked() {
        stats.estimatedLatencyMs = (2 * stats.ioBufferDuration + stats.inputLatency + stats.outputLatency) * 1000
            + stats.chunkMs * 2 + stats.lookaheadMs + stats.inferenceMedianMs
    }
}
