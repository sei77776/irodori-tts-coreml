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
    private var converter: AVAudioConverter?
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
        guard let conv = AVAudioConverter(from: hwFormat, to: Self.inFormat) else {
            throw BeatriceError.unsupported("入力形式 \(hwFormat) を 16kHz に変換できません。")
        }
        converter = conv
        input.installTap(onBus: 0, bufferSize: 256, format: hwFormat) { [weak self] buffer, _ in
            self?.capture(buffer)
        }

        let outputSamplesPerChunk = streamer.outputChunkSamples
        let source = AVAudioSourceNode(format: Self.outFormat) { [weak self] (_, _, frameCount, abl) -> OSStatus in
            guard let self else { return noErr }
            let buffers = UnsafeMutableAudioBufferListPointer(abl)
            guard let raw = buffers[0].mData else { return noErr }
            let dst = raw.assumingMemoryBound(to: Float.self)
            let n = Int(frameCount)
            self.stateLock.lock()
            var primed = self.primed
            self.stateLock.unlock()
            if !primed, self.outputFIFO.count >= outputSamplesPerChunk {
                primed = true
                self.stateLock.lock(); self.primed = true; self.stateLock.unlock()
            }
            var got = 0
            if primed { got = self.outputFIFO.read(into: dst, n) }
            if got < n {
                dst.advanced(by: got).initialize(repeating: 0, count: n - got)
                if primed {
                    self.stateLock.lock(); self.stats.underruns += 1; self.stateLock.unlock()
                }
            }
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
        engine.inputNode.removeTap(onBus: 0)
        if let source = sourceNode {
            engine.detach(source)
            sourceNode = nil
        }
        stateLock.lock(); stats.running = false; stateLock.unlock()
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        #endif
    }

    private func capture(_ buffer: AVAudioPCMBuffer) {
        guard let converter else { return }
        let ratio = Self.inFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio + 32)
        guard let out = AVAudioPCMBuffer(pcmFormat: Self.inFormat, frameCapacity: capacity) else { return }
        var supplied = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if supplied { status.pointee = .noDataNow; return nil }
            supplied = true
            status.pointee = .haveData
            return buffer
        }
        if let data = out.floatChannelData?[0], out.frameLength > 0 {
            inputFIFO.append(data, Int(out.frameLength))
        }
        kick()
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
